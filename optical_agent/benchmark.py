"""Evaluate orchestration and observed facts separately from visual accuracy."""

import json
import time

import numpy as np

from agent import OpticalAgent
from metrics import match_counts
from optical_agent.llm import CloudClient, ScriptedClient
from optical_agent.reports import build_report, evidence_map
from optical_agent.runtime import AgentRuntime
from optical_agent.state import Session
from optical_agent.tools import TOOLS, ToolContext, execute_tool


class FixtureDetector:
    """Harness fixture, never a reported CV model."""
    dataset = 'sodd'
    threshold = 0.33

    def predict(self, image):
        if float(image.mean()) < 8 or float(image.mean()) > 245:
            return []
        return [{'class_id': 1, 'class_name': 'propeller', 'box': [20, 20, 40, 40],
                 'score': 0.9, 'estimated_box_precision': 0.8}]


def score_task(task, result, context, system):
    tools = [t for t in result['trace'] if t.get('type') == 'tool']
    allowed = set(task['allowed_tools']) if system == 'fine' else {'analyze_image_full_pipeline', 'retrieve_knowledge'}
    evidence_ids = {e['evidence_id'] for e in result['evidence']}
    observed = evidence_map(context)
    fact_match = all(e['data'] == observed[e['evidence_id']]['data'] for e in result['evidence'])
    has_result = task['required_evidence'] in evidence_ids
    source_ok = not task['requires_citations'] or bool(result['citations'])
    valid_arguments = all(t['ok'] for t in tools)
    relevant_tools = all(t['tool'] in allowed for t in tools)
    selected = result.get('vision_result')
    overrides = int(selected is not None and selected != context.reports.get('original'))
    completion = (result['execution_status'] == 'completed' and has_result and fact_match and source_ok and not overrides)
    return {'tool_selection_correct': bool(relevant_tools and has_result),
            'argument_correct': bool(valid_arguments), 'task_completed': bool(completion),
            'valid_tool_calls': sum(t['ok'] for t in tools),
            'invalid_tool_calls': sum(not t['ok'] for t in tools),
            'unregistered_tool_executions': sum(t['ok'] and t['tool'] not in TOOLS for t in tools),
            'reliability_overrides': overrides, 'fact_match': bool(fact_match)}


def fixed_run(session, task):
    start = time.monotonic()
    # 对照的是已有固定视觉流程，不赋予它不存在的RAG能力。
    report, views = OpticalAgent(session.context.detector).inspect([session.context.images['original']])
    ctx = session.context
    ctx.qualities.update(original=report['quality_before'], corrected=report['quality_after'])
    ctx.images['corrected'] = views['corrected']
    ctx.images.update({f'report_{key}': value for key, value in views.items()})
    ctx.detections.update(original=report['detections_original'], corrected=report['detections_corrected'])
    ctx.reports['original'] = report
    result = build_report(ctx, {'evidence_ids': [task['required_evidence']], 'citation_ids': []},
                          'completed', [], {'available': False}, 'fixed', 'existing_vision_pipeline')
    result.update(latency_ms=round((time.monotonic()-start)*1000, 2), model_calls=0, tool_calls=0)
    return result


def aggregate(rows, system, backend, vision_backend):
    mean = lambda key: float(np.mean([row['score'][key] for row in rows]))
    latencies = [row['latency_ms'] for row in rows]
    call_counts = [row['tool_calls'] for row in rows]
    token_counts = [row['total_tokens'] for row in rows if row['tokens_available']]
    argument_accuracy = (sum(row['score']['valid_tool_calls'] for row in rows)/sum(call_counts)
                         if sum(call_counts) else None)
    metrics = {'task_count': len(rows), 'tool_selection_accuracy': mean('tool_selection_correct') if system != 'fixed' else None,
               'argument_accuracy': argument_accuracy if system != 'fixed' else None,
               'task_completion_rate': mean('task_completed'),
               'invalid_tool_call_rate': sum(row['score']['invalid_tool_calls'] for row in rows)/max(sum(call_counts), 1),
               'unregistered_tool_executions': sum(row['score']['unregistered_tool_executions'] for row in rows),
               'reliability_overrides': sum(row['score']['reliability_overrides'] for row in rows),
               'latency_mean_ms': float(np.mean(latencies)), 'latency_p95_ms': float(np.percentile(latencies, 95)),
               'tool_calls_mean': float(np.mean(call_counts)), 'tool_calls_p95': float(np.percentile(call_counts, 95)),
               'tokens_mean': float(np.mean(token_counts)) if token_counts else None,
               'tokens_p95': float(np.percentile(token_counts, 95)) if token_counts else None,
               'token_usage_task_count': len(token_counts),
               'by_kind': {kind: {'count': sum(r['kind'] == kind for r in rows),
                   'completion_rate': float(np.mean([r['score']['task_completed'] for r in rows if r['kind'] == kind]))}
                   for kind in sorted({r['kind'] for r in rows})}}
    comparable = [r for r in rows if r['kind'] != 'explanation']
    metrics['visual_common_task_completion_rate'] = float(np.mean([r['score']['task_completed'] for r in comparable]))
    metrics['live_targets_met'] = (backend == 'live' and system != 'fixed'
        and metrics['tool_selection_accuracy'] >= .90 and (metrics['argument_accuracy'] or 0) >= .95
        and metrics['task_completion_rate'] >= .85 and not metrics['unregistered_tool_executions'] and not metrics['reliability_overrides'])
    if vision_backend == 'sodd':
        selected = [r for r in rows if r.get('cv_counts') is not None]
        if selected:
            totals = {key: sum(r['cv_counts'][key] for r in selected) for key in ('tp','fp','fn')}
            metrics['cv_reliable_only'] = {**totals, 'recall': totals['tp']/max(totals['tp']+totals['fn'],1),
                'fp_per_task': totals['fp']/len(selected), 'rejection_rate': float(np.mean([r['vision_status'] != 'reliable' for r in selected])),
                'note': 'Repeated task samples; not a new independent scene benchmark.'}
    return metrics


def run_benchmark(tasks, detector, image_loader, retriever, backend='scripted', vision_backend='fixture',
                  systems=('fixed','coarse','fine'), repeats=3):
    output = {'backend': backend, 'vision_backend': vision_backend, 'repeats': repeats,
              'qualification': ('scripted validates runtime and scoring only; it is not LLM accuracy'
                                if backend == 'scripted' else 'live provider tool routing evaluation'),
              'metric_definition': 'Completion requires expected observed evidence, exact fact match, citations when requested, and unchanged vision gate.',
              'cost_definition': 'Task totals include setup and final turns; per-turn costs and traces are retained separately.',
              'systems': {}}
    for system in systems:
        rows = []
        for task in tasks:
            for repetition in range(repeats):
                image, truth = image_loader(task)
                session = Session(ToolContext(detector, image, retriever))
                client = CloudClient() if backend == 'live' else ScriptedClient()
                runtime = AgentRuntime(client, tool_mode='coarse' if system == 'coarse' else 'fine')
                setup = []
                for message in task.get('setup', []):
                    if system == 'fixed':
                        setup_result = fixed_run(session, dict(task, required_evidence='comparison:original'))
                    else:
                        setup_result = runtime.run(session, message)
                    setup_ids = [e['evidence_id'] for e in setup_result.get('evidence', [])]
                    setup_trace = setup_result.get('trace', [])
                    setup_usage = setup_result.get('usage') or {}
                    setup_tools = [t for t in setup_trace if t.get('type') == 'tool']
                    setup.append({'execution_status': setup_result['execution_status'], 'latency_ms': setup_result['latency_ms'],
                                  'evidence_ids': setup_ids,
                                  'evidence_ready': task.get('setup_required_evidence', 'comparison:original') in setup_ids,
                                  'tool_calls': setup_result.get('tool_calls', 0),
                                  'model_calls': setup_result.get('model_calls', 0),
                                  'tokens_available': setup_usage.get('available', False),
                                  'total_tokens': setup_usage.get('total_tokens'),
                                  'valid_tool_calls': sum(t['ok'] for t in setup_tools),
                                  'invalid_tool_calls': sum(not t['ok'] for t in setup_tools),
                                  'relevant_tools': all(t['tool'] in {'assess_image_quality', 'correct_image_exposure',
                                      'detect_objects', 'compare_detection_evidence', 'analyze_image_full_pipeline'} for t in setup_tools),
                                  'trace': setup_trace})
                result = fixed_run(session, task) if system == 'fixed' else runtime.run(session, task['message'])
                score = score_task(task, result, session.context, system)
                if any(s['execution_status'] != 'completed' or not s['evidence_ready'] for s in setup):
                    score['task_completed'] = False
                score['tool_selection_correct'] = score['tool_selection_correct'] and all(s['relevant_tools'] for s in setup)
                score['valid_tool_calls'] += sum(s['valid_tool_calls'] for s in setup)
                score['invalid_tool_calls'] += sum(s['invalid_tool_calls'] for s in setup)
                score['argument_correct'] = score['argument_correct'] and all(not s['invalid_tool_calls'] for s in setup)
                tokens_available = result['usage'].get('available', False) and all(s['tokens_available'] for s in setup)
                total_tokens = (result['usage']['total_tokens'] + sum(s['total_tokens'] for s in setup)) if tokens_available else None
                row = {'task_id': task['id'], 'kind': task['kind'], 'split': task['split'], 'repetition': repetition,
                       'source_index': task['source_index'], 'image_variant': task['image_variant'],
                       'execution_status': result['execution_status'], 'vision_status': result['vision_status'],
                       'score': score, 'latency_ms': result['latency_ms'] + sum(s['latency_ms'] for s in setup),
                       'tool_calls': result['tool_calls'] + sum(s['tool_calls'] for s in setup),
                       'model_calls': result['model_calls'] + sum(s['model_calls'] for s in setup),
                       'final_turn_latency_ms': result['latency_ms'], 'final_turn_tool_calls': result['tool_calls'],
                       'final_turn_model_calls': result['model_calls'], 'final_turn_usage': result['usage'],
                       'tokens_available': bool(tokens_available), 'total_tokens': total_tokens, 'trace': result['trace'],
                       'evidence_ids': [e['evidence_id'] for e in result['evidence']],
                       'citation_ids': [c['citation_id'] for c in result['citations']], 'setup': setup}
                if truth is not None and result['vision_result'] is not None:
                    vision = result['vision_result']
                    predictions = vision['accepted_detections'] if vision['status'] == 'reliable' else []
                    row['cv_counts'] = match_counts([predictions], [truth], detector.threshold)
                rows.append(row)
        output['systems'][system] = {'metrics': aggregate(rows, system, backend, vision_backend), 'tasks': rows}
    return output
