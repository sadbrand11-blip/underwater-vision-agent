"""Independent development oracle + same-tools fixed policy, no labels in LLM input."""
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from optical_agent.adaptive_fixture import scripted_goal
from optical_agent.adaptive_goals import CLASSES, parse_goal
from optical_agent.adaptive_tools import AdaptiveState, SPECS, execute, pixel_hash, render_result, scope_id
from optical_agent.independent_eval import digest, write_json


class FixedVisualFixture:
    """Authored visual observations only. Not a detector accuracy benchmark."""
    dataset, threshold, calibration = 'sodd', .33, True

    def __init__(self, decline=False):
        self.decline = decline

    def predict(self, image):
        score = .4 if self.decline and image.mean() > 70 else .9
        return [{'class_name': name, 'class_id': cid, 'box': [12, 12, 28, 28], 'score': score,
                 'estimated_box_precision': score} for name, cid in [('pipe', 6), ('qr_codes', 5)]]


def fixed_workflow(session, message):
    """Declared fixed policy: try both bounded methods whenever correction is allowed/needed.

    Uses a deterministic parser for this restricted language. It has no LLM cost.
    Not an autonomous planning baseline and not a fair general language parser comparison.
    """
    if not hasattr(session, 'adaptive'):
        session.adaptive = AdaptiveState(session.context)
    state, start = session.adaptive, time.monotonic()
    before = dict(state.counts)
    first_event = len(state.events)
    goal = parse_goal(json.dumps(scripted_goal(message, state.pending), ensure_ascii=False), state, state.selected_image)
    state.begin(goal)
    ids, selected = [], goal.image_id
    def call(name, **args):
        observation = execute(state, name, args)
        if not observation['ok']:
            raise ValueError(observation['error'])
        oid = observation['data'].get('observation_id')
        if oid:
            ids.append(oid)
        return observation['data']['data'] if oid else observation['data']
    status = 'completed'
    if goal.disposition != 'ready':
        status = goal.disposition
        result = {'answer': goal.question, 'evidence': [], 'detections': [], 'target_count': None,
                  'vision_status': None, 'vision_result': None, 'citations': [], 'image_refs': ['original'],
                  'selected_image_id': 'original'}
    else:
        if goal.task_type == 'quality':
            call('assess_image_quality', image_id=selected)
        elif goal.task_type == 'detection' and not goal.require_reliability:
            call('detect_objects', image_id=selected)
        else:
            q = call('assess_image_quality', image_id='original')['quality']
            needs = goal.correction_policy == 'always' or (goal.correction_policy == 'if_needed' and q['exposure_state'] != 'normal')
            candidates = []
            if q['quality_pass'] is True and needs:
                for method in ('gamma_only', 'local_bounded'):
                    candidates.append(call('generate_exposure_candidate', image_id='original', method=method)['image_id'])
            if goal.task_type == 'correction':
                selected = candidates[-1] if candidates else 'original'
            elif q['quality_pass'] is not True:
                call('assess_reliability', image_id='original')
            else:
                call('detect_objects', image_id='original')
                for image_id in candidates:
                    call('assess_image_quality', image_id=image_id)
                    call('detect_objects', image_id=image_id)
                if candidates:
                    reports = call('compare_candidates', candidate_image_ids=candidates)['reports']
                    # Choose among program-approved selections; reliability remains exactly as measured.
                    def rank(report):
                        boxes = state.filtered(report['selected_image'])
                        return (report['status'] == 'reliable', sum(x['score'] for x in boxes))
                    selected = max(reports, key=rank)['selected_image']
                else:
                    call('assess_reliability', image_id='original')
            if goal.requires_citations:
                call('retrieve_knowledge', query='水下曝光信息丢失与目标拒识原因')
        # Make filtered detection evidence for whichever image is actually selected available.
        det_id = f'detections:{selected}:{scope_id(goal)}'
        if det_id in state.observations and det_id not in ids:
            ids.append(det_id)
        result = render_result(state, {'selected_image_id': selected, 'evidence_ids': ids,
                                     'citation_ids': list(state.citations) if goal.requires_citations else []})
        state.selected_image = selected
    events = copy.deepcopy(state.events[first_event:])
    result.update(goal_contract=goal.as_dict(), task_status=status, execution_status=status,
        execution_mode='fixed_workflow', model='none', model_calls=0, request_attempts=0, model_request_attempts=0,
        tool_calls=len(events), trace=events, initial_plan=[], plan_revisions=[],
        computation_counts={k: state.counts[k] - before[k] for k in state.counts},
        latency_ms=round((time.monotonic() - start) * 1000, 2),
        usage={'available': True, 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})
    return result


def score(expected, result, state, start_candidate=None, original_hash=None):
    """Never uses task_validation/contract requirements as a ground truth."""
    goal = result.get('goal_contract') or {}
    parsed = bool(goal.get('task_type'))
    intent = parsed and goal.get('task_type') in expected['types']
    wanted_classes = set(CLASSES) if expected['classes'] == 'all' else set(expected['classes']) if expected['classes'] != 'ignore' else None
    classes = wanted_classes is None or set(goal.get('target_classes', [])) == wanted_classes
    policy = expected['policy'] == 'ignore' or goal.get('correction_policy') == expected['policy']
    image_id = start_candidate if expected['image'] == 'last_candidate' else expected['image']
    image_ok = expected['image'] == 'ignore' or goal.get('image_id') == image_id
    status_ok = result.get('task_status') == expected['status']
    evidence = result.get('evidence', [])
    facts = bool(state) or not evidence
    selected_obs = {}
    for e in evidence:
        oid = e.get('evidence_id')
        original = state.observations.get(oid) if state else None
        facts &= original is not None and {k: v for k, v in e.items() if k != 'evidence_id'} == original
        if original:
            selected_obs[oid] = original
    chosen = result.get('selected_image_id')
    q = state.qualities.get('original') if state else None
    types = {o['type'] for o in selected_obs.values()}
    det = [o for o in selected_obs.values() if o['type'] == 'detections' and o['data']['image_id'] == chosen]
    expected_boxes = [x for x in state.raw_detections.get(chosen, []) if wanted_classes is None or x['class_name'] in wanted_classes] if state else []
    facts &= result.get('detections', []) == (expected_boxes if det else [])
    facts &= result.get('target_count') == (len(expected_boxes) if det else None)
    facts &= all(c.get('citation_id') in state.citations and c == state.citations[c['citation_id']]
                 for c in result.get('citations', [])) if state else not result.get('citations')
    reports = [o['data'] for o in selected_obs.values() if o['type'] == 'reliability']
    reports += [r for o in selected_obs.values() if o['type'] == 'comparison' for r in o['data']['reports'] if r['selected_image'] == chosen]
    visual = result.get('vision_result')
    facts &= (visual in reports if visual is not None else result.get('vision_status') is None)
    reliability_overrides = int(visual is not None and (visual not in reports or result.get('vision_status') != visual['status']))
    if q and q['quality_pass'] is not True and result.get('vision_status') == 'reliable':
        reliability_overrides += 1
    required = expected['evidence']
    if required == 'none':
        evidence_ok = not evidence and result.get('tool_calls') == 0
    elif required == 'quality':
        evidence_ok = any(o['type'] == 'quality' and o['data']['image_id'] == image_id for o in selected_obs.values()) and chosen == image_id
    elif required == 'correction':
        evidence_ok = 'quality' in types and (('correction' in types and chosen in state.candidates) or (q and (q['quality_pass'] is not True or q['exposure_state'] == 'normal')))
    elif required == 'detections':
        evidence_ok = bool(det) and chosen == image_id
    elif required == 'quality_failure':
        evidence_ok = {'quality', 'reliability'} <= types and result.get('vision_status') == 'quality_failure' and chosen == 'original'
    else:
        evidence_ok = 'quality' in types and bool(reports) and visual is not None
        if q and q['quality_pass'] is True:
            evidence_ok &= bool(det)
            for cid in (state.active_candidates if state else []):
                evidence_ok &= any(o['type'] == 'comparison' and any(r['candidate_image_id'] == cid for r in o['data']['reports']) for o in selected_obs.values())
        if expected.get('candidate_required'):
            evidence_ok &= bool(state and state.candidates and 'comparison' in types)
    if expected.get('candidate_required') and required == 'correction':
        evidence_ok &= bool(state and state.candidates)
    if expected.get('selected_original'):
        evidence_ok &= chosen == 'original'
    tools = [t for t in result.get('trace', []) if t['type'] == 'tool']
    permitted = set(SPECS)
    if required == 'quality':
        permitted = {'assess_image_quality'}
    elif required == 'detections':
        permitted = {'detect_objects', 'revise_plan'}
    elif required == 'correction':
        permitted = {'assess_image_quality', 'generate_exposure_candidate', 'revise_plan'}
    elif required == 'none':
        permitted = set()
    irrelevant = sum(t['tool'] not in permitted for t in tools)
    tools_ok = all(t['tool'] in permitted for t in tools)
    parameters_ok = all(t.get('arguments_valid') and (t.get('ok') or t.get('error_category') not in {'parameters', 'prerequisite'}) for t in tools)
    if expected.get('no_correction'):
        tools_ok &= not any(t['tool'] == 'generate_exposure_candidate' for t in tools)
    if expected.get('no_detection'):
        tools_ok &= not any(t['tool'] == 'detect_objects' for t in tools)
    integrity = bool(not state or original_hash is None or pixel_hash(state.images['original']) == original_hash)
    integrity &= all(c['source_image_id'] == 'original' and c['source_sha256'] == original_hash and
                     c['pixels_sha256'] == pixel_hash(state.images[k]) for k, c in state.candidates.items()) if state else True
    revisions = result.get('plan_revisions', [])
    revision_ok = len(revisions) <= 2 and all(r['observation_ids'] and all(x in state.observations for x in r['observation_ids']) for r in revisions)
    passed = all([intent, classes, policy, image_ok, status_ok, evidence_ok, facts, tools_ok, parameters_ok, integrity, revision_ok]) and reliability_overrides == 0
    api = result.get('execution_status') in {'authentication_error', 'insufficient_balance', 'network_error', 'missing_api_key', 'invalid_response'}
    checks = dict(intent=intent, classes=classes, policy=policy, image=image_ok, status=status_ok, evidence=bool(evidence_ok),
                  facts=bool(facts), tools=bool(tools_ok), parameters=bool(parameters_ok), integrity=bool(integrity), revisions=bool(revision_ok))
    return dict(passed=bool(passed), parsed=parsed, checks=checks, failure='api' if api else 'budget' if result.get('execution_status') in {'budget_exceeded', 'request_budget_exceeded'} else next((k for k, v in checks.items() if not v), None),
        irrelevant_calls=int(irrelevant), unregistered_executions=sum(t['tool'] not in SPECS and t.get('executed', False) for t in tools),
        reliability_overrides=reliability_overrides)


def freeze(root, tasks, inputs, client):
    filenames = ['agent.py', 'quality.py', 'detector.py', 'data.py', 'metrics.py', 'models/sodd_detector.pt',
                 'evaluate_adaptive.py', 'eval/adaptive_tasks.json']
    filenames += [str(p.relative_to(root)).replace('\\', '/') for p in (root / 'optical_agent').glob('*.py')]
    filenames += [str(p.relative_to(root)).replace('\\', '/') for p in (root / 'knowledge').rglob('*') if p.is_file()]
    # RAG sources can include README and reports; hash the actual declared source contents too.
    filenames += [x['path'] for x in json.loads((root / 'knowledge/sources.json').read_text(encoding='utf-8'))]
    return dict(files={name: digest((root / name).read_bytes()) for name in sorted(set(filenames))},
        tasks=digest(tasks), inputs=inputs, request_config={'model': client.model_name, 'base_url': getattr(client, 'base_url', None),
        'max_model_calls':10, 'max_tool_calls':12, 'max_seconds':180}, visual_config=asdict(__import__('quality').UNDERWATER_QUALITY_CONFIG))


def save(path, value):
    write_json(Path(path), value)


def summarize(records, planned, budget):
    rows = [turn for record in records for turn in record.get('turns', [])]
    def stats(values):
        return {'mean': round(float(np.mean(values)), 3), 'p95': round(float(np.percentile(values, 95)), 3)} if values else {'mean': None, 'p95': None}
    def system_stats(name):
        data = [row[name] for row in rows if name in row]
        scores = [r['score'] for r in data]
        tools = [t for r in data for t in r['result'].get('trace', []) if t['type'] == 'tool']
        n = len(data)
        return {'turns': n, 'passed': sum(s['passed'] for s in scores),
            'parse_coverage': sum(s['parsed'] for s in scores) / n if n else None,
            'goal_conformance': sum(all(s['checks'][k] for k in ('intent', 'classes', 'policy', 'image')) for s in scores) / n if n else None,
            'image_accuracy': sum(s['checks']['image'] for s in scores) / n if n else None,
            'irrelevant_calls': sum(s['irrelevant_calls'] for s in scores),
            'unregistered_executions': sum(s['unregistered_executions'] for s in scores),
            'reliability_overrides': sum(s['reliability_overrides'] for s in scores),
            'fact_failures': sum(not s['checks']['facts'] for s in scores),
            'tool_selection_rate': sum(t['tool'] in SPECS for t in tools) / len(tools) if tools else None,
            'parameter_rate': sum(bool(t.get('arguments_valid')) and t.get('error_category') not in {'parameters','prerequisite'} for t in tools) / len(tools) if tools else None,
            'plan_revisions': sum(len(r['result'].get('plan_revisions', [])) for r in data),
            'latency_ms': stats([r['result']['latency_ms'] for r in data]),
            'model_calls': stats([r['result']['model_calls'] for r in data]),
            'tool_calls': stats([r['result']['tool_calls'] for r in data]),
            'total_computations': stats([sum(r['result']['computation_counts'].values()) for r in data]),
            'tokens': stats([r['result']['usage']['total_tokens'] for r in data if r['result']['usage'].get('available')]),
            'token_total': sum(r['result']['usage']['total_tokens'] for r in data if r['result']['usage'].get('available')),
            'failures': [{'case': row['case'], 'repeat': row['repeat'], 'turn': row['turn'], 'category': row[name]['score']['failure']}
                         for row in rows if name in row and not row[name]['score']['passed']]}
    return {'formal_acceptance': False, 'scope': 'development_and_demo_only', 'planned_cases': planned,
        'recorded_cases': len(records), 'case_completion_rate': sum(bool(r.get('turns')) and all(t.get('adaptive', {}).get('score', {}).get('passed', False) for t in r['turns']) and r.get('status') == 'recorded' for r in records) / planned,
        'budget': budget, 'adaptive': system_stats('adaptive'), 'fixed_workflow': system_stats('fixed')}
