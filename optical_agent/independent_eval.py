"""V2 oracle scoring. Expected labels never enter the runtime or model messages."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from metrics import match_counts
from optical_agent.diagnostics import redact
from optical_agent.llm import CloudClient, ScriptedClient
from optical_agent.reports import evidence_map
from optical_agent.runtime import AgentRuntime, SYSTEM
from optical_agent.state import Session
from optical_agent.tasks import INTENT_SYSTEM
from optical_agent.tools import TOOLS, ToolContext, get_tool_descriptions


FAILURE_NAMES = {
    'intent': '任务理解错误', 'image_reference': '图片指向错误',
    'tool_selection': '工具选择错误', 'parameters': '参数或前提错误',
    'computation': '计算失败', 'report': '报告缺项',
    'api': 'API失败', 'budget': '预算停止', 'parsing': '未有效解析',
    'setup': '准备轮失败', 'facts': '事实不一致',
}


def digest(value):
    raw = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def score_turn(expected, result, context, *, setup_ok=True, original_hash=None):
    """Read only independent labels + observed results, never contract requirements."""
    contract = result.get('task_contract')
    parsed = isinstance(contract, dict) and contract.get('task_type') is not None
    intent_ok = parsed and contract['task_type'] in expected['task_types']
    image_ok = parsed and (expected['image_ids'] is None or contract.get('image_id') in expected['image_ids'])
    status_ok = result.get('task_status') in expected['task_statuses']
    observed = evidence_map(context)
    evidence = result.get('evidence', [])
    selected = {e.get('evidence_id') for e in evidence}
    required = set(expected['required_evidence'])
    evidence_ok = required <= selected and required <= observed.keys()
    prerequisites = set()
    if 'comparison:original' in required:
        prerequisites.update(['quality:original', 'quality:corrected', 'correction:original',
                              'detections:original', 'detections:corrected'])
    if 'correction:original' in required or 'comparison:original' in required:
        evidence_ok = evidence_ok and 'corrected' in context.images and bool(context.corrections.get('original'))
    evidence_ok = evidence_ok and prerequisites <= observed.keys()
    facts_ok = all(e.get('evidence_id') in observed and
                   e.get('data') == observed[e['evidence_id']]['data'] for e in evidence)
    citations = result.get('citations', [])
    citations_ok = all(c.get('citation_id') in context.citations and
                       c == context.citations[c['citation_id']] for c in citations)
    citations_ok = citations_ok and (not expected['requires_citations'] or bool(citations))
    tools = [t for t in result.get('trace', []) if t.get('type') == 'tool']
    allowed = set(expected['allowed_tools'])
    route_ok = all(t['tool'] in allowed for t in tools)
    relevant_executions = all(not t.get('executed', t.get('ok')) or t['tool'] in allowed for t in tools)
    invalid_arguments = sum(not t.get('arguments_valid', t.get('ok', False))
                            or t.get('error_category') in {'task_constraint', 'prerequisite'} for t in tools)
    cache_ok = not expected.get('cache_only') or all(t.get('cached') for t in tools if t.get('ok'))
    original_ok = original_hash is None or digest(context.images['original'].tobytes()) == original_hash
    vision = result.get('vision_result')
    overrides = int(vision is not None and vision != context.reports.get('original'))
    if vision is None and result.get('vision_status') is not None:
        overrides += 1
    if vision is not None and result.get('vision_status') != vision.get('status'):
        overrides += 1
    if result.get('task_status') in {'needs_clarification', 'unsupported'}:
        status_ok = status_ok and bool(result.get('answer', '').strip()) and not tools and not evidence
    completion = bool(intent_ok and image_ok and status_ok and evidence_ok and facts_ok and citations_ok
                      and setup_ok and cache_ok and original_ok and not overrides and relevant_executions)
    execution = result.get('execution_status')
    failures = []
    if execution in {'request_budget_exceeded', 'budget_exceeded'}:
        failures.append('budget')
    elif execution in {'network_error', 'authentication_error', 'insufficient_balance', 'missing_api_key'}:
        failures.append('api')
    elif execution == 'invalid_response' and (result.get('error_details') or {}).get('stage') in {
            'api_request', 'response_decode', 'connection'}:
        failures.append('api')
    if not parsed:
        failures.append('parsing')
    else:
        if not intent_ok:
            failures.append('intent')
        if not image_ok:
            failures.append('image_reference')
    if not route_ok:
        failures.append('tool_selection')
    if invalid_arguments:
        failures.append('parameters')
    if any(t.get('error_category') == 'computation' for t in tools):
        failures.append('computation')
    if not setup_ok:
        failures.append('setup')
    if not evidence_ok or not citations_ok or not status_ok or not cache_ok:
        failures.append('report')
    if not facts_ok or overrides or not original_ok:
        failures.append('facts')
    return {'parsed': bool(parsed), 'intent_correct': bool(intent_ok), 'image_correct': bool(image_ok),
            'tool_selection_correct': bool(parsed and route_ok and evidence_ok and status_ok),
            'evidence_correct': bool(evidence_ok), 'citations_correct': bool(citations_ok),
            'fact_match': bool(facts_ok), 'task_completed': completion, 'setup_ok': bool(setup_ok),
            'original_preserved': original_ok, 'cache_correct': bool(cache_ok),
            'tool_call_count': len(tools), 'invalid_argument_count': int(invalid_arguments),
            'unregistered_tool_executions': sum(t.get('executed', t.get('ok')) and t['tool'] not in TOOLS for t in tools),
            'reliability_overrides': overrides, 'failure_categories': list(dict.fromkeys(failures))}


def aggregate_cases(rows, planned_count):
    turns = [t for r in rows for t in r['turns']]
    parsed = [t for t in turns if t['score']['parsed']]
    calls = sum(t['score']['tool_call_count'] for t in turns)
    usage_turns = [t for t in turns if t['result'].get('usage', {}).get('available')]
    rate = lambda key, group: sum(t['score'][key] for t in group) / len(group) if group else None
    distribution = lambda values: {'mean': float(np.mean(values)), 'p95': float(np.percentile(values, 95))} if values else {'mean': None, 'p95': None}
    metrics = {
        'planned_cases': planned_count, 'executed_cases': len(rows), 'executed_turns': len(turns),
        'parsing_coverage': len(parsed) / len(turns) if turns else 0.,
        'intent_accuracy_on_parsed': rate('intent_correct', parsed),
        'image_accuracy_on_parsed': rate('image_correct', parsed),
        'intent_accuracy_all_requests': rate('intent_correct', turns),
        'image_accuracy_all_requests': rate('image_correct', turns),
        'tool_selection_accuracy': rate('tool_selection_correct', turns),
        'argument_accuracy': 1 - sum(t['score']['invalid_argument_count'] for t in turns) / calls if calls else None,
        'independent_turn_completion_rate': rate('task_completed', turns),
        'independent_case_completion_rate': sum(r['completed'] for r in rows) / len(rows) if rows else 0.,
        'planned_case_completion_rate': sum(r['completed'] for r in rows) / planned_count if planned_count else 0.,
        'unregistered_tool_executions': sum(t['score']['unregistered_tool_executions'] for t in turns),
        'reliability_overrides': sum(t['score']['reliability_overrides'] for t in turns),
        'fact_violations': sum(not t['score']['fact_match'] or not t['score']['original_preserved'] for t in turns),
        'request_attempts': sum(t['result'].get('request_attempts', 0) for t in turns),
        'model_calls': distribution([sum(t['result'].get('model_calls', 0) for t in r['turns']) for r in rows]),
        'tool_calls': distribution([sum(t['result'].get('tool_calls', 0) for t in r['turns']) for r in rows]),
        'latency_ms': distribution([sum(t['result'].get('latency_ms', 0) for t in r['turns']) for r in rows]),
        'tokens': distribution([sum(t['result']['usage']['total_tokens'] for t in r['turns']) for r in rows
                               if all(t['result'].get('usage', {}).get('available') for t in r['turns'])]),
        'known_token_usage': {k: sum(t['result']['usage'].get(k, 0) for t in usage_turns)
                             for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
        'token_usage_turns': len(usage_turns),
        'failures': {k: sum(k in t['score']['failure_categories'] for t in turns) for k in FAILURE_NAMES},
        'formal_acceptance': False,
        'by_kind': {k: {'cases': sum(r['kind'] == k for r in rows),
                        'completed': sum(r['completed'] for r in rows if r['kind'] == k)} for k in sorted({r['kind'] for r in rows})},
    }
    metrics['pilot_targets_met'] = bool((metrics['tool_selection_accuracy'] or 0) >= .9
        and (metrics['argument_accuracy'] or 0) >= .95 and metrics['planned_case_completion_rate'] >= .85
        and not metrics['unregistered_tool_executions'] and not metrics['reliability_overrides'])
    return metrics


def write_json(path, value, secrets=()):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(redact(value, secrets), handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def freeze_manifest(run_dir, task_file, model):
    """Freeze ALL labels, including held-out ones, before the first model request."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    task_hash = digest(Path(task_file).read_bytes())
    path = run_dir / 'manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest['task_sha256'] != task_hash or manifest['model'] != model:
            raise ValueError('题目或模型与本运行的冻结记录不一致，不能覆盖原记录')
    else:
        manifest = {'schema_version': 2, 'frozen_at': datetime.now(timezone.utc).isoformat(),
                    'task_sha256': task_hash, 'model': model,
                    'labels_sent_to_model': False, 'heldout_used_for_tuning': False}
        write_json(path, manifest)
        (run_dir / 'frozen_tasks.json').write_bytes(Path(task_file).read_bytes())
    return manifest


def version_metadata(root):
    root = Path(root)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    files = ['runtime.py', 'tasks.py', 'tools.py', 'llm.py', 'independent_eval.py', 'request_budget.py', 'reports.py']
    return {'code_commit': commit,
            'source_sha256': {f: digest((root / 'optical_agent' / f).read_bytes()) for f in files},
            'prompt_sha256': digest({'intent': INTENT_SYSTEM, 'dispatch': SYSTEM, 'tools': get_tool_descriptions()})}


def batch_stop_reason(result):
    status = result.get('execution_status')
    if status in {'authentication_error', 'insufficient_balance', 'missing_api_key'}:
        return '账户或配置错误，停止批次'
    if status == 'request_budget_exceeded':
        return '达到持久化 HTTP 请求上限'
    return None


def run_independent(cases, detector, image_loader, retriever, *, backend, vision_backend,
                    run_dir, task_file, root, phase='before', budget=None, client_factory=None, repeats=1):
    if backend == 'live' and budget is None:
        raise ValueError('真实评测必须传入持久化请求预算')
    if phase not in {'before', 'fix1', 'fix2', 'after', 'offline'}:
        raise ValueError('无效评测阶段')
    client_factory = client_factory or (lambda: CloudClient(request_budget=budget) if backend == 'live' else ScriptedClient())
    probe = client_factory()
    secrets = (getattr(probe, 'api_key', None),)
    manifest = freeze_manifest(run_dir, task_file, probe.model_name)
    metadata = version_metadata(root)
    tag = f'{phase}_{vision_backend}_{backend}'
    output_path = Path(run_dir) / (tag + '.json')
    rows, stop = [], None
    if output_path.exists():
        prior = json.loads(output_path.read_text(encoding='utf-8'))
        if prior['versions'] != metadata:
            raise ValueError('本阶段代码已改变，请使用下一修复阶段，不能覆盖前版本')
        if prior['selected_case_ids'] != [c['id'] for c in cases] or prior['repeats'] != repeats:
            raise ValueError('本阶段案例选择或重复次数改变，不能覆盖原结果')
        rows = prior['cases']
        stop = prior['stop_reason']
    output = {'schema_version': 2, 'backend': backend, 'vision_backend': vision_backend,
              'phase': phase, 'repeats': repeats, 'model': probe.model_name, 'versions': metadata,
              'task_sha256': manifest['task_sha256'], 'selected_case_ids': [c['id'] for c in cases],
              'qualification': '小样本调度验证；固定视觉夹具不是视觉精度评测；未运行保留验收题。',
              'cases': rows}
    done = {(r['task_id'], r['repetition']) for r in rows}
    consecutive_service_failures = 0
    for r in reversed(rows):
        if any(t['result'].get('execution_status') == 'network_error' for t in r['turns']):
            consecutive_service_failures += 1
        else:
            break

    def save():
        planned = len(cases) * repeats
        output.update(stop_reason=stop, metrics=aggregate_cases(rows, planned),
                      budget=budget.snapshot() if budget else None,
                      skipped=[{'task_id': c['id'], 'repetition': n, 'reason': stop or '尚未运行'}
                               for c in cases for n in range(repeats) if (c['id'], n) not in done])
        output['metrics']['pilot_targets_met'] &= backend == 'live'
        write_json(output_path, output, secrets)

    save()
    for case in cases:
        for repetition in range(repeats):
            if stop or (case['id'], repetition) in done:
                continue
            if budget and budget.snapshot()['remaining'] == 0:
                stop = '达到持久化 HTTP 请求上限'
                break
            image, truth, source = image_loader(case)
            original_hash = digest(image.tobytes())
            session = Session(ToolContext(detector, image, retriever))
            runtime = AgentRuntime(client_factory())
            row = {'task_id': case['id'], 'kind': case['kind'], 'split': case['split'],
                   'repetition': repetition, 'source_index': case['source_index'], 'source': source,
                   'image_sha256': original_hash, 'image_variant': case['image_variant'], 'turns': [], 'completed': False}
            service_failed, setup_ok = False, True
            for turn_index, turn in enumerate(case['turns']):
                # The ONLY case-derived field sent to AgentRuntime is the natural-language message.
                result = runtime.run(session, turn['message'])
                score = score_turn(turn['expected'], result, session.context, setup_ok=setup_ok, original_hash=original_hash)
                item = {'turn_index': turn_index, 'message': turn['message'], 'expected': turn['expected'],
                        'result': result, 'score': score,
                        'observed_evidence_ids': list(evidence_map(session.context))}
                if truth is not None:
                    counts = {}
                    for image_id in ('original', 'corrected'):
                        if image_id in session.context.detections:
                            counts[image_id] = match_counts([session.context.detections[image_id]], [truth], detector.threshold)
                    if result.get('vision_result') is not None:
                        vision = result['vision_result']
                        counts['accepted'] = match_counts([vision['accepted_detections'] if vision['status'] == 'reliable' else []], [truth], detector.threshold)
                    item['cv_counts'] = counts
                row['turns'].append(item)
                write_json(Path(run_dir) / tag / f"{case['id']}_{repetition}_turn{turn_index}.json", item, secrets)
                setup_ok = setup_ok and score['task_completed']
                service_failed |= result.get('execution_status') == 'network_error'
                stop = batch_stop_reason(result)
                if stop or (budget and budget.snapshot()['remaining'] == 0):
                    stop = stop or '达到持久化 HTTP 请求上限'
                    break
            row['completed'] = len(row['turns']) == len(case['turns']) and all(t['score']['task_completed'] for t in row['turns'])
            row['unrun_turns'] = list(range(len(row['turns']), len(case['turns'])))
            rows.append(row)
            done.add((case['id'], repetition))
            consecutive_service_failures = consecutive_service_failures + 1 if service_failed else 0
            if consecutive_service_failures >= 2:
                stop = '连续两个案例发生连接或服务不可用错误'
            save()
            print(json.dumps({'case': case['id'], 'completed': row['completed'],
                              'failures': [t['score']['failure_categories'] for t in row['turns']],
                              'budget': output['budget']}, ensure_ascii=False), flush=True)
    save()
    return output


def paired_comparison(before, after):
    """Compare the same oracle, vision mode, source pixels and repetition only."""
    if before['task_sha256'] != after['task_sha256'] or before['vision_backend'] != after['vision_backend']:
        raise ValueError('不同题目或视觉后端不可作前后对照')
    key = lambda r: (r['task_id'], r['repetition'], r['image_sha256'])
    previous = {key(r): r for r in before['cases']}
    pairs = [(previous[key(r)], r) for r in after['cases'] if key(r) in previous]
    return {'paired_cases': len(pairs), 'before_completed': sum(a['completed'] for a, b in pairs),
            'after_completed': sum(b['completed'] for a, b in pairs),
            'improved_ids': [b['task_id'] for a, b in pairs if not a['completed'] and b['completed']],
            'regressed_ids': [b['task_id'] for a, b in pairs if a['completed'] and not b['completed']]}
