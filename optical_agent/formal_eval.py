"""Frozen held-out evaluation, durable case journal and deterministic acceptance.

The task oracle is used only after runtime execution. No prompt tuning happens here.
"""

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

from metrics import match_counts
from optical_agent.independent_eval import (aggregate_cases, digest, score_turn,
                                            version_metadata, write_json)
from optical_agent.llm import CloudClient
from optical_agent.reports import evidence_map
from optical_agent.runtime import AgentRuntime
from optical_agent.state import Session
from optical_agent.tools import ToolContext


HELD_IDS = [f'hold_{i:02d}' for i in range(1, 21)]
QUALIFICATION = ('20条预设保留任务的正式验收，重复3次；不是60条独立任务。'
                 '标签在调用前预设，尚未经过外部人工复核；调度成绩不代表识别精度。')


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_protocol(cases, backend, vision_backend, repeats):
    if (backend != 'live' or vision_backend != 'sodd' or repeats != 3
            or [c['id'] for c in cases] != HELD_IDS
            or any(c['split'] != 'test' for c in cases)
            or len({c['source_index'] for c in cases}) != 20
            or sum(len(c['turns']) for c in cases) != 26):
        raise ValueError('正式模式必须使用真实LLM、SODD、全部20条保留案例，重复3次（78轮问题）')


def file_hash(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def collect_bindings(root, task_file, cases, detector, dataset, image_loader, retriever, client):
    """Offline preflight: load every source/label, hash actual inference configuration."""
    root = Path(root)
    files = set(root.glob('*.py')) | set((root / 'optical_agent').glob('*.py'))
    files.update([Path(task_file), root / 'pyproject.toml', root / 'knowledge' / 'sources.json'])
    for source in json.loads((root / 'knowledge' / 'sources.json').read_text(encoding='utf-8')):
        files.add(root / source['path'])
    weight = Path(detector.model_path)
    files.add(weight)
    calibration = weight.with_suffix('.calibration.json')
    if calibration.exists():
        files.add(calibration)
    images = []
    for case in cases:
        image, truth, source = image_loader(case)
        raw, label = dataset.items[case['source_index']]
        if truth is None:
            raise ValueError('正式视觉预检缺少真实标注')
        images.append({'task_id': case['id'], 'source_index': case['source_index'],
                       'source': source, 'image_path': str(Path(raw).resolve()),
                       'label_path': str(Path(label).resolve()),
                       'image_file_sha256': file_hash(raw), 'label_file_sha256': file_hash(label),
                       'pixels_sha256': digest(image.tobytes()), 'truth_sha256': digest(truth),
                       'shape': list(image.shape), 'image_variant': case['image_variant']})
    from urllib.parse import urlparse
    endpoint = urlparse(client.base_url)
    if (endpoint.scheme != 'https' or endpoint.hostname != 'api.deepseek.com'
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
        raise ValueError('本次正式协议仅接受官方HTTPS DeepSeek地址，地址不得包含凭据')
    packages = {}
    for name in ['numpy', 'torch', 'torchvision', 'opencv-python', 'scikit-learn', 'requests', 'scikit-image']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'versions': version_metadata(root),
            'files': {str(p.resolve()): file_hash(p) for p in sorted(files)},
            'images': images, 'knowledge_chunks_sha256': digest(retriever.chunks),
            'retrieval_min_score': retriever.min_score,
            'quality_config': asdict(ToolContext(detector).quality_config),
            'detector': {'dataset': detector.dataset, 'classes': detector.classes,
                         'threshold': detector.threshold, 'device': str(detector.device),
                         'calibration_sha256': digest(detector.calibration)},
            'request_config': {'base_url': client.base_url, 'model': client.model_name,
                               'stream': False, 'max_tokens': 2048, 'response_format': 'json_object',
                               'thinking': 'disabled', 'max_transport_attempts_per_call': 2,
                               'model_calls_per_turn': 10, 'tool_calls_per_turn': 12,
                               'seconds_per_turn_checked_at_boundaries': 180},
            'environment': {'python': sys.version, 'platform': platform.platform(), 'packages': packages}}


@contextmanager
def evaluation_lock(run_dir):
    """One process owns a run; OS releases the lock on interruption."""
    path = Path(run_dir) / 'evaluation.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError('此运行编号已有评测进程，不能并发执行') from None
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def freeze_formal(run_dir, root, task_file, bindings, budget_limit, *, resume):
    path = Path(run_dir) / 'formal_manifest.json'
    if path.exists():
        if not resume:
            raise ValueError('运行编号已使用；请用 --resume 继续，不能覆盖或重新挑选结果')
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest['bindings'] != bindings or manifest['http_limit'] != budget_limit:
            raise ValueError('冻结内容或请求上限发生变化；拒绝继续此运行编号')
        if digest((Path(run_dir) / 'frozen_tasks.json').read_bytes()) != manifest['task_sha256']:
            raise ValueError('冻结题目损坏；拒绝继续')
        return manifest
    if resume or (Path(run_dir) / 'manifest.json').exists():
        raise ValueError('找不到正式冻结清单，或编号属于开发运行；不能复用')
    manifest = {'schema_version': 3, 'frozen_at': now(), 'baseline_behavior_version': '0.2.3',
                'task_sha256': file_hash(task_file), 'http_limit': budget_limit,
                'labels_sent_to_model': False, 'heldout_used_for_tuning': False,
                'bindings': bindings}
    # Retain small code/knowledge inputs so later documentation changes are reproducible.
    for name in bindings['files']:
        p = Path(name)
        if p.suffix not in {'.py', '.md', '.json', '.toml'}:
            continue
        if not p.resolve().is_relative_to(Path(root).resolve()):
            continue
        relative = p.resolve().relative_to(Path(root).resolve())
        copy = Path(run_dir) / 'frozen_inputs' / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(p.read_bytes())
    (Path(run_dir) / 'frozen_tasks.json').write_bytes(Path(task_file).read_bytes())
    write_json(path, manifest)
    return manifest


def formal_metrics(rows, cases, *, backend, vision_backend, repeats, frozen_valid=True):
    planned = {(c['id'], r): len(c['turns']) for r in range(3) for c in cases}
    keys = [(r['task_id'], r['repetition']) for r in rows]
    valid_rows = (len(keys) == len(set(keys)) and set(keys) <= set(planned)
                  and all(r.get('journal_status') in {'started', 'completed', 'interrupted'} for r in rows)
                  and all([t['turn_index'] for t in r['turns']] == list(range(len(r['turns'])))
                          and len(r['turns']) <= planned.get((r['task_id'], r['repetition']), 0)
                          for r in rows))
    cloud_observed = all(t['result'].get('execution_mode') == 'cloud'
                         and (t['result'].get('request_attempts', 0) > 0
                              or t['result'].get('execution_status') == 'request_budget_exceeded')
                         for r in rows for t in r['turns'])
    eligible = (backend == 'live' and vision_backend == 'sodd' and repeats == 3
                and [c['id'] for c in cases] == HELD_IDS and len(planned) == 60
                and sum(planned.values()) == 78 and frozen_valid and valid_rows and cloud_observed)
    complete = eligible and set(keys) == set(planned) and all(
        r.get('journal_status') == 'completed' and len(r['turns']) == planned[(r['task_id'], r['repetition'])]
        for r in rows)
    # Do not trust a persisted completed flag in isolation from the turn oracle.
    scored_rows = [dict(r, completed=bool(r.get('completed') and r.get('journal_status') == 'completed'
                   and len(r['turns']) == planned.get((r['task_id'], r['repetition']))
                   and all(t['score']['task_completed'] for t in r['turns']))) for r in rows]
    metrics = aggregate_cases(scored_rows, 60)
    metrics['planned_turns'] = 78
    metrics['planned_parsing_coverage'] = sum(t['score']['parsed'] for r in rows for t in r['turns']) / 78
    metrics['turn_execution_coverage'] = metrics['executed_turns'] / 78
    metrics['interrupted_cases'] = sum(r.get('journal_status') == 'interrupted' for r in rows)
    metrics['evaluation_status'] = 'invalid' if not eligible else 'complete' if complete else 'incomplete'
    metrics['acceptance_status'] = ('passed' if complete and metrics['pilot_targets_met']
                                    and not metrics['fact_violations'] else 'failed' if complete else 'not_evaluated')
    metrics['formal_acceptance'] = metrics['acceptance_status'] == 'passed'
    metrics['rounds'] = {str(r + 1): aggregate_cases([row for row in scored_rows if row['repetition'] == r], 20)
                         for r in range(3)}
    return metrics


def terminal_stop(result):
    status = result.get('execution_status')
    if status in {'authentication_error', 'insufficient_balance', 'missing_api_key'}:
        return status
    if status == 'request_budget_exceeded':
        return 'http_budget_exhausted'
    return None


def aggregate_vision(rows):
    """Report unique source/image counts; repetitions and cached counts are not new scenes."""
    grouped = {}
    for row in rows:
        for turn in row['turns']:
            for image_id, counts in turn.get('cv_counts', {}).items():
                key = (row['image_sha256'], image_id)
                entry = grouped.setdefault(key, {'source': row['source'], 'image_id': image_id,
                                                  'counts': counts, 'consistent': True})
                entry['consistent'] &= counts == entry['counts']
    output = {}
    for image_id in ('original', 'corrected', 'accepted'):
        entries = [v for v in grouped.values() if v['image_id'] == image_id]
        sums = {k: sum(e['counts'][k] for e in entries) for k in ('tp', 'fp', 'fn')}
        output[image_id] = {'unique_sources': len(entries), **sums,
                            'recall': sums['tp'] / (sums['tp'] + sums['fn']) if sums['tp'] + sums['fn'] else None,
                            'precision': sums['tp'] / (sums['tp'] + sums['fp']) if sums['tp'] + sums['fp'] else None,
                            'repeat_counts_consistent': all(e['consistent'] for e in entries)}
    return output


def run_formal(cases, detector, image_loader, retriever, *, run_dir, task_file, root,
               bindings, budget, resume=False, client_factory=None):
    validate_protocol(cases, 'live', 'sodd', 3)
    with evaluation_lock(run_dir):
        return _run_formal(cases, detector, image_loader, retriever, run_dir=Path(run_dir),
                           task_file=task_file, root=root, bindings=bindings, budget=budget,
                           resume=resume, client_factory=client_factory)


def _run_formal(cases, detector, image_loader, retriever, *, run_dir, task_file, root,
                bindings, budget, resume, client_factory):
    client_factory = client_factory or (lambda: CloudClient(request_budget=budget))
    probe = client_factory()
    secrets = (getattr(probe, 'api_key', None),)
    manifest = freeze_formal(run_dir, root, task_file, bindings, budget.limit, resume=resume)
    path = run_dir / 'formal_sodd_live.json'
    if path.exists():
        output = json.loads(path.read_text(encoding='utf-8'))
        if output['manifest_sha256'] != digest(manifest):
            raise ValueError('结果与冻结清单不符')
        if output.get('stop_reason') in {'authentication_error', 'insufficient_balance', 'missing_api_key'}:
            raise ValueError('认证或余额错误已停止此批次；修复账户后使用新的评测编号')
    else:
        if budget.snapshot()['used']:
            raise ValueError('存在已消耗请求却缺少正式进度；拒绝重置或重跑')
        output = {'schema_version': 3, 'backend': 'live', 'vision_backend': 'sodd', 'phase': 'formal',
                  'model': probe.model_name, 'repeats': 3, 'qualification': QUALIFICATION,
                  'task_sha256': manifest['task_sha256'], 'manifest_sha256': digest(manifest),
                  'selected_case_ids': HELD_IDS, 'started_at': now(), 'cases': [], 'stop_history': []}
    rows = output['cases']
    if output.get('stop_reason'):
        output['stop_history'].append({'reason': output['stop_reason'], 'resumed_at': now()})
    stop = None
    for row in rows:
        if row['journal_status'] == 'started':
            row.update(journal_status='interrupted', completed=False,
                       interruption_reason='process_interrupted', ended_at=now())
    if formal_metrics(rows, cases, backend='live', vision_backend='sodd', repeats=3)['evaluation_status'] == 'invalid':
        raise ValueError('案例进度损坏或重复；拒绝继续')
    done = {(r['task_id'], r['repetition']) for r in rows}

    def save():
        # A currently started row is temporarily incomplete, never eligible to pass.
        output.update(updated_at=now(), stop_reason=stop, budget=budget.snapshot(),
                      metrics=formal_metrics(rows, cases, backend='live', vision_backend='sodd', repeats=3),
                      skipped=[{'task_id': c['id'], 'repetition': r, 'reason': stop or 'not_started'}
                               for r in range(3) for c in cases if (c['id'], r) not in done])
        output['metrics']['ledger_http_attempts'] = output['budget']['used']
        output['metrics']['unattributed_reserved_attempts'] = output['budget']['used'] - output['metrics']['request_attempts']
        output['vision_metrics'] = aggregate_vision(rows)
        write_json(path, output, secrets)

    save()
    consecutive_service_failures = 0
    for repetition in range(3):
        for case in cases:
            if stop or (case['id'], repetition) in done:
                continue
            if not budget.snapshot()['remaining']:
                stop = 'http_budget_exhausted'
                break
            row = {'task_id': case['id'], 'kind': case['kind'], 'split': 'test',
                   'repetition': repetition, 'source_index': case['source_index'],
                   'source': bindings['images'][HELD_IDS.index(case['id'])]['source'],
                   'image_sha256': bindings['images'][HELD_IDS.index(case['id'])]['pixels_sha256'],
                   'image_variant': case['image_variant'], 'turns': [], 'completed': False,
                   'journal_status': 'started', 'started_at': now()}
            rows.append(row)
            done.add((case['id'], repetition))
            save()  # Durable before the first request: interrupted cases cannot be cherry-picked.
            service_failed, setup_ok = False, True
            try:
                image, truth, source = image_loader(case)
                if digest(image.tobytes()) != row['image_sha256'] or digest(truth) != bindings['images'][HELD_IDS.index(case['id'])]['truth_sha256']:
                    raise ValueError('输入图像或标注在预检之后发生变化')
                session = Session(ToolContext(detector, image, retriever))
                runtime = AgentRuntime(client_factory())
                for index, turn in enumerate(case['turns']):
                    # Labels, case ID and source index are NOT passed to the model.
                    result = runtime.run(session, turn['message'])
                    score = score_turn(turn['expected'], result, session.context,
                                       setup_ok=setup_ok, original_hash=row['image_sha256'])
                    item = {'turn_index': index, 'message': turn['message'], 'expected': turn['expected'],
                            'result': result, 'score': score,
                            'observed_evidence_ids': list(evidence_map(session.context)), 'cv_counts': {}}
                    for image_id in ('original', 'corrected'):
                        if image_id in session.context.detections:
                            item['cv_counts'][image_id] = match_counts([session.context.detections[image_id]], [truth], detector.threshold)
                    if result.get('vision_result') is not None:
                        vision = result['vision_result']
                        item['cv_counts']['accepted'] = match_counts([
                            vision['accepted_detections'] if vision['status'] == 'reliable' else []], [truth], detector.threshold)
                    row['turns'].append(item)
                    write_json(run_dir / 'formal_sodd_live' / f"{case['id']}_{repetition}_turn{index}.json", item, secrets)
                    setup_ok &= score['task_completed']
                    service_failed |= result.get('execution_status') == 'network_error'
                    stop = terminal_stop(result)
                    save()
                    if stop:
                        break
                row.update(journal_status='completed', ended_at=now(),
                           completed=len(row['turns']) == len(case['turns']) and all(t['score']['task_completed'] for t in row['turns']),
                           unrun_turns=list(range(len(row['turns']), len(case['turns']))))
            except BaseException:
                row.update(journal_status='interrupted', completed=False, ended_at=now(),
                           interruption_reason='execution_interrupted',
                           unrun_turns=list(range(len(row['turns']), len(case['turns']))))
                stop = 'execution_interrupted'
                save()
                raise
            consecutive_service_failures = consecutive_service_failures + 1 if service_failed else 0
            if consecutive_service_failures >= 2:
                stop = 'consecutive_service_failures'
            save()
            print(json.dumps({'round': repetition + 1, 'case': case['id'], 'completed': row['completed'],
                              'failures': [t['score']['failure_categories'] for t in row['turns']],
                              'budget': output['budget']}, ensure_ascii=False), flush=True)
    save()
    return output
