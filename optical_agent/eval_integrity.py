"""Offline verification of saved formal artifacts before publishing a report.

Replay uses saved detector observations, never new neural inference or LLM calls.
It verifies quality math, task-oracle scoring and GT box matching; it does not add
semantic/class-selection criteria absent from the frozen oracle.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from data import SODDDataset, simulate_exposure
from metrics import match_counts
from optical_agent.formal_eval import aggregate_vision, file_hash, formal_metrics
from optical_agent.independent_eval import digest, score_turn
from optical_agent.rag import KnowledgeRetriever
from optical_agent.reports import evidence_map
from optical_agent.tools import ToolContext, execute_tool


def require(condition, message):
    if not condition:
        raise ValueError('报告完整性检查失败：' + message)


def verify_formal_artifacts(run_dir, root):
    run_dir, root = Path(run_dir), Path(root)
    output = json.loads((run_dir / 'formal_sodd_live.json').read_text(encoding='utf-8'))
    manifest = json.loads((run_dir / 'formal_manifest.json').read_text(encoding='utf-8'))
    require(digest(manifest) == output['manifest_sha256'], '结果与冻结清单摘要不一致')
    require(file_hash(run_dir / 'frozen_tasks.json') == manifest['task_sha256'] == output['task_sha256'], '冻结题目损坏')
    bindings = manifest['bindings']
    checked = 0
    for name, expected in bindings['files'].items():
        source = Path(name)
        copy_path = (run_dir / 'frozen_inputs' / source.relative_to(root)
                     if source.is_relative_to(root) else source)
        chosen = copy_path if copy_path.exists() else source
        require(chosen.is_file() and file_hash(chosen) == expected, '冻结文件损坏或缺失：' + source.name)
        checked += 1
    # Replay math must match the measured baseline, even when package/docs versions change.
    for name in ['data.py', 'metrics.py', 'quality.py', 'agent.py', 'optical_agent/tools.py',
                 'optical_agent/reports.py', 'optical_agent/independent_eval.py', 'optical_agent/rag.py',
                 'optical_agent/formal_eval.py']:
        require(file_hash(root / name) == bindings['files'][str((root / name).resolve())],
                '重放计算代码已变化，请使用冻结版本：' + name)
    cases = [c for c in json.loads((run_dir / 'frozen_tasks.json').read_text(encoding='utf-8'))['cases'] if c['split'] == 'test']
    case_map = {c['id']: c for c in cases}
    inputs = {}
    dataset = SODDDataset(Path(bindings['images'][0]['image_path']).parents[2], 'test')
    for entry in bindings['images']:
        require(file_hash(entry['image_path']) == entry['image_file_sha256'], '来源图片变化')
        require(file_hash(entry['label_path']) == entry['label_file_sha256'], '来源标注变化')
        tensor, target, source = dataset[entry['source_index']]
        image = np.uint8(np.round(tensor.permute(1, 2, 0).numpy() * 255))
        image = simulate_exposure(image, entry['image_variant'])
        truth = [{'box': b.tolist(), 'class_id': int(label)} for b, label in zip(target['boxes'], target['labels'])]
        require(digest(image.tobytes()) == entry['pixels_sha256'] and digest(truth) == entry['truth_sha256'], '像素或转换标注变化')
        require(Path(source).name == entry['source'], '来源映射变化')
        inputs[entry['task_id']] = (image, truth, entry)
    ledger = json.loads((run_dir / 'http_budget.json').read_text(encoding='utf-8'))
    require(ledger['limit'] == manifest['http_limit'] and 0 <= ledger['used'] <= ledger['limit'], '账本上限不符')
    require([a['number'] for a in ledger['attempts']] == list(range(1, ledger['used'] + 1)), '账本不连续')
    require(output['budget'] == {'limit': ledger['limit'], 'used': ledger['used'], 'remaining': ledger['limit'] - ledger['used']}, '账本与报告计数不符')
    metrics = formal_metrics(output['cases'], cases, backend=output['backend'], vision_backend=output['vision_backend'], repeats=output['repeats'])
    require(metrics['evaluation_status'] != 'invalid', '案例重复或正式协议不符')
    retriever = KnowledgeRetriever(run_dir / 'frozen_inputs')
    require(digest(retriever.chunks) == bindings['knowledge_chunks_sha256'], '冻结知识分段不符')
    http_count = turns_checked = cv_checked = 0
    for row in output['cases']:
        case = case_map[row['task_id']]
        image, truth, entry = inputs[row['task_id']]
        require(row['image_sha256'] == entry['pixels_sha256'] and row['source_index'] == entry['source_index'], '案例图片指向不符')
        predictions = {}
        for item in row['turns']:
            reports = [e['data'] for e in item['result'].get('evidence', []) if e['type'] == 'comparison']
            if item['result'].get('vision_result'):
                reports.append(item['result']['vision_result'])
            for report in reports:
                for image_id in ('original', 'corrected'):
                    boxes = report['detections_' + image_id]
                    require(image_id not in predictions or predictions[image_id] == boxes, '同会话检测缓存变化')
                    predictions[image_id] = boxes
            for evidence in item['result'].get('evidence', []):
                if evidence['type'] == 'detections':
                    image_id = evidence['image_id']
                    require(image_id not in predictions or predictions[image_id] == evidence['data'], '同会话检测缓存变化')
                    predictions[image_id] = evidence['data']
        detector = SimpleNamespace(**bindings['detector'])
        detector.quality_config = bindings['quality_config']
        detector.calibration = bindings['detector']['calibration_sha256'] != digest(None)
        detector.model_path = str(root / 'models' / 'sodd_detector.pt')
        detector.model_sha256 = bindings['files'][str(Path(detector.model_path).resolve())]
        detector.active_image_id = None
        def predict(rgb):
            require(detector.active_image_id in predictions, '缺少可重放的实测检测证据')
            return copy.deepcopy(predictions[detector.active_image_id])
        detector.predict = predict
        context = ToolContext(detector, image, retriever)
        setup_ok = True
        for item in row['turns']:
            index, result = item['turn_index'], item['result']
            require(item['message'] == case['turns'][index]['message'] and item['expected'] == case['turns'][index]['expected'], '逐轮题目或标签变化')
            path = run_dir / 'formal_sodd_live' / f"{row['task_id']}_{row['repetition']}_turn{index}.json"
            require(json.loads(path.read_text(encoding='utf-8')) == item, '逐轮文件与总结果不符')
            for event in result['trace']:
                if event['type'] == 'tool' and event.get('ok'):
                    detector.active_image_id = event['arguments'].get('image_id')
                    observation = execute_tool(context, event['tool'], event['arguments'])
                    require(observation['ok'], '成功工具无法重放：' + event['tool'] + ' ' + str(observation.get('error')))
                    require(context.events[-1]['cached'] == event.get('cached'), '工具缓存记录不符')
            # Comparison wall-clock timing is measured, not recomputed by offline replay.
            if result.get('vision_result') is not None and 'original' in context.reports:
                context.reports['original']['latency_ms'] = result['vision_result']['latency_ms']
            score = score_turn(item['expected'], result, context, setup_ok=setup_ok, original_hash=row['image_sha256'])
            require(score == item['score'], f"独立oracle评分无法重现：{row['task_id']}轮{row['repetition']+1}问题{index+1} "
                    + ','.join(k for k in score if score[k] != item['score'].get(k)))
            require(list(evidence_map(context)) == item['observed_evidence_ids'], '实际证据列表不符')
            setup_ok &= score['task_completed']
            counts = {key: match_counts([boxes], [truth], detector.threshold) for key, boxes in context.detections.items()}
            if result.get('vision_result') is not None:
                vision = result['vision_result']
                counts['accepted'] = match_counts([vision['accepted_detections'] if vision['status'] == 'reliable' else []], [truth], detector.threshold)
            require(counts == item.get('cv_counts', {}), 'GT视觉框匹配计数不符')
            cv_checked += len(counts)
            observed_requests = sum(len(e.get('http_trace', [])) for e in result['trace'])
            require(observed_requests == result['request_attempts'], '逐轮HTTP计数不符')
            models = [e for e in result['trace'] if e['type'] == 'model']
            require(result['model_calls'] == len(models) and result['tool_calls'] == sum(e['type'] == 'tool' for e in result['trace']), '调用次数不符')
            require(result['format_repairs'] == sum(e['type'] == 'format_error' and e.get('will_retry', False)
                    for e in result['trace']), '格式修复计数不符')
            token_keys = ('prompt_tokens', 'completion_tokens', 'total_tokens')
            valid_usage = [e.get('usage', {}) for e in models if isinstance(e.get('usage'), dict)
                           and all(type(e['usage'].get(k)) is int and e['usage'][k] >= 0 for k in token_keys)]
            require(result['usage']['available'] == bool(models and len(valid_usage) == len(models)), 'token可用性不符')
            if result['usage']['available']:
                require(all(result['usage'][k] == sum(u[k] for u in valid_usage) for k in token_keys), '逐请求token合计不符')
            else:
                require(all(result['usage'][k] is None for k in token_keys), '未知token不得填零或当作完整合计')
            http_count += observed_requests
            turns_checked += 1
    require(http_count == metrics['request_attempts'] <= ledger['used'], '总HTTP计数不符')
    require(output['metrics']['unattributed_reserved_attempts'] == ledger['used'] - http_count, '未归属预留计数不符')
    require(all(metrics[k] == output['metrics'][k] for k in metrics), '汇总指标不符')
    require(aggregate_vision(output['cases']) == output['vision_metrics'], '视觉去重汇总不符')
    return {'status': 'verified', 'bound_files_checked': checked, 'images_checked': len(inputs),
            'turns_checked': turns_checked, 'cv_counts_checked': cv_checked,
            'http_attempts_checked': http_count, 'new_cloud_requests': 0,
            'detector_observations': 'saved_observations_replayed_not_new_inference',
            'semantic_scope': 'frozen_task_type_image_tool_evidence_oracle_no_class_filter_check'}
