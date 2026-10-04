"""0.3.0 bounded development evaluation; separate from the frozen formal 20 cases."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np

from data import SODDDataset
from optical_agent.adaptive_eval import FixedVisualFixture, fixed_workflow, freeze, save, score, summarize
from optical_agent.adaptive_fixture import AdaptiveScriptedClient
from optical_agent.adaptive_runtime import AdaptiveRuntime
from optical_agent.adaptive_tools import pixel_hash
from optical_agent.diagnostics import redact
from optical_agent.independent_eval import digest
from optical_agent.llm import CloudClient, load_local_env
from optical_agent.rag import KnowledgeRetriever
from optical_agent.request_budget import RequestBudget
from optical_agent.state import Session
from optical_agent.tools import ToolContext
from quality import assess, UNDERWATER_QUALITY_CONFIG
from metrics import match_counts


def inputs(root, cases, dataset_root):
    dataset = SODDDataset(dataset_root, 'test')
    prepared, inventory = {}, {}
    for case in cases:
        if case['split'] == 'dev':
            image = np.full((64, 64, 3), case['pixels'], np.uint8)
            info = {'kind':'authored_visual_fixture', 'pixels':case['pixels'], 'declining_fixture':case.get('declining_fixture', False), 'ground_truth_sha256':None}
        else:
            tensor, target, source = dataset[case['source_index']]
            original = np.uint8(np.round(tensor.permute(1, 2, 0).numpy() * 255))
            truth = {'boxes':target['boxes'].tolist(), 'labels':target['labels'].tolist()}
            info = {'kind':'real_sodd' if case['variant'] == 'original' else 'simulated_exposure_perturbation',
                    'source_index':case['source_index'], 'source_name':Path(source).name,
                    'source_pixels_sha256':pixel_hash(original), 'ground_truth_sha256':digest(truth), 'truth':truth}
            image = original if case['variant'] == 'original' else np.uint8(original.astype(float) * .6) if case['variant'] == 'multiply_0.6_simulated' else np.zeros_like(original)
        quality = assess(image, UNDERWATER_QUALITY_CONFIG)[0]
        info.update(input_pixels_sha256=pixel_hash(image), exposure_state=quality['exposure_state'], quality_pass=quality['quality_pass'])
        prepared[case['id']], inventory[case['id']] = image, info
    for name, exposure, passed in [('demo_normal','normal',True), ('demo_dark','underexposed',True), ('demo_lost','underexposed',False)]:
        if inventory[name]['exposure_state'] != exposure or inventory[name]['quality_pass'] != passed:
            raise ValueError(f'演示输入前提不符：{name}；不能改变预设标签凑成三条路径')
    return prepared, inventory


def run_history(run_dir):
    """Read the entire run, independent of selected split/cases or phase."""
    records=[]
    for path in Path(run_dir).glob('*/*.json'):
        if re.fullmatch(r'(dev|demo)_.+_\d+',path.stem):
            records.append(json.loads(path.read_text(encoding='utf-8')))
    records.sort(key=lambda r:r.get('started_at',''))
    consecutive=0
    for record in records:
        if record.get('stop_batch'):
            return consecutive,record['stop_batch']
        statuses=[t.get('adaptive',{}).get('result',{}).get('execution_status') for t in record.get('turns',[])]
        # A turn is checkpointed before stop_batch; recover a crash in that window.
        terminal=next((s for s in statuses if s in {'authentication_error','insufficient_balance','request_budget_exceeded'}),None)
        if terminal:
            return consecutive,terminal
        network='network_error' in statuses
        consecutive=consecutive+1 if network else 0
        if consecutive>=2:
            return consecutive,'two_consecutive_connection_failures'
    return consecutive,None


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['scripted','live'], default='scripted')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--phase', choices=['offline','before','fix1'], default='offline')
    parser.add_argument('--split', choices=['all','dev','demo'], default='all')
    parser.add_argument('--case-ids', nargs='+')
    parser.add_argument('--root', default=r'D:\CodexData\optical_agent\sodd\SODD\data')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', args.run_id):
        parser.error('run-id格式无效')
    if (args.backend == 'live') == (args.phase == 'offline'):
        parser.error('云端使用before/fix1，离线使用offline；不得混用记录')
    all_cases = json.loads((root/'eval/adaptive_tasks.json').read_text(encoding='utf-8'))['cases']
    chosen = [c for c in all_cases if args.split == 'all' or c['split'] == args.split]
    if args.case_ids:
        if set(args.case_ids) - {c['id'] for c in chosen}:
            parser.error('指定案例不存在于所选分组')
        chosen = [c for c in chosen if c['id'] in args.case_ids]
    prepared, inventory = inputs(root, all_cases, args.root)
    load_local_env(root/'.env')
    client = CloudClient() if args.backend == 'live' else AdaptiveScriptedClient()
    manifest = freeze(root, all_cases, inventory, client)
    if args.preflight:
        print(json.dumps({'preflight':'passed', 'http_attempts':0, 'frozen_hash':digest(manifest),
                          'planned_case_runs':sum(c.get('repeats',1) for c in chosen), 'inputs':inventory}, ensure_ascii=False))
        return
    if args.backend == 'live' and not client.configured:
        parser.error('未配置本地密钥；未发送请求')
    run_dir = root/'runs/adaptive_eval'/args.run_id
    stop_file = run_dir/'terminal_stop.json'
    if stop_file.exists():
        print(json.dumps({'run_id':args.run_id,'stopped':json.loads(stop_file.read_text(encoding='utf-8')),'http_attempts_this_invocation':0},ensure_ascii=False))
        return
    prior_connection_failures,prior_stop=run_history(run_dir)
    if prior_stop:
        save(stop_file,{'reason':prior_stop,'recovered_from_run_history':True})
        print(json.dumps({'stopped':prior_stop,'http_attempts_this_invocation':0},ensure_ascii=False))
        return
    phase_dir = run_dir/args.phase
    manifest_file = phase_dir/'freeze.json'
    if manifest_file.exists():
        if not args.resume:
            parser.error('运行记录已经存在；请用--resume继续，不能覆盖失败')
        if json.loads(manifest_file.read_text(encoding='utf-8')) != manifest:
            parser.error('冻结内容变化，停止继续运行；唯一允许的开发修复使用fix1新阶段')
    else:
        if args.resume:
            parser.error('没有可以继续的冻结阶段')
        if args.phase == 'fix1' and not (run_dir/'before/freeze.json').exists():
            parser.error('fix1必须保留同一运行编号下的初始记录')
        save(manifest_file, manifest)
    budget = RequestBudget(run_dir/'http_budget.json', limit=120,
                           must_exist=args.resume or args.phase == 'fix1') if args.backend == 'live' else None
    if budget:
        client.request_budget = budget
    from detector import TorchDetector
    real_detector = TorchDetector(root/'models/sodd_detector.pt')
    retriever = KnowledgeRetriever(root)
    records, connection_failures, stop = [], prior_connection_failures, None
    planned = sum(c.get('repeats',1) for c in chosen)
    def current_budget():
        return budget.snapshot() if budget else {'used':0,'limit':120,'remaining':120}
    def refresh():
        summary = summarize(records, planned, current_budget())
        summary.update(run_id=args.run_id, phase=args.phase, backend=args.backend, stop_reason=stop, frozen_hash=digest(manifest))
        save(phase_dir/('summary_'+args.split+'.json'), summary)
        return summary
    for case in chosen:
        for repetition in range(case.get('repeats',1)):
            path = phase_dir/f'{case["id"]}_{repetition}.json'
            if path.exists():
                record = json.loads(path.read_text(encoding='utf-8'))
                if record['status'] == 'started':
                    record['status'], record['failure'] = 'interrupted_failed', '进程中断；不选择重跑后的最好成绩'
                    save(path, record)
                records.append(record)
                if record.get('stop_batch'):
                    stop = record['stop_batch']
                    break
                continue
            if current_budget()['remaining'] <= 0:
                stop = 'request_budget_exceeded'
                break
            image = prepared[case['id']]
            detector = FixedVisualFixture(case.get('declining_fixture', False)) if case['split'] == 'dev' else real_detector
            adaptive_session = Session(ToolContext(detector, image, retriever))
            fixed_session = Session(ToolContext(detector, image, retriever))
            record = {'case':case['id'], 'repeat':repetition, 'status':'started', 'turns':[],
                      'source':inventory[case['id']], 'frozen_hash':digest(manifest), 'started_at':datetime.now(timezone.utc).isoformat()}
            save(path, record)
            try:
                for index, turn in enumerate(case['turns']):
                    previous = getattr(adaptive_session, 'adaptive', None)
                    start_candidate = previous.last_candidate if previous else None
                    result = AdaptiveRuntime(client).run(adaptive_session, turn['message'])
                    observed = getattr(adaptive_session, 'adaptive', None)
                    adaptive_score = score(turn['expected'], result, observed, start_candidate, pixel_hash(image))
                    fixed_previous = getattr(fixed_session, 'adaptive', None)
                    fixed_start = fixed_previous.last_candidate if fixed_previous else None
                    fixed_result = fixed_workflow(fixed_session, turn['message'])
                    fixed_score = score(turn['expected'], fixed_result, fixed_session.adaptive, fixed_start, pixel_hash(image))
                    row = {'case':case['id'], 'repeat':repetition, 'turn':index, 'message':turn['message'], 'expected':turn['expected'],
                           'adaptive':{'result':result,'score':adaptive_score}, 'fixed':{'result':fixed_result,'score':fixed_score}}
                    if case['split'] == 'demo':
                        truth = inventory[case['id']]['truth']
                        wanted_ids = {6,2}
                        targets = [{'box':b, 'class_id':int(c)} for b,c in zip(truth['boxes'],truth['labels']) if int(c) in wanted_ids]
                        for name, state in [('adaptive', observed), ('fixed', fixed_session.adaptive)]:
                            row[name]['visual_counts'] = {image_id:match_counts(
                                [[b for b in boxes if b['class_id'] in wanted_ids]], [targets], detector.threshold)
                                for image_id, boxes in state.raw_detections.items()} if state else {}
                    record['turns'].append(row)
                    save(path, redact(record, (getattr(client, 'api_key', None),)))
                    print(json.dumps({'case':case['id'],'repeat':repetition,'turn':index,'passed':adaptive_score['passed'],
                        'failure':adaptive_score['failure'], 'status':result['execution_status'],'selected':result['selected_image_id'],
                        'revisions':len(result['plan_revisions']), 'tools':result['tool_calls'], 'budget':current_budget()}, ensure_ascii=False), flush=True)
                    if result['execution_status'] in {'authentication_error','insufficient_balance','request_budget_exceeded'}:
                        stop = result['execution_status']
                        break
                    if result['execution_status'] == 'network_error':
                        break
                record['status'] = 'recorded'
                network_case = any(t['adaptive']['result']['execution_status'] == 'network_error' for t in record['turns'])
                connection_failures = connection_failures + 1 if network_case else 0
                if connection_failures >= 2:
                    stop = 'two_consecutive_connection_failures'
            except Exception as exc:
                record['status'], record['failure'] = 'local_failed', redact(type(exc).__name__ + ': ' + str(exc), (getattr(client,'api_key',None),))
                stop = 'local_error'
            if stop:
                record['stop_batch'] = stop
                save(stop_file, {'reason':stop, 'case':case['id'], 'repeat':repetition,
                                 'phase':args.phase, 'budget':current_budget()})
            save(path, redact(record, (getattr(client,'api_key',None),)))
            records.append(record)
            refresh()
            if stop:
                break
        if stop:
            break
    if stop and not stop_file.exists():
        save(stop_file,{'reason':stop,'phase':args.phase,'budget':current_budget()})
    summary = refresh()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
