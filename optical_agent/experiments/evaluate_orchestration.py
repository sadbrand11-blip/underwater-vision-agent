"""Run the frozen task set with a scripted harness or a real cloud model."""
from optical_agent.paths import PROJECT_ROOT

import argparse
import json
import re
from pathlib import Path

import numpy as np

from data import SODDDataset, simulate_exposure
from optical_agent.benchmark import FixtureDetector, run_benchmark
from optical_agent.llm import CloudClient, load_local_env
from optical_agent.rag import KnowledgeRetriever
from optical_agent.request_budget import RequestBudget, RequestBudgetError
from optical_agent.independent_eval import digest, run_independent


def main():
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['pilot', 'formal'], default='pilot')
    parser.add_argument('--preflight', action='store_true', help='Validate inputs offline; send zero HTTP requests')
    parser.add_argument('--resume', action='store_true', help='Continue a frozen formal run without rerunning recorded cases')
    parser.add_argument('--backend', choices=['scripted','live'], default='scripted')
    parser.add_argument('--vision', choices=['fixture','sodd'], default='fixture')
    parser.add_argument('--split', choices=['dev','test','all'])
    parser.add_argument('--systems', nargs='+', choices=['fixed','coarse','fine'])
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--task-file', default='eval/agent_tasks.json')
    parser.add_argument('--task-ids', nargs='+')
    parser.add_argument('--run-id', help='Same ID across before/after, real images and resumed requests')
    parser.add_argument('--phase', choices=['before','fix1','fix2','after','offline'], default='before')
    parser.add_argument('--max-http-attempts', type=int, default=120)
    parser.add_argument('--root', default=r'D:\CodexData\optical_agent\sodd\SODD\data')
    parser.add_argument('--model', default=str(root/'models'/'sodd_detector.pt'))
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.mode == 'formal' and args.task_file == 'eval/agent_tasks.json':
        args.task_file = 'eval/agent_tasks_v2.json'
    task_path = root / args.task_file
    task_data = json.loads(task_path.read_text(encoding='utf-8'))
    v2 = isinstance(task_data, dict) and task_data.get('schema_version') == 2
    args.repeats = args.repeats if args.repeats is not None else (3 if args.mode == 'formal' else 1 if v2 else 3)
    args.split = args.split or ('test' if args.mode == 'formal' else 'dev' if v2 else 'test')
    args.systems = args.systems or (['fine'] if v2 else ['fixed','coarse','fine'])
    if args.repeats < 1:
        parser.error('repeats must be positive')
    if not 1 <= args.max_http_attempts <= 600:
        parser.error('HTTP attempt limit must be 1-600 (default remains 120)')
    if args.mode != 'formal' and (args.preflight or args.resume):
        parser.error('--preflight and --resume require formal mode')
    if args.run_id and not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', args.run_id):
        parser.error('run-id must contain 1—64 letters, numbers, underscores or hyphens')
    if v2 and args.systems != ['fine']:
        parser.error('V2 independent intent pilot supports fine; legacy file retains all three baselines')
    if v2 and args.backend == 'live' and not args.run_id:
        parser.error('Live V2 evaluation requires an explicit persistent run-id')
    load_local_env(root/'.env')
    if args.backend == 'live' and not args.preflight and not CloudClient().configured:
        parser.error('Cloud key is not configured; live evaluation was not run.')
    tasks = task_data['cases'] if v2 else task_data
    tasks = [t for t in tasks if args.split == 'all' or t['split'] == args.split]
    if args.task_ids:
        missing = set(args.task_ids) - {t['id'] for t in tasks}
        if missing:
            parser.error('Unknown task IDs in this split: ' + ', '.join(sorted(missing)))
        tasks = [t for t in tasks if t['id'] in args.task_ids]
    if not tasks:
        parser.error('No evaluation cases selected')
    if args.mode == 'formal':
        from optical_agent.formal_eval import validate_protocol
        try:
            validate_protocol(tasks, args.backend, args.vision, args.repeats)
            if not v2 or args.task_ids or args.split != 'test':
                raise ValueError('Formal mode requires the full V2 held-out test split')
        except ValueError as exc:
            parser.error(str(exc))
    detector = FixtureDetector()
    if args.vision == 'sodd':
        from detector import TorchDetector
        detector = TorchDetector(args.model)
        dataset = SODDDataset(args.root, 'test')
        def load_image(task):
            index = task['source_index'] if v2 else task['source_index'] % len(dataset)
            if not 0 <= index < len(dataset):
                raise ValueError('Source index outside existing SODD test subset')
            tensor, target, source = dataset[index]
            image = np.uint8(np.round(tensor.permute(1,2,0).numpy()*255))
            truth = [{'box': b.tolist(), 'class_id': int(label)} for b,label in zip(target['boxes'],target['labels'])]
            pair = (simulate_exposure(image, task['image_variant']), truth)
            return (*pair, Path(source).name) if v2 else pair
    else:
        def load_image(task):
            image = np.full((64,64,3),100,np.uint8)
            image[20:40,20:40] = 80
            pair = (simulate_exposure(image,task['image_variant']), None)
            return (*pair, 'fixed_visual_fixture') if v2 else pair
    if args.mode == 'formal':
        from optical_agent.formal_eval import collect_bindings, run_formal
        retriever = KnowledgeRetriever(root)
        try:
            bindings = collect_bindings(root, task_path, tasks, detector, dataset, load_image, retriever, CloudClient())
            if args.preflight:
                print(json.dumps({'preflight': 'passed', 'http_requests': 0, 'cases': 20, 'repeats': 3,
                                  'planned_turns': 78, 'bindings_sha256': digest(bindings),
                                  'cloud_key_configured': CloudClient().configured}, ensure_ascii=True))
                return
            run_dir = root / 'runs' / 'agent_eval' / args.run_id
            budget = RequestBudget(run_dir / 'http_budget.json', args.max_http_attempts,
                must_exist=args.resume or (run_dir / 'formal_manifest.json').exists())
            result = run_formal(tasks, detector, load_image, retriever, run_dir=run_dir, task_file=task_path,
                root=root, bindings=bindings, budget=budget, resume=args.resume)
        except (ValueError, RequestBudgetError) as exc:
            parser.error(str(exc))
        if args.output:
            from optical_agent.independent_eval import write_json
            write_json(root / args.output, result, (CloudClient().api_key,))
        print(json.dumps(result['metrics'], ensure_ascii=True))
        return
    if v2:
        run_dir = root / 'runs' / 'agent_eval' / (args.run_id or 'offline_v2')
        try:
            budget = (RequestBudget(run_dir / 'http_budget.json', args.max_http_attempts,
                                    must_exist=(run_dir / 'manifest.json').exists())
                      if args.backend == 'live' else None)
        except RequestBudgetError as exc:
            parser.error(str(exc))
        result = run_independent(tasks, detector, load_image, KnowledgeRetriever(root),
            backend=args.backend, vision_backend=args.vision, run_dir=run_dir, task_file=task_path,
            root=root, phase=args.phase, budget=budget, repeats=args.repeats)
        if args.output:
            from optical_agent.independent_eval import write_json
            write_json(root / args.output, result, (CloudClient().api_key,))
        print(json.dumps(result['metrics'], ensure_ascii=True))
        return
    result = run_benchmark(tasks,detector,load_image,KnowledgeRetriever(root),args.backend,args.vision,args.systems,args.repeats)
    result['split'] = args.split
    result['unique_task_count'] = len(tasks)
    result['unique_source_count'] = len({t['source_index'] for t in tasks})
    result['dataset_root'] = args.root if args.vision == 'sodd' else None
    output = Path(args.output or 'runs/orchestration_scripted.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({s: r['metrics'] for s,r in result['systems'].items()},ensure_ascii=True))


if __name__ == '__main__':
    main()
