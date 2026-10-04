"""Two actual Python processes, real SODD detector, offline rule dispatch, isolated memory."""
from optical_agent.paths import PROJECT_ROOT
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2

from detector import TorchDetector
from optical_agent.adaptive_fixture import AdaptiveScriptedClient
from optical_agent.adaptive_runtime import AdaptiveRuntime
from optical_agent.memory import MemoryStore, image_hash
from optical_agent.state import Session
from optical_agent.tools import ToolContext

ROOT = PROJECT_ROOT
DATA = Path(r'D:\CodexData\optical_agent')


def stage(name, output):
    store = MemoryStore(output / 'memory.sqlite3')
    if name == 'first':
        store.set_preferences({'enabled': True, 'target_classes': ['管道'], 'display_detail': 'compact'})
    index = 6 if name == 'first' else 7
    files = sorted((DATA / 'sodd/SODD/data/test/images').glob('*_original.jpg'))
    if len(files) <= index:
        raise FileNotFoundError('D盘缺少既有SODD测试图，本演示不下载数据')
    bgr = cv2.imread(str(files[index]))
    if bgr is None:
        raise ValueError('示例图无法读取')
    detector = TorchDetector(ROOT / 'models/sodd_detector.pt')
    session = Session(ToolContext(detector, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    session.input_provenance = {'kind': 'real_sodd', 'source_index': index, 'file_name': files[index].name}
    result = AdaptiveRuntime(AdaptiveScriptedClient(), memory_store=store).run(session, '只识别图中的目标')
    record = {'process_id': os.getpid(), 'input': session.input_provenance,
              'input_sha256': image_hash(session.context.images['original']), 'task_status': result['task_status'],
              'task_validation': result['task_validation'], 'target_classes': result['goal_contract']['target_classes'],
              'candidate_count': result['target_count'], 'visual_status': result['vision_status'],
              'computation_counts': result['computation_counts'], 'memory_context': result['memory_context'],
              'preferences': store.preferences(), 'history_count': len(store.list_history()),
              'trace': result['trace'], 'actual_HTTP_requests': result['request_attempts'], 'paid_API_requests': 0,
              'dispatch': 'offline_rule_fixture_not_LLM_ability', 'vision_backend': 'real_existing_SODD_detector'}
    (output / (name + '.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=DATA / 'memory/runs/demo_v051')
    parser.add_argument('--stage', choices=['first', 'second'])
    args = parser.parse_args()
    output = args.output_dir.resolve()
    allowed = (DATA / 'memory').resolve()
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError('演示记录必须位于D盘memory的独立子目录，不使用个人记忆数据库')
    output.mkdir(parents=True, exist_ok=True)
    if args.stage:
        stage(args.stage, output)
        return
    if (output / 'memory.sqlite3').exists():
        raise ValueError('已有演示记录不会覆盖；请选择新的独立输出目录')
    for name in ('first', 'second'):
        subprocess.run([sys.executable, '-u', '-X', 'utf8', '-B', str(Path(__file__).resolve()),
                        '--stage', name, '--output-dir', str(output)], cwd=ROOT, check=True)
    first, second = [json.loads((output / (name + '.json')).read_text(encoding='utf-8')) for name in ('first', 'second')]
    checks = {'distinct_processes': first['process_id'] != second['process_id'],
              'both_tasks_completed': all(r['task_status'] == 'completed' and r['task_validation']['passed'] for r in (first, second)),
              'preferences_persist': first['preferences'] == second['preferences'],
              'different_images': first['input_sha256'] != second['input_sha256'],
              'history_loaded_after_restart': bool(second['memory_context']['history_refs']),
              'new_image_detection_computed': second['computation_counts']['detection'] == 1,
              'two_summaries_saved': second['history_count'] == 2,
              'no_paid_API_requests': first['actual_HTTP_requests'] == second['actual_HTTP_requests'] == 0}
    summary = {'version': '0.5.1', 'checks': checks, 'passed': all(checks.values()),
               'first': first, 'second': second, 'personal_memory_modified': False,
               'limitation': 'Functional persistence demonstration; not real LLM ability or visual accuracy evaluation.'}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': summary['passed'], 'checks': checks, 'records': str(output)}, ensure_ascii=False))
    if not summary['passed']:
        raise RuntimeError('离线记忆演示未通过，保留记录供检查')


if __name__ == '__main__':
    main()
