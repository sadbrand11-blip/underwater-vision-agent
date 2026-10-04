"""Offline demo: a premature finish is rejected, evidence is completed, validation passes."""
import argparse
import json
from pathlib import Path

import numpy as np

from optical_agent.benchmark import FixtureDetector
from optical_agent.runtime import AgentRuntime
from optical_agent.state import Session
from optical_agent.tools import ToolContext


def finish(evidence_id):
    return {'message': {'content': json.dumps({'evidence_ids': [evidence_id], 'citation_ids': []})}}


def actions(items):
    return {'message': {'content': None, 'tool_calls': [
        {'id': f'demo_{i}', 'type': 'function', 'function': {
            'name': name, 'arguments': json.dumps(args)}} for i, (name, args) in enumerate(items)]}}


class DemoClient:
    provider = 'scripted'
    model_name = 'offline_premature_finish_fixture_not_an_llm'

    def __init__(self):
        self.responses = iter([
            {'message': {'content': json.dumps({
                'task_type': 'comparison', 'image_reference': 'original', 'question': ''})}},
            actions([('assess_image_quality', {'image_id': 'original'})]),
            finish('quality:original'),
            actions([('correct_image_exposure', {'image_id': 'original'}),
                     ('assess_image_quality', {'image_id': 'corrected'}),
                     ('detect_objects', {'image_id': 'original'}),
                     ('detect_objects', {'image_id': 'corrected'}),
                     ('compare_detection_evidence', {'original_image_id': 'original', 'corrected_image_id': 'corrected'})]),
            finish('comparison:original')])

    def complete(self, *args, **kwargs):
        return next(self.responses)


def run_demo():
    image = np.full((64, 64, 3), 100, np.uint8)
    session = Session(ToolContext(FixtureDetector(), image))
    result = AgentRuntime(DemoClient()).run(session, '比较校正前后，识别有没有改善')
    np.testing.assert_array_equal(image, session.context.images['original'])
    if result['task_status'] != 'completed' or not result['task_validation']['passed']:
        raise RuntimeError('离线完成校验示例未通过')
    result['qualification'] = '合成图片与检测夹具，模型响应为预设回放；只验证调度和校验，不评价真实LLM或视觉精度。'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/task_validation_demo.json')
    args = parser.parse_args()
    result = run_demo()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(result['qualification'])
    for item in result['trace']:
        if item['type'] == 'task_understanding':
            print('任务理解：' + item['contract']['task_name'])
            print('必要结果：' + '；'.join(item['contract']['requirements']))
        elif item['type'] == 'tool':
            print('Action → 工具执行：' + item['tool'] + ('（成功）' if item['ok'] else '（失败）'))
        elif item['type'] == 'task_validation':
            print('完成校验：' + ('通过' if item['passed'] else '未通过'))
            for missing in item['missing']:
                print('  缺项：' + missing['requirement'] + '，' + missing['reason'])
    print(f'最终：任务完成，模型响应 {result["model_calls"]} 次，工具 {result["tool_calls"]} 次，结果纠错 {result["format_repairs"]} 次。')
    print('保存完整记录：' + str(target.resolve()))


if __name__ == '__main__':
    main()
