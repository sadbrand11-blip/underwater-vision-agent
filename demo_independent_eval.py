"""Observe a self-validated wrong intent rejected by an independent oracle; no API."""

import argparse
import json
from pathlib import Path

import numpy as np

from optical_agent.benchmark import FixtureDetector
from optical_agent.independent_eval import score_turn, write_json
from optical_agent.llm import ScriptedClient
from optical_agent.runtime import AgentRuntime
from optical_agent.state import Session
from optical_agent.tools import ToolContext


class WrongIntentFixture:
    provider, model_name = 'scripted', 'deliberately_wrong_intent_fixture_not_an_llm'

    def __init__(self):
        self.responses = iter([
            {'content': json.dumps({'task_type': 'quality', 'image_reference': 'original', 'question': ''})},
            {'content': None, 'tool_calls': [{'id': 'fixture_quality', 'type': 'function',
                'function': {'name': 'assess_image_quality', 'arguments': '{"image_id":"original"}'}}]},
            {'content': '{"evidence_ids":["quality:original"],"citation_ids":[]}'},
        ])

    def complete(self, *args, **kwargs):
        return {'message': next(self.responses), 'usage': {}, 'http_attempts': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/agent_eval/offline_demo.json')
    args = parser.parse_args()
    root = Path(__file__).parent
    case = json.loads((root / 'eval' / 'agent_tasks_v2.json').read_text(encoding='utf-8'))['cases'][3]
    image = np.full((64, 64, 3), 100, np.uint8)
    session = Session(ToolContext(FixtureDetector(), image))
    wrong = AgentRuntime(WrongIntentFixture()).run(session, case['turns'][0]['message'])
    wrong_score = score_turn(case['turns'][0]['expected'], wrong, session.context)
    correct = AgentRuntime(ScriptedClient()).run(session, case['turns'][0]['message'])
    correct_score = score_turn(case['turns'][0]['expected'], correct, session.context)
    result = {'qualification': '离线故障注入演示，不代表真实LLM的准确率或改善',
              'expected': case['turns'][0],
              'wrong_intent': {'result': wrong, 'independent_score': wrong_score},
              'correct_intent': {'result': correct, 'independent_score': correct_score}}
    assert wrong['task_validation']['passed'] and not wrong_score['task_completed']
    assert correct_score['task_completed']
    write_json(root / args.output, result)
    print('预期任务：比较校正前后 → 误解析：曝光评估 → 程序自校验：通过 → 独立评分：失败（任务理解错误、缺少比较证据）')
    print('使用正确任务解析 → 完整比较证据 → 独立评分：通过；未调用云端API。')


if __name__ == '__main__':
    main()
