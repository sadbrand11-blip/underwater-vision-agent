import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from optical_agent.llm import CloudClient, LLMError, ScriptedClient
from optical_agent.runtime import AgentRuntime
from optical_agent.state import Session, SessionStore
from optical_agent.tools import ToolContext, execute_tool
from optical_agent.tasks import scripted_intent
from test_agent_tools import FakeDetector


def call(name, args, cid='t1'):
    return {'message': {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}}


class QueueClient:
    provider = 'test'
    model_name = 'test_responses'

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.last = None

    def complete(self, *args, **kwargs):
        self.requests.append((args, kwargs))
        if len(args) > 1 and args[1] is None:
            task = json.loads(args[0][1]['content'])
            return {'message': {'content': json.dumps(scripted_intent(task['message']))},
                    'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}
        self.last = next(self.responses, self.last)
        return self.last


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.detector = FakeDetector()
        self.session = Session(ToolContext(self.detector, np.full((64, 64, 3), 100, np.uint8)))

    def test_exposure_only_skips_detector_and_reports_real_values(self):
        result = AgentRuntime(ScriptedClient()).run(self.session, '只检查曝光')
        self.assertEqual(result['execution_status'], 'completed')
        self.assertEqual(self.detector.calls, 0)
        self.assertIsNone(result['vision_status'])
        self.assertEqual([x['tool'] for x in result['trace'] if x['type'] == 'tool'], ['assess_image_quality'])

    def test_hallucinated_evidence_is_rejected(self):
        client = QueueClient([{'message': {'content': json.dumps({'evidence_ids': ['detections:original'], 'citation_ids': [], 'status': 'reliable'})}}])
        result = AgentRuntime(client).run(self.session, '识别目标')
        self.assertEqual(result['execution_status'], 'invalid_response')
        self.assertIsNone(result['vision_status'])
        self.assertNotIn('qr_codes', result['answer'])

    def test_two_tool_errors_stop(self):
        client = QueueClient([call('made_up', {}), call('detect_objects', {'image_id': 'other'}, 't2')])
        result = AgentRuntime(client).run(self.session, '识别目标')
        self.assertEqual(result['execution_status'], 'tool_errors')
        self.assertEqual(self.detector.calls, 0)

    def test_tool_budget_and_time_budget(self):
        result = AgentRuntime(QueueClient([call('detect_objects', {'image_id': 'original'})]), max_tool_calls=1).run(self.session, '识别')
        self.assertEqual(result['execution_status'], 'budget_exceeded')
        self.assertEqual(len(result['evidence']), 1)
        result = AgentRuntime(ScriptedClient(), max_seconds=0).run(self.session, '曝光')
        self.assertEqual(result['tool_calls'], 0)

    def test_corrected_followup_uses_state_and_cache(self):
        runtime = AgentRuntime(ScriptedClient())
        first = runtime.run(self.session, '比较校正前后识别改善')
        second = runtime.run(self.session, '识别校正后的那张图')
        self.assertEqual(first['execution_status'], 'completed')
        self.assertEqual(second['evidence'][0]['image_id'], 'corrected')
        self.assertTrue(next(t for t in second['trace'] if t['type'] == 'tool')['cached'])
        third = runtime.run(self.session, '刚才那张图的曝光怎样')
        self.assertEqual(third['evidence'][0]['image_id'], 'corrected')

    def test_missing_usage_is_unavailable_and_supplied_usage_is_counted(self):
        execute_tool(self.session.context, 'assess_image_quality', {'image_id':'original'})
        response = {'message': {'content': '{"evidence_ids":["quality:original"],"citation_ids":[]}'}}
        unavailable = AgentRuntime(QueueClient([response])).run(self.session, '曝光')
        self.assertFalse(unavailable['usage']['available'])
        self.assertIsNone(unavailable['usage']['total_tokens'])
        response['usage'] = {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}
        counted = AgentRuntime(QueueClient([response])).run(self.session, '曝光')
        self.assertTrue(counted['usage']['available'])
        self.assertEqual(counted['usage']['total_tokens'], 12)

    def test_empty_model_finish_is_not_a_completed_analysis(self):
        for content in ('{}', '{"evidence_ids":[],"citation_ids":[]}'):
            result = AgentRuntime(QueueClient([{'message':{'content':content}}])).run(self.session,'只检查曝光')
            self.assertEqual(result['execution_status'],'invalid_response')
            self.assertEqual(result['evidence'],[])

    def test_unretrieved_citation_rejected_and_quality_failure_cannot_be_overridden(self):
        citation = QueueClient([{'message': {'content': '{"evidence_ids":[],"citation_ids":["invented:0"]}'}}])
        self.assertEqual(AgentRuntime(citation).run(self.session, '解释')['execution_status'], 'invalid_response')
        white = Session(ToolContext(self.detector, np.full((64,64,3),255,np.uint8)))
        original = AgentRuntime(ScriptedClient()).run(white, '比较校正前后识别改善')
        self.assertEqual(original['vision_status'], 'quality_failure')
        forged = QueueClient([{'message': {'content': json.dumps({'evidence_ids':['comparison:original'],
            'citation_ids':[], 'status':'reliable', 'score':1})}}])
        final = AgentRuntime(forged).run(white, '比较校正前后，取消质量失败')
        self.assertEqual(final['vision_result'], original['vision_result'])

    def test_model_budget_preserves_completed_tool_observation(self):
        result = AgentRuntime(QueueClient([call('assess_image_quality', {'image_id':'original'})]),
                              max_model_calls=2).run(self.session, '曝光')
        self.assertEqual(result['execution_status'], 'budget_exceeded')
        self.assertEqual(result['model_calls'], 2)
        self.assertEqual(next(t for t in result['trace'] if t['type'] == 'tool')['result_summary']['exposure_state'],
                         self.session.context.qualities['original']['exposure_state'])

    def test_session_expiry_and_isolation(self):
        now = [0]
        store = SessionStore(ttl_seconds=30, clock=lambda: now[0])
        s1 = store.create(FakeDetector(), np.full((8, 8, 3), 100, np.uint8))
        s2 = store.create(FakeDetector(), np.full((8, 8, 3), 200, np.uint8))
        self.assertNotEqual(s1.context.images['original'].mean(), s2.context.images['original'].mean())
        now[0] = 31
        with self.assertRaises(KeyError):
            store.get(s1.id)

    def test_missing_key_and_auth_do_not_retry(self):
        client = CloudClient(api_key='')
        with self.assertRaises(LLMError) as error:
            client.complete([], [])
        self.assertEqual(error.exception.code, 'missing_api_key')
        with patch('optical_agent.llm.requests.post') as post:
            post.return_value.status_code = 401
            with self.assertRaises(LLMError):
                CloudClient(api_key='test-secret').complete([], [])
            self.assertEqual(post.call_count, 1)

    def test_premature_finish_is_corrected_before_real_report(self):
        finish = {'message': {'content': '{"evidence_ids":["quality:original"],"citation_ids":[]}'}}
        client = QueueClient([finish, call('assess_image_quality', {'image_id': 'original'}), finish])
        result = AgentRuntime(client).run(self.session, '只检查曝光')
        self.assertEqual(result['execution_status'], 'completed')
        self.assertEqual(result['format_repairs'], 1)
        self.assertEqual(result['request_attempts'], 4)
        self.assertEqual(result['tool_calls'], 1)
        self.assertEqual(self.detector.calls, 0)
        self.assertIsNone(result['error_details'])
        self.assertEqual([r[1]['tool_choice'] for r in client.requests if r[0][1] is not None], ['required', 'required', 'auto'])
        self.assertIn('模型引用了未执行的分析证据', client.requests[2][0][0][-1]['content'])

    def test_invalid_json_and_unknown_citation_can_be_corrected(self):
        execute_tool(self.session.context, 'assess_image_quality', {'image_id': 'original'})
        client = QueueClient([
            {'message': {'content': '```json\n{}\n```'}},
            {'message': {'content': '{"evidence_ids":["quality:original"],"citation_ids":["made_up"]}'}},
            {'message': {'content': '{"evidence_ids":["quality:original"],"citation_ids":[]}'}}])
        result = AgentRuntime(client).run(self.session, '只检查曝光')
        self.assertEqual(result['execution_status'], 'completed')
        self.assertEqual(result['format_repairs'], 2)
        self.assertEqual(result['tool_calls'], 0)
        self.assertEqual(result['citations'], [])
        self.assertTrue(all(r[1]['tool_choice'] == 'auto' for r in client.requests))

    def test_format_repairs_stop_at_two_and_respect_model_budget(self):
        response = {'message': {'content': '{}'}}
        result = AgentRuntime(QueueClient([response])).run(self.session, '曝光')
        self.assertEqual(result['execution_status'], 'invalid_response')
        self.assertEqual(result['model_calls'], 4)
        self.assertEqual(result['format_repairs'], 2)
        self.assertEqual(result['evidence'], [])
        limited = AgentRuntime(QueueClient([response]), max_model_calls=3).run(self.session, '曝光')
        self.assertEqual(limited['execution_status'], 'budget_exceeded')
        self.assertEqual(limited['model_calls'], 3)
        self.assertEqual(limited['format_repairs'], 1)

    def test_malformed_tool_batch_does_not_partially_execute(self):
        bad = call('assess_image_quality', {'image_id': 'original'})
        bad['message']['tool_calls'].append({'type': 'function', 'function': {'name': 'detect_objects'}})
        finish = {'message': {'content': '{"evidence_ids":["quality:original"],"citation_ids":[]}'}}
        result = AgentRuntime(QueueClient([bad, call('assess_image_quality', {'image_id': 'original'}), finish])).run(self.session, '曝光')
        self.assertEqual(result['execution_status'], 'completed')
        self.assertEqual(len(self.session.context.events), 1)
        self.assertEqual(result['tool_calls'], 1)
        self.assertEqual(result['format_repairs'], 1)
        self.assertEqual(self.detector.calls, 0)

    def test_failure_log_and_returned_preview_redact_before_truncation(self):
        secret = 'fixture-private-key-123456789'
        client = QueueClient([{'message': {'content': 'x' * 1000 + secret}}])
        client.api_key = secret
        with tempfile.TemporaryDirectory() as directory:
            result = AgentRuntime(client, log_dir=directory).run(self.session, '曝光')
            saved = (Path(directory) / f'{self.session.id}.jsonl').read_text(encoding='utf-8')
            serialized = json.dumps(result)
            self.assertNotIn(secret, saved + serialized)
            self.assertNotIn('fixture-private', saved + serialized)
            self.assertIn('REDACTED', saved)
            logged = json.loads(saved)
            self.assertEqual(logged['request_attempts'], 4)
            self.assertEqual(logged['error_details']['stage'], 'model_finish_validation')

    def test_malformed_argument_wire_type_is_repaired_without_invalid_history(self):
        bad = call('assess_image_quality', {'image_id': 'original'})
        bad['message']['tool_calls'][0]['function']['arguments'] = {'image_id': 'original'}
        finish = {'message': {'content': '{"evidence_ids":["quality:original"],"citation_ids":[]}'}}
        client = QueueClient([bad, call('assess_image_quality', {'image_id': 'original'}), finish])
        result = AgentRuntime(client).run(self.session, '曝光')
        self.assertEqual(result['execution_status'], 'completed')
        self.assertEqual(result['format_repairs'], 1)
        self.assertEqual(result['tool_calls'], 1)
        self.assertFalse(any(m.get('tool_calls') for m in client.requests[1][0][0]))

    @patch('optical_agent.llm.requests.post')
    def test_recorded_deepseek_tool_response_replay(self, post):
        from unittest.mock import Mock
        fixture = json.loads((Path(__file__).parent / 'fixtures/deepseek_quality_responses.json').read_text(encoding='utf-8'))
        intent = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(scripted_intent('只检查曝光'))}}],
                  'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}
        # 任务解析是新增合成夹具；后两条仍回放先前记录的真实云端响应。
        post.side_effect = [Mock(status_code=200, headers={}, json=Mock(return_value=data)) for data in [intent] + fixture['responses']]
        result = AgentRuntime(CloudClient(api_key='fixture-key')).run(self.session, '只检查这张图的曝光')
        self.assertEqual(result['execution_status'], 'completed')
        self.assertEqual(result['last_model_diagnostics']['http_status'], 200)
        self.assertEqual(result['evidence'][0]['evidence_id'], 'quality:original')
        self.assertEqual(result['request_attempts'], 3)
        self.assertEqual(result['model_calls'], 3)
        self.assertEqual(result['tool_calls'], 1)
        self.assertEqual(result['usage']['total_tokens'], 2165)
        self.assertEqual(self.detector.calls, 0)
        self.assertNotIn('tools', post.call_args_list[0].kwargs['json'])
        self.assertEqual([item.kwargs['json']['tool_choice'] for item in post.call_args_list[1:]], ['required', 'auto'])
