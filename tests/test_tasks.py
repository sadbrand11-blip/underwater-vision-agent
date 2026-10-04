import json
import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from optical_agent.llm import ScriptedClient
from optical_agent.runtime import AgentRuntime
from optical_agent.state import Session
from optical_agent.tasks import parse_contract
from optical_agent.tools import ToolContext, execute_tool
from test_agent_tools import FakeDetector
from test_runtime import call


def intent(kind, image='original', question=''):
    return {'message': {'content': json.dumps({
        'task_type': kind, 'image_reference': image, 'question': question})}}


def finish(*ids, citations=None, **extra):
    return {'message': {'content': json.dumps({
        'evidence_ids': list(ids), 'citation_ids': citations or [], **extra})}}


def comparison_batch():
    actions = [('correct_image_exposure', {'image_id': 'original'}),
               ('assess_image_quality', {'image_id': 'corrected'}),
               ('detect_objects', {'image_id': 'original'}),
               ('detect_objects', {'image_id': 'corrected'}),
               ('compare_detection_evidence', {'original_image_id': 'original', 'corrected_image_id': 'corrected'})]
    return {'message': {'tool_calls': [call(name, args, f't{i}')['message']['tool_calls'][0]
                                      for i, (name, args) in enumerate(actions)]}}


class WireQueue:
    provider = 'test'
    model_name = 'offline_intent_and_execution_fixture'

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.last = None

    def complete(self, *args, **kwargs):
        self.requests.append((args, kwargs))
        self.last = next(self.responses, self.last)
        return self.last


class TaskContractTests(unittest.TestCase):
    def setUp(self):
        self.detector = FakeDetector()
        self.image = np.full((64, 64, 3), 100, np.uint8)
        self.session = Session(ToolContext(self.detector, self.image))
        self.ctx = self.session.context

    def quality(self):
        execute_tool(self.ctx, 'assess_image_quality', {'image_id': 'original'})

    def full(self):
        execute_tool(self.ctx, 'analyze_image_full_pipeline', {'image_id': 'original'})

    def test_comparison_quality_only_rejected_then_completed(self):
        client = WireQueue([intent('comparison'),
                            call('assess_image_quality', {'image_id': 'original'}),
                            finish('quality:original'), comparison_batch(), finish('comparison:original')])
        result = AgentRuntime(client).run(self.session, '比较校正前后识别有没有改善')
        self.assertEqual(result['task_status'], 'completed')
        self.assertTrue(result['task_validation']['passed'])
        checks = [t for t in result['trace'] if t['type'] == 'task_validation']
        self.assertFalse(checks[0]['passed'])
        self.assertIn('comparison:original', [x['requirement'] for x in checks[0]['missing']])
        self.assertTrue(checks[-1]['passed'])
        self.assertEqual(result['format_repairs'], 1)
        self.assertEqual(result['model_calls'], 5)
        self.assertEqual(result['tool_calls'], 6)
        self.assertIn('comparison:original', client.requests[3][0][0][-1]['content'])
        np.testing.assert_array_equal(self.image, self.ctx.images['original'])

    def test_comparison_still_incomplete_after_two_repairs(self):
        self.quality()
        result = AgentRuntime(WireQueue([intent('comparison'), finish('quality:original')])).run(self.session, '比较前后')
        self.assertEqual(result['execution_status'], 'task_incomplete')
        self.assertEqual(result['task_status'], 'incomplete')
        self.assertEqual(result['format_repairs'], 2)
        self.assertEqual(result['model_calls'], 4)
        self.assertEqual(result['evidence'][0]['evidence_id'], 'quality:original')
        self.assertIn('comparison:original', [x['requirement'] for x in result['task_validation']['missing']])

    def test_detection_cannot_finish_with_old_quality(self):
        self.quality()
        result = AgentRuntime(WireQueue([intent('detection'), finish('quality:original')])).run(self.session, '检测目标')
        self.assertEqual(result['task_status'], 'incomplete')
        self.assertFalse(result['task_validation']['passed'])
        self.assertEqual(self.detector.calls, 0)
        self.assertEqual(result['evidence'], [])
        self.assertIn('detections:original', [x['requirement'] for x in result['task_validation']['missing']])

    def test_corrected_detection_cannot_deliver_original_detection(self):
        self.full()
        result = AgentRuntime(WireQueue([intent('detection', 'corrected'), finish('detections:original')])).run(self.session, '识别校正后的图')
        self.assertEqual(result['task_status'], 'incomplete')
        self.assertEqual(result['task_contract']['image_id'], 'corrected')
        self.assertIn('detections:corrected', [x['requirement'] for x in result['task_validation']['missing']])
        self.assertNotIn('detections:original', [x['evidence_id'] for x in result['evidence']])

    def test_cached_required_evidence_finishes_without_tools(self):
        execute_tool(self.ctx, 'detect_objects', {'image_id': 'original'})
        detector_calls = self.detector.calls
        result = AgentRuntime(WireQueue([intent('detection'), finish('detections:original')])).run(self.session, '检测目标')
        self.assertEqual(result['task_status'], 'completed')
        self.assertEqual(result['tool_calls'], 0)
        self.assertEqual(self.detector.calls, detector_calls)
        self.assertEqual(result['model_calls'], 2)

    def test_cached_final_report_updates_next_turn_image_reference(self):
        self.full()
        self.ctx.current_image_id = 'corrected'
        client = WireQueue([intent('detection'), finish('detections:original'),
                            intent('quality', 'current'), finish('quality:original')])
        runtime = AgentRuntime(client)
        self.assertEqual(runtime.run(self.session, '识别原图')['task_status'], 'completed')
        followup = runtime.run(self.session, '刚才那张曝光怎么样')
        self.assertEqual(followup['task_contract']['image_id'], 'original')
        self.assertEqual(followup['tool_calls'], 0)

    def test_evidence_exists_but_not_selected_is_rejected(self):
        self.full()
        result = AgentRuntime(WireQueue([intent('comparison'), finish('quality:original')])).run(self.session, '比较前后')
        missing = result['task_validation']['missing']
        self.assertIn({'requirement': 'comparison:original', 'reason': '证据存在，但最终报告未选用'}, missing)
        self.assertEqual(result['task_status'], 'incomplete')

    def test_contract_cannot_be_overridden_by_model_finish(self):
        self.quality()
        result = AgentRuntime(WireQueue([intent('comparison'), finish('quality:original',
            task_contract={'task_type': 'quality', 'required_evidence_ids': ['quality:original']})])).run(self.session, '比较前后')
        self.assertEqual(result['task_contract']['task_type'], 'comparison')
        self.assertEqual(result['task_contract']['required_evidence_ids'], ['comparison:original'])
        self.assertEqual(result['task_status'], 'incomplete')
        contract = parse_contract(intent('comparison')['message']['content'], self.ctx, 'original')
        with self.assertRaises(FrozenInstanceError):
            contract.required_evidence_ids = ()

    def test_parser_cannot_supply_completion_requirements(self):
        bad = intent('quality')
        data = json.loads(bad['message']['content'])
        data['required_evidence_ids'] = []
        bad['message']['content'] = json.dumps(data)
        result = AgentRuntime(WireQueue([bad])).run(self.session, '只检查曝光')
        self.assertIsNone(result['task_contract'])
        self.assertEqual(result['error_details']['stage'], 'task_understanding')
        self.assertEqual(result['model_calls'], 3)
        self.assertEqual(result['tool_calls'], 0)

    def test_quality_task_blocks_detection_before_execution(self):
        client = WireQueue([intent('quality'), call('detect_objects', {'image_id': 'original'}),
                            call('assess_image_quality', {'image_id': 'original'}), finish('quality:original')])
        result = AgentRuntime(client).run(self.session, '只检查曝光')
        self.assertEqual(result['task_status'], 'completed')
        self.assertEqual(self.detector.calls, 0)
        blocked = next(t for t in result['trace'] if t['type'] == 'tool')
        self.assertFalse(blocked['ok'])
        self.assertIn('无关工具', blocked['error'])

    def test_corrected_task_blocks_wrong_image_before_execution(self):
        self.quality()
        execute_tool(self.ctx, 'correct_image_exposure', {'image_id': 'original'})
        client = WireQueue([intent('detection', 'corrected'),
                            call('detect_objects', {'image_id': 'original'}),
                            call('detect_objects', {'image_id': 'corrected'}),
                            finish('detections:corrected')])
        result = AgentRuntime(client).run(self.session, '识别校正后的图')
        self.assertEqual(result['task_status'], 'completed')
        self.assertNotIn('original', self.ctx.detections)
        self.assertEqual(self.detector.calls, 1)

    def test_current_image_is_frozen_at_task_start(self):
        self.full()
        self.ctx.current_image_id = 'corrected'
        # 模型解析指向 current，即使此前历史中出现原图，仍绑定开始时的校正图。
        result = AgentRuntime(WireQueue([intent('detection', 'current'), finish('detections:corrected')])).run(self.session, '识别刚才那张')
        self.assertEqual(result['task_contract']['start_image_id'], 'corrected')
        self.assertEqual(result['task_contract']['image_id'], 'corrected')
        self.assertEqual(result['task_status'], 'completed')

    def test_missing_corrected_image_clarifies_without_tools(self):
        result = AgentRuntime(WireQueue([intent('detection', 'corrected')])).run(self.session, '识别校正后的图')
        self.assertEqual(result['task_status'], 'needs_clarification')
        self.assertIn('还没有校正图', result['answer'])
        self.assertEqual(result['tool_calls'], 0)
        self.assertEqual(result['model_calls'], 1)
        self.assertNotIn('corrected', self.ctx.images)
        self.assertIsNotNone(self.session.pending_clarification)

    def test_clarification_reply_includes_original_question(self):
        AgentRuntime(WireQueue([intent('clarify', question='要检查哪张图的曝光？')])).run(self.session, '想看看曝光怎么样')
        client = WireQueue([intent('quality'), call('assess_image_quality', {'image_id': 'original'}),
                            finish('quality:original')])
        result = AgentRuntime(client).run(self.session, '原图')
        supplied = json.loads(client.requests[0][0][0][1]['content'])
        self.assertEqual(supplied['pending_clarification']['original_message'], '想看看曝光怎么样')
        self.assertEqual(supplied['message'], '原图')
        self.assertEqual(result['task_status'], 'completed')
        self.assertIsNone(self.session.pending_clarification)
        self.assertIn('想看看曝光怎么样', self.session.messages[-2]['content'])

    def test_scripted_clarification_reply_original_switches_image(self):
        runtime = AgentRuntime(ScriptedClient())
        self.assertEqual(runtime.run(self.session, '识别校正后的那张图')['task_status'], 'needs_clarification')
        answer = runtime.run(self.session, '原图')
        self.assertEqual(answer['task_contract']['task_type'], 'detection')
        self.assertEqual(answer['task_contract']['image_id'], 'original')
        self.assertEqual(answer['task_status'], 'completed')
        self.assertEqual(answer['evidence'][0]['evidence_id'], 'detections:original')

    def test_unsupported_and_conditional_tasks_do_not_execute(self):
        for message, status in [('识别图里的鱼', 'unsupported'),
                                ('如果曝光不好就校正，否则只识别', 'needs_clarification')]:
            with self.subTest(message=message):
                session = Session(ToolContext(FakeDetector(), self.image))
                result = AgentRuntime(ScriptedClient()).run(session, message)
                self.assertEqual(result['task_status'], status)
                self.assertEqual(result['tool_calls'], 0)
                self.assertEqual(result['evidence'], [])

    def test_identity_correction_has_image_parameters_without_detection(self):
        result = AgentRuntime(ScriptedClient()).run(self.session, '只校正曝光')
        self.assertEqual(result['task_status'], 'completed')
        self.assertEqual(result['evidence'][0]['evidence_id'], 'correction:original')
        self.assertFalse(result['evidence'][0]['data']['parameters']['applied'])
        self.assertIn('未调整', result['answer'])
        self.assertIn('corrected', result['image_refs'])
        self.assertEqual(self.detector.calls, 0)
        np.testing.assert_array_equal(self.ctx.images['corrected'], self.image)

    def test_empty_detection_is_completed_but_candidate_only(self):
        class EmptyDetector(FakeDetector):
            def predict(self, image):
                self.calls += 1
                return []
        session = Session(ToolContext(EmptyDetector(), self.image))
        result = AgentRuntime(ScriptedClient()).run(session, '识别目标')
        self.assertEqual(result['task_status'], 'completed')
        self.assertEqual(result['evidence'][0]['data'], [])
        self.assertIn('不能据此证明', result['answer'])
        self.assertIsNone(result['vision_status'])

    def test_no_knowledge_matches_cannot_finish_explanation(self):
        class EmptyRetriever:
            def search(self, query):
                return []
        self.ctx.retriever = EmptyRetriever()
        result = AgentRuntime(ScriptedClient()).run(self.session, '解释为什么不能接受')
        self.assertEqual(result['task_status'], 'incomplete')
        self.assertEqual(result['citations'], [])
        self.assertIn('knowledge_citation', [x['requirement'] for x in result['task_validation']['missing']])
        self.assertIn('没有选用', result['answer'])
        self.assertLessEqual(result['model_calls'], 10)
        self.assertLessEqual(result['format_repairs'], 2)

    def test_quality_failure_is_a_completed_comparison_not_a_reliable_result(self):
        session = Session(ToolContext(FakeDetector(), np.full((64, 64, 3), 255, np.uint8)))
        result = AgentRuntime(ScriptedClient()).run(session, '比较前后')
        self.assertEqual(result['task_status'], 'completed')
        self.assertTrue(result['task_validation']['passed'])
        self.assertEqual(result['vision_status'], 'quality_failure')

    def test_intent_consumes_model_budget_and_all_repairs_share_limit(self):
        parser_error = {'message': {'content': '{}'}}
        self.quality()
        client = WireQueue([parser_error, intent('comparison'), finish('quality:original')])
        result = AgentRuntime(client).run(self.session, '比较前后')
        self.assertEqual(result['format_repairs'], 2)
        self.assertEqual(result['model_calls'], 4)
        self.assertEqual(result['task_status'], 'incomplete')
        limited = AgentRuntime(WireQueue([intent('quality')]), max_model_calls=1).run(self.session, '只检查曝光')
        self.assertEqual(limited['execution_status'], 'budget_exceeded')
        self.assertEqual(limited['model_calls'], 1)
        self.assertEqual(limited['tool_calls'], 0)
        self.assertFalse(limited['task_validation']['passed'])

    def test_intent_tokens_and_elapsed_time_share_task_budget(self):
        self.quality()
        first, last = intent('quality'), finish('quality:original')
        first['usage'] = {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
        last['usage'] = {'prompt_tokens': 20, 'completion_tokens': 5, 'total_tokens': 25}
        result = AgentRuntime(WireQueue([first, last])).run(self.session, '曝光')
        self.assertTrue(result['usage']['available'])
        self.assertEqual(result['usage']['total_tokens'], 40)
        now = [0]
        class SlowClient(WireQueue):
            def complete(client, *args, **kwargs):
                response = super().complete(*args, **kwargs)
                now[0] += 95
                return response
        timed = AgentRuntime(SlowClient([intent('quality'), call('assess_image_quality', {'image_id': 'original'})]),
                             clock=lambda: now[0]).run(self.session, '曝光')
        self.assertEqual(timed['execution_status'], 'budget_exceeded')
        self.assertEqual(timed['model_calls'], 2)
        self.assertEqual(timed['tool_calls'], 0)

    def test_coarse_quality_retains_full_flow_but_selects_quality(self):
        result = AgentRuntime(ScriptedClient(), tool_mode='coarse').run(self.session, '只检查曝光')
        self.assertEqual(result['task_status'], 'completed')
        self.assertEqual([x['tool'] for x in result['trace'] if x['type'] == 'tool'], ['analyze_image_full_pipeline'])
        self.assertEqual(result['evidence'][0]['evidence_id'], 'quality:original')
        self.assertIsNone(result['vision_status'])
