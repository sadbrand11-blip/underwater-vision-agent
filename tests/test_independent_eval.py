import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from optical_agent.benchmark import FixtureDetector
from optical_agent.independent_eval import (aggregate_cases, digest, freeze_manifest,
    paired_comparison, run_independent, score_turn)
from optical_agent.llm import LLMError, ScriptedClient
from optical_agent.rag import KnowledgeRetriever
from optical_agent.reports import build_report
from optical_agent.runtime import AgentRuntime
from optical_agent.state import Session
from optical_agent.tools import ToolContext, execute_tool
from test_tasks import WireQueue, intent, finish
from test_runtime import call


ROOT = Path(__file__).resolve().parents[1]
TASK_FILE = ROOT / 'eval' / 'agent_tasks_v2.json'
CASES = json.loads(TASK_FILE.read_text(encoding='utf-8'))['cases']


class IndependentEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((64, 64, 3), 100, np.uint8)
        self.image[20:40, 20:40] = 80
        self.session = Session(ToolContext(FixtureDetector(), self.image, KnowledgeRetriever(ROOT)))
        self.ctx = self.session.context

    def expected(self, case_index, turn=0):
        return CASES[case_index]['turns'][turn]['expected']

    def test_self_validation_passes_but_wrong_intent_independently_fails(self):
        result = AgentRuntime(WireQueue([intent('quality'),
            call('assess_image_quality', {'image_id': 'original'}), finish('quality:original')])).run(
                self.session, CASES[3]['turns'][0]['message'])
        self.assertTrue(result['task_validation']['passed'])
        self.assertEqual(result['task_status'], 'completed')
        score = score_turn(self.expected(3), result, self.ctx)
        self.assertFalse(score['task_completed'])
        self.assertIn('intent', score['failure_categories'])
        self.assertIn('report', score['failure_categories'])

    def test_original_corrected_mixup_and_unused_required_evidence(self):
        execute_tool(self.ctx, 'analyze_image_full_pipeline', {'image_id': 'original'})
        result = AgentRuntime(ScriptedClient()).run(self.session, '检测原图中的目标')
        score = score_turn(self.expected(5, 1), result, self.ctx)
        self.assertFalse(score['task_completed'])
        self.assertIn('image_reference', score['failure_categories'])
        self.assertIn('report', score['failure_categories'])
        result = AgentRuntime(ScriptedClient()).run(self.session, '检查原图曝光')
        score = score_turn(self.expected(3), result, self.ctx)
        self.assertIn('comparison:original', self.ctx.reports and ['comparison:original'])
        self.assertFalse(score['evidence_correct'])

    def test_failed_setup_cannot_be_hidden_by_successful_final_turn(self):
        result = AgentRuntime(ScriptedClient()).run(self.session, '检测原图中的目标')
        score = score_turn(self.expected(2), result, self.ctx, setup_ok=False)
        self.assertFalse(score['task_completed'])
        self.assertIn('setup', score['failure_categories'])

    def test_runner_detects_failed_preparation_even_when_final_runtime_completes(self):
        class PreparationFailureFixture(WireQueue):
            model_name = 'wrong_preparation_fixture'
            def __init__(self):
                super().__init__([intent('quality'),
                    call('assess_image_quality', {'image_id': 'original'}), finish('quality:original'),
                    intent('detection'), call('detect_objects', {'image_id': 'original'}), finish('detections:original')])
        with tempfile.TemporaryDirectory() as temp:
            output = run_independent([CASES[6]], FixtureDetector(), lambda c: (self.image, None, 'fixture'),
                KnowledgeRetriever(ROOT), backend='scripted', vision_backend='fixture', run_dir=temp,
                task_file=TASK_FILE, root=ROOT, phase='offline', client_factory=PreparationFailureFixture)
            row = output['cases'][0]
            self.assertEqual(row['turns'][1]['result']['task_status'], 'completed')
            self.assertFalse(row['completed'])
            self.assertIn('setup', row['turns'][1]['score']['failure_categories'])

    def test_valid_cache_empty_detection_and_noop_correction_pass(self):
        class EmptyDetector(FixtureDetector):
            def predict(self, image):
                return []
        session = Session(ToolContext(EmptyDetector(), self.image))
        runtime = AgentRuntime(ScriptedClient())
        runtime.run(session, '检测原图中的目标')
        result = runtime.run(session, '再次检测原图中的目标')
        self.assertTrue(score_turn(self.expected(6, 1), result, session.context)['task_completed'])
        self.assertEqual(result['evidence'][0]['data'], [])
        result = runtime.run(session, '只校正原图曝光')
        self.assertFalse(session.context.corrections['original']['applied'])
        self.assertTrue(score_turn(self.expected(1), result, session.context)['task_completed'])

    def test_quality_failure_clarification_and_unsupported_are_valid_outcomes(self):
        black = Session(ToolContext(FixtureDetector(), np.zeros_like(self.image)))
        result = AgentRuntime(ScriptedClient()).run(black, '检查原图曝光')
        self.assertFalse(black.context.qualities['original']['quality_pass'])
        self.assertTrue(score_turn(self.expected(0), result, black.context)['task_completed'])
        result = AgentRuntime(ScriptedClient()).run(self.session, CASES[7]['turns'][0]['message'])
        self.assertTrue(score_turn(self.expected(7), result, self.ctx)['task_completed'])
        result = AgentRuntime(ScriptedClient()).run(self.session, '识别鱼和珊瑚')
        self.assertTrue(score_turn(self.expected(8), result, self.ctx)['task_completed'])

    def test_api_failure_separate_from_wrong_intent_and_parsing_denominator(self):
        class FailingClient:
            provider, model_name = 'cloud', 'fixture'
            def complete(self, *args, **kwargs):
                raise LLMError('network_error', 'offline network fixture')
        result = AgentRuntime(FailingClient()).run(self.session, '检查原图曝光')
        score = score_turn(self.expected(0), result, self.ctx)
        self.assertIn('api', score['failure_categories'])
        self.assertNotIn('intent', score['failure_categories'])
        metrics = aggregate_cases([{'kind': 'quality', 'completed': False,
                                   'turns': [{'result': result, 'score': score}]}], 2)
        self.assertEqual(metrics['parsing_coverage'], 0)
        self.assertIsNone(metrics['intent_accuracy_on_parsed'])
        self.assertEqual(metrics['intent_accuracy_all_requests'], 0)
        self.assertEqual(metrics['planned_case_completion_rate'], 0)
        successful = AgentRuntime(ScriptedClient()).run(self.session, '检查原图曝光')
        success_score = score_turn(self.expected(0), successful, self.ctx)
        mixed = aggregate_cases([
            {'kind': 'quality', 'completed': False, 'turns': [{'result': result, 'score': score}]},
            {'kind': 'quality', 'completed': True, 'turns': [{'result': successful, 'score': success_score}]}], 2)
        self.assertEqual(mixed['parsing_coverage'], .5)
        self.assertEqual(mixed['intent_accuracy_on_parsed'], 1)
        self.assertEqual(mixed['intent_accuracy_all_requests'], .5)

    def test_empty_retrieval_does_not_pass_citation_requirement(self):
        class NoKnowledge:
            def search(self, query):
                return []
        self.ctx.retriever = NoKnowledge()
        result = AgentRuntime(ScriptedClient()).run(self.session, '解释为什么不能接受并给出依据')
        score = score_turn(self.expected(4), result, self.ctx)
        self.assertEqual(result['task_status'], 'incomplete')
        self.assertFalse(score['citations_correct'])
        self.assertFalse(score['task_completed'])
        self.assertIn('report', score['failure_categories'])

    def test_machine_error_categories_do_not_depend_on_error_wording(self):
        for tool, arguments, category in [('unknown_tool', {}, 'tool_selection'),
                ('detect_objects', {'image_id': 1}, 'parameters'),
                ('detect_objects', {'image_id': 'corrected'}, 'prerequisite'),
                ('correct_image_exposure', {'image_id': 'original'}, 'prerequisite')]:
            observation = execute_tool(self.ctx, tool, arguments)
            self.assertEqual(observation['error_category'], category)
            self.assertTrue(observation['error'])
        class Broken(FixtureDetector):
            def predict(self, image):
                raise RuntimeError('computation fixture')
        observation = execute_tool(ToolContext(Broken(), self.image), 'detect_objects', {'image_id': 'original'})
        self.assertEqual(observation['error_category'], 'computation')
        self.assertTrue(observation['arguments_valid'])

    def test_oracle_and_heldout_freeze_without_model_label_leak(self):
        self.assertEqual(sum(c['split'] == 'dev' for c in CASES), 10)
        self.assertEqual(sum(c['split'] == 'test' for c in CASES), 20)
        client = WireQueue([intent('quality'), call('assess_image_quality', {'image_id': 'original'}), finish('quality:original')])
        AgentRuntime(client).run(self.session, CASES[0]['turns'][0]['message'])
        for args, kwargs in client.requests:
            sent = json.dumps(args[0], ensure_ascii=False)
            self.assertNotIn('task_types', sent)
            self.assertNotIn('allowed_tools', sent)
            self.assertNotIn('required_evidence"', sent)
        with tempfile.TemporaryDirectory() as temp:
            task_file = Path(temp) / 'tasks.json'
            task_file.write_bytes(TASK_FILE.read_bytes())
            freeze_manifest(Path(temp) / 'run', task_file, 'fixture')
            task_file.write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                freeze_manifest(Path(temp) / 'run', task_file, 'fixture')

    def test_original_protection_fact_fabrication_and_reliability_override_detected(self):
        original_hash = digest(self.ctx.images['original'].tobytes())
        result = AgentRuntime(ScriptedClient()).run(self.session, '检测原图中的目标')
        altered = copy.deepcopy(result)
        altered['evidence'][0]['data'][0]['score'] = .9999
        self.assertFalse(score_turn(self.expected(2), altered, self.ctx)['fact_match'])
        altered = copy.deepcopy(result)
        altered['vision_status'] = 'reliable'
        self.assertEqual(score_turn(self.expected(2), altered, self.ctx)['reliability_overrides'], 1)
        self.ctx.images['original'][0, 0, 0] = 0
        self.assertFalse(score_turn(self.expected(2), result, self.ctx, original_hash=original_hash)['original_preserved'])

    def test_all_development_cases_multiturn_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            kwargs = dict(backend='scripted', vision_backend='fixture', run_dir=temp,
                          task_file=TASK_FILE, root=ROOT, phase='offline')
            loader = lambda c: (self.image.copy(), None, 'fixture')
            output = run_independent(CASES[:10], FixtureDetector(), loader, KnowledgeRetriever(ROOT), **kwargs)
            self.assertEqual(output['metrics']['executed_cases'], 10)
            self.assertEqual(output['metrics']['executed_turns'], 13)
            self.assertEqual(output['metrics']['independent_case_completion_rate'], 1)
            second = run_independent(CASES[:10], FixtureDetector(), loader, KnowledgeRetriever(ROOT), **kwargs)
            self.assertEqual(len(second['cases']), 10)
            self.assertEqual(paired_comparison(output, second)['paired_cases'], 10)

    def test_batch_auth_and_two_consecutive_service_failures_stop(self):
        for code, expected_count in [('authentication_error', 1), ('insufficient_balance', 1), ('network_error', 2)]:
            class FailingClient:
                provider, model_name, api_key = 'cloud', 'fixture', 'private-fixture-key'
                def complete(self, *args, **kwargs):
                    raise LLMError(code, self.api_key, extra_secrets=(self.api_key,))
            with self.subTest(code=code), tempfile.TemporaryDirectory() as temp:
                output = run_independent(CASES[:3], FixtureDetector(), lambda c: (self.image, None, 'fixture'),
                    KnowledgeRetriever(ROOT), backend='scripted', vision_backend='fixture', run_dir=temp,
                    task_file=TASK_FILE, root=ROOT, phase='offline', client_factory=FailingClient)
                self.assertEqual(len(output['cases']), expected_count)
                self.assertTrue(output['stop_reason'])
                self.assertNotIn('private-fixture-key', (Path(temp) / 'offline_fixture_scripted.json').read_text(encoding='utf-8'))
                self.assertEqual(len(output['skipped']), 3 - expected_count)
