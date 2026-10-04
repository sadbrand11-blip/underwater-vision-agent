"""Offline protocol gates and crash recovery; no cloud requests are made."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from optical_agent.benchmark import FixtureDetector
from optical_agent.formal_eval import (formal_metrics, freeze_formal, run_formal,
                                      validate_protocol, aggregate_vision)
from optical_agent.independent_eval import digest
from optical_agent.llm import CloudClient, LLMError, ScriptedClient
from optical_agent.rag import KnowledgeRetriever
from optical_agent.request_budget import RequestBudget, RequestBudgetError

ROOT = Path(__file__).resolve().parents[1]
TASK_FILE = ROOT / 'eval' / 'agent_tasks_v2.json'
CASES = [c for c in json.loads(TASK_FILE.read_text(encoding='utf-8'))['cases'] if c['split'] == 'test']


def scored_rows():
    # Metric fixtures explicitly provide an oracle outcome, not runtime self-validation.
    score = dict(parsed=True, intent_correct=True, image_correct=True, tool_selection_correct=True,
                 fact_match=True, original_preserved=True, task_completed=True,
                 tool_call_count=1, invalid_argument_count=0, unregistered_tool_executions=0,
                 reliability_overrides=0, failure_categories=[])
    return [dict(task_id=c['id'], repetition=r, kind=c['kind'], journal_status='completed', completed=True,
                 turns=[dict(turn_index=i, score=score.copy(), result={'execution_mode': 'cloud', 'request_attempts': 1})
                        for i in range(len(c['turns']))]) for r in range(3) for c in CASES]


def metrics(rows, **kw):
    return formal_metrics(rows, CASES, backend=kw.get('backend', 'live'),
                          vision_backend=kw.get('vision_backend', 'sodd'), repeats=3,
                          frozen_valid=kw.get('frozen_valid', True))


class FormalGateTests(unittest.TestCase):
    def test_complete_protocol_and_missing_duplicate_or_simulated_cannot_pass(self):
        rows = scored_rows()
        self.assertTrue(metrics(rows)['formal_acceptance'])
        for altered, options in [(rows[:-1], {}), (rows[:-1] + [rows[0]], {}),
                                  (rows, {'backend': 'scripted'}), (rows, {'vision_backend': 'fixture'}),
                                  (rows, {'frozen_valid': False})]:
            with self.subTest(options=options, length=len(altered)):
                self.assertFalse(metrics(altered, **options)['formal_acceptance'])
        rows[0]['turns'][0]['result']['execution_mode'] = 'scripted'
        self.assertEqual(metrics(rows)['evaluation_status'], 'invalid')

    def test_metrics_reject_completed_flag_and_keep_planned_denominator(self):
        rows = scored_rows()
        for row in rows[:10]:
            row['turns'][0]['score']['task_completed'] = False
        outcome = metrics(rows)
        self.assertEqual(outcome['evaluation_status'], 'complete')
        self.assertEqual(outcome['acceptance_status'], 'failed')
        self.assertAlmostEqual(outcome['planned_case_completion_rate'], 50 / 60)
        self.assertAlmostEqual(metrics(rows[:20])['planned_case_completion_rate'], 10 / 60)

    def test_facts_reliability_and_unregistered_execution_are_zero_tolerance(self):
        for field, value in [('fact_match', False), ('original_preserved', False),
                             ('reliability_overrides', 1), ('unregistered_tool_executions', 1)]:
            rows = scored_rows()
            rows[0]['turns'][0]['score'][field] = value
            self.assertFalse(metrics(rows)['formal_acceptance'])

    def test_interrupted_or_missing_turn_keeps_eval_incomplete(self):
        rows = scored_rows()
        rows[0]['journal_status'] = 'interrupted'
        self.assertEqual(metrics(rows)['evaluation_status'], 'incomplete')
        rows = scored_rows()
        rows[0]['turns'] = []
        self.assertEqual(metrics(rows)['executed_turns'], 77)
        self.assertAlmostEqual(metrics(rows)['planned_parsing_coverage'], 77 / 78)

    def test_protocol_cannot_select_subset_or_reduce_repeats(self):
        validate_protocol(CASES, 'live', 'sodd', 3)
        for cases, back, vision, n in [(CASES[:-1], 'live', 'sodd', 3), (CASES, 'live', 'sodd', 1),
                                      (CASES, 'scripted', 'sodd', 3), (CASES, 'live', 'fixture', 3)]:
            with self.assertRaises(ValueError):
                validate_protocol(cases, back, vision, n)

    def test_vision_deduplicates_repeated_and_cached_counts(self):
        counts = {'tp': 2, 'fp': 1, 'fn': 1}
        rows = [dict(image_sha256='same-photo', source='one.jpg', turns=[{'cv_counts': {'original': counts}}]) for _ in range(3)]
        value = aggregate_vision(rows)['original']
        self.assertEqual(value['unique_sources'], 1)
        self.assertEqual(value['tp'], 2)
        rows[-1]['turns'][0]['cv_counts']['original'] = dict(counts, tp=1)
        self.assertFalse(aggregate_vision(rows)['original']['repeat_counts_consistent'])

    @patch('optical_agent.llm.requests.post')
    def test_601st_request_never_sent_and_cap_cannot_change(self, post):
        post.return_value = Mock(status_code=200, headers={}, json=Mock(return_value={
            'choices': [{'message': {'content': '{}'}}]}))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'budget.json'
            budget = RequestBudget(path, 600)
            client = CloudClient(api_key='local-fixture', request_budget=budget)
            for _ in range(600):
                client.complete([])
            with self.assertRaises(LLMError):
                CloudClient(api_key='local-fixture', request_budget=RequestBudget(path, 600)).complete([])
            self.assertEqual(post.call_count, 600)
            with self.assertRaises(RequestBudgetError):
                RequestBudget(path, 120)
            path.write_text('{}', encoding='utf-8')
            with self.assertRaises(RequestBudgetError):
                budget.reserve()


class FormalRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_dir = self.root / 'run'
        self.image = np.full((32, 32, 3), 100, np.uint8)
        self.truth = []
        self.bindings = {'files': {}, 'images': [
            {'source': f'{c["id"]}.jpg', 'pixels_sha256': digest(self.image.tobytes()),
             'truth_sha256': digest(self.truth)} for c in CASES]}
        self.budget = RequestBudget(self.run_dir / 'http_budget.json', 600)
        self.retriever = KnowledgeRetriever(ROOT)
        self.kwargs = dict(run_dir=self.run_dir, root=ROOT, task_file=TASK_FILE,
                           bindings=self.bindings, budget=self.budget)

    def factory(self, *, fail=False, capture=None):
        budget = self.budget
        class OfflineCloudFixture(ScriptedClient):
            provider, model_name, api_key = 'cloud', 'explicit_offline_cloud_fixture', 'private-fixture-key'
            def complete(self, messages, *args, **kw):
                budget.reserve()
                if capture is not None:
                    capture.append(copy.deepcopy(messages))
                if fail:
                    raise KeyboardInterrupt('simulate process interruption')
                response = super().complete(messages, *args, **kw)
                response['http_attempts'] = 1
                response['http_trace'] = [{'http_status': 200}]
                return response
        return OfflineCloudFixture

    def run_eval(self, **kw):
        return run_formal(CASES, FixtureDetector(), lambda c: (self.image, self.truth, c['id'] + '.jpg'),
                          self.retriever, **self.kwargs, **kw)

    def test_interrupt_resume_keeps_failed_case_and_never_resends_labels(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_eval(client_factory=self.factory(fail=True))
        before = json.loads((self.run_dir / 'formal_sodd_live.json').read_text(encoding='utf-8'))
        self.assertEqual(before['cases'][0]['journal_status'], 'interrupted')
        captured = []
        result = self.run_eval(resume=True, client_factory=self.factory(capture=captured))
        self.assertEqual(len(result['cases']), 60)
        self.assertEqual(result['cases'][0], before['cases'][0])
        self.assertFalse(result['metrics']['formal_acceptance'])
        self.assertEqual(result['metrics']['executed_turns'], 77)
        self.assertEqual([r['task_id'] for r in result['cases'][:20]], [c['id'] for c in CASES])
        for messages in captured:
            sent = json.dumps(messages)
            self.assertNotIn('allowed_tools', sent)
            self.assertNotIn('task_types', sent)
            self.assertNotIn('hold_01', sent)
        used = self.budget.snapshot()['used']
        second = self.run_eval(resume=True, client_factory=self.factory())
        self.assertEqual(self.budget.snapshot()['used'], used)
        self.assertEqual(second['cases'], result['cases'])
        self.assertNotIn('private-fixture-key', (self.run_dir / 'formal_sodd_live.json').read_text(encoding='utf-8'))

    def test_existing_run_requires_resume_and_changed_binding_is_rejected(self):
        freeze_formal(self.run_dir, ROOT, TASK_FILE, self.bindings, 600, resume=False)
        with self.assertRaises(ValueError):
            freeze_formal(self.run_dir, ROOT, TASK_FILE, self.bindings, 600, resume=False)
        for bindings, limit in [(dict(self.bindings, changed='config'), 600), (self.bindings, 120)]:
            with self.assertRaises(ValueError):
                freeze_formal(self.run_dir, ROOT, TASK_FILE, bindings, limit, resume=True)
        (self.run_dir / 'frozen_tasks.json').write_text('{}', encoding='utf-8')
        with self.assertRaises(ValueError):
            freeze_formal(self.run_dir, ROOT, TASK_FILE, self.bindings, 600, resume=True)

    def test_started_journal_from_killed_process_is_failed_on_resume(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_eval(client_factory=self.factory(fail=True))
        path = self.run_dir / 'formal_sodd_live.json'
        output = json.loads(path.read_text(encoding='utf-8'))
        output['cases'][0]['journal_status'] = 'started'
        path.write_text(json.dumps(output), encoding='utf-8')
        result = self.run_eval(resume=True, client_factory=self.factory())
        self.assertEqual(result['cases'][0]['interruption_reason'], 'process_interrupted')
        self.assertFalse(result['cases'][0]['completed'])
