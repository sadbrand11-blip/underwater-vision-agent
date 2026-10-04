"""Tiny synthetic local files test report tamper gates; no dataset/API download."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from data import SODDDataset
from optical_agent.benchmark import FixtureDetector
from optical_agent.eval_integrity import verify_formal_artifacts
from optical_agent.formal_eval import aggregate_vision, file_hash, formal_metrics
from optical_agent.independent_eval import digest, score_turn, write_json
from optical_agent.llm import ScriptedClient
from optical_agent.rag import KnowledgeRetriever
from optical_agent.reports import evidence_map
from optical_agent.request_budget import RequestBudget
from optical_agent.runtime import AgentRuntime
from optical_agent.state import Session
from optical_agent.tools import ToolContext
from test_runtime import call
from test_tasks import WireQueue, intent

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = ['data.py', 'metrics.py', 'quality.py', 'agent.py', 'optical_agent/tools.py',
              'optical_agent/reports.py', 'optical_agent/independent_eval.py', 'optical_agent/rag.py',
              'optical_agent/formal_eval.py']


class ReportIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'project'
        self.run = self.root / 'runs' / 'fixture'
        frozen = self.run / 'frozen_inputs'
        task = json.loads((ROOT / 'eval/agent_tasks_v2.json').read_text(encoding='utf-8'))
        self.cases = [c for c in task['cases'] if c['split'] == 'test']
        names = ALGORITHMS + ['knowledge/sources.json'] + [s['path'] for s in json.loads((ROOT / 'knowledge/sources.json').read_text(encoding='utf-8'))]
        files = {}
        for name in set(names):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
            snapshot = frozen / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(target.read_bytes())
            files[str(target)] = file_hash(target)
        weight = self.root / 'models/sodd_detector.pt'
        weight.parent.mkdir(parents=True)
        weight.write_bytes(b'explicit synthetic integrity fixture, not a neural weight')
        files[str(weight)] = file_hash(weight)
        data_root = Path(self.temp.name) / 'synthetic_data'
        (data_root / 'test/images').mkdir(parents=True)
        (data_root / 'test/labels').mkdir(parents=True)
        for n in range(40):
            cv2.imwrite(str(data_root / f'test/images/frame_{n:03d}_original.jpg'), np.full((24, 24, 3), 100, np.uint8))
            (data_root / f'test/labels/frame_{n:03d}_original.txt').write_text('', encoding='utf-8')
        dataset = SODDDataset(data_root, 'test')
        retriever = KnowledgeRetriever(self.root)
        entries = []
        for case in self.cases:
            tensor, target, source = dataset[case['source_index']]
            image = np.uint8(np.round(tensor.permute(1, 2, 0).numpy() * 255))
            raw, label = dataset.items[case['source_index']]
            entries.append({'task_id': case['id'], 'source_index': case['source_index'], 'source': Path(source).name,
                            'image_path': str(raw), 'label_path': str(label), 'image_file_sha256': file_hash(raw),
                            'label_file_sha256': file_hash(label), 'pixels_sha256': digest(image.tobytes()),
                            'truth_sha256': digest([]), 'image_variant': 'none'})
        detector = FixtureDetector()
        detector.dataset = 'sodd'
        context = ToolContext(detector, image, retriever)
        bindings = {'files': files, 'images': entries, 'quality_config': asdict(context.quality_config),
                    'knowledge_chunks_sha256': digest(retriever.chunks),
                    'detector': {'dataset': 'sodd', 'classes': [], 'threshold': detector.threshold,
                                 'calibration_sha256': digest(None)}}
        self.run.mkdir(parents=True, exist_ok=True)
        write_json(self.run / 'frozen_tasks.json', task)
        manifest = {'bindings': bindings, 'http_limit': 600, 'task_sha256': file_hash(self.run / 'frozen_tasks.json')}
        write_json(self.run / 'formal_manifest.json', manifest)
        case = self.cases[0]
        # Generate one scripted quality record then explicitly label this transport fixture.
        result = AgentRuntime(ScriptedClient()).run(Session(context), case['turns'][0]['message'])
        result['execution_mode'] = 'cloud'
        for event in result['trace']:
            if event['type'] == 'model':
                event['http_trace'] = [{'http_status': 200}]
        result['request_attempts'] = sum(len(e.get('http_trace', [])) for e in result['trace'])
        item = {'turn_index': 0, 'message': case['turns'][0]['message'], 'expected': case['turns'][0]['expected'],
                'result': result, 'score': score_turn(case['turns'][0]['expected'], result, context,
                    original_hash=entries[0]['pixels_sha256']), 'observed_evidence_ids': list(evidence_map(context)), 'cv_counts': {}}
        self.row = {'task_id': case['id'], 'repetition': 0, 'kind': case['kind'], 'journal_status': 'completed',
                    'completed': True, 'turns': [item], 'source': entries[0]['source'], 'source_index': case['source_index'],
                    'image_sha256': entries[0]['pixels_sha256']}
        budget = RequestBudget(self.run / 'http_budget.json', 600)
        for _ in range(result['request_attempts']):
            budget.reserve()
        self.output = {'manifest_sha256': digest(manifest), 'task_sha256': manifest['task_sha256'],
                       'backend': 'live', 'vision_backend': 'sodd', 'repeats': 3, 'budget': budget.snapshot(),
                       'cases': [self.row], 'metrics': formal_metrics([self.row], self.cases, backend='live', vision_backend='sodd', repeats=3),
                       'vision_metrics': aggregate_vision([self.row])}
        self.output['metrics']['unattributed_reserved_attempts'] = 0
        self.save()

    def save(self):
        write_json(self.run / 'formal_sodd_live.json', self.output)
        write_json(self.run / 'formal_sodd_live/hold_01_0_turn0.json', self.row['turns'][0])

    @patch('optical_agent.llm.requests.post')
    def test_offline_verifier_replays_valid_partial_record_without_requests(self, post):
        verified = verify_formal_artifacts(self.run, self.root)
        self.assertEqual(verified['turns_checked'], 1)
        self.assertEqual(verified['images_checked'], 20)
        self.assertEqual(verified['new_cloud_requests'], 0)
        post.assert_not_called()

    def test_frozen_file_hash_damage_is_rejected(self):
        (self.run / 'frozen_inputs/quality.py').write_text('damaged', encoding='utf-8')
        with self.assertRaises(ValueError):
            verify_formal_artifacts(self.run, self.root)

    def test_label_file_damage_is_rejected(self):
        manifest = json.loads((self.run / 'formal_manifest.json').read_text(encoding='utf-8'))
        Path(manifest['bindings']['images'][0]['label_path']).write_text('0 0.5 0.5 0.3 0.3', encoding='utf-8')
        with self.assertRaises(ValueError):
            verify_formal_artifacts(self.run, self.root)

    def test_unknown_tokens_remain_none_and_tampering_is_rejected(self):
        self.assertFalse(self.row['turns'][0]['result']['usage']['available'])
        self.assertIsNone(self.row['turns'][0]['result']['usage']['total_tokens'])
        verify_formal_artifacts(self.run, self.root)
        self.row['turns'][0]['result']['usage']['total_tokens'] = 1000
        self.save()
        with self.assertRaises(ValueError):
            verify_formal_artifacts(self.run, self.root)

    def test_exhausted_format_repairs_remain_a_valid_failed_record(self):
        detector = FixtureDetector()
        detector.dataset = 'sodd'
        context = ToolContext(detector, np.full((24, 24, 3), 100, np.uint8), KnowledgeRetriever(self.root))
        case = self.cases[0]
        client = WireQueue([intent('quality'), call('assess_image_quality', {'image_id': 'original'}),
                            {'message': {'content': 'invalid'}}])
        result = AgentRuntime(client).run(Session(context), case['turns'][0]['message'])
        result['execution_mode'] = 'cloud'
        for event in result['trace']:
            if event['type'] == 'model':
                event['http_trace'] = [{'http_status': 200}]
        result['request_attempts'] = sum(len(e.get('http_trace', [])) for e in result['trace'])
        self.assertEqual(result['format_repairs'], 2)
        errors = [e for e in result['trace'] if e['type'] == 'format_error']
        self.assertEqual(len(errors), 3)
        self.assertFalse(errors[-1]['will_retry'])
        item = self.row['turns'][0]
        item['result'] = result
        item['score'] = score_turn(item['expected'], result, context, original_hash=self.row['image_sha256'])
        item['observed_evidence_ids'] = list(evidence_map(context))
        self.row['completed'] = False
        budget = RequestBudget(self.run / 'http_budget.json', 600)
        while budget.snapshot()['used'] < result['request_attempts']:
            budget.reserve()
        self.output['budget'] = budget.snapshot()
        self.output['metrics'] = formal_metrics([self.row], self.cases, backend='live', vision_backend='sodd', repeats=3)
        self.output['metrics']['unattributed_reserved_attempts'] = 0
        self.output['vision_metrics'] = aggregate_vision([self.row])
        self.save()
        verified = verify_formal_artifacts(self.run, self.root)
        self.assertEqual(verified['turns_checked'], 1)
        self.assertFalse(item['score']['task_completed'])

    def test_oracle_and_score_tampering_is_rejected_even_when_turn_file_matches(self):
        self.row['turns'][0]['expected']['task_types'] = ['detection']
        self.save()
        with self.assertRaises(ValueError):
            verify_formal_artifacts(self.run, self.root)

    def test_saved_score_not_trusted(self):
        self.row['turns'][0]['score']['task_completed'] = False
        self.save()
        with self.assertRaises(ValueError):
            verify_formal_artifacts(self.run, self.root)

    def test_ledger_or_cv_counts_tampering_is_rejected(self):
        ledger = json.loads((self.run / 'http_budget.json').read_text(encoding='utf-8'))
        ledger['used'] += 1
        write_json(self.run / 'http_budget.json', ledger)
        with self.assertRaises(ValueError):
            verify_formal_artifacts(self.run, self.root)

    def test_cv_counts_tampering_is_rejected(self):
        self.row['turns'][0]['cv_counts'] = {'original': {'tp': 1000}}
        self.save()
        with self.assertRaises(ValueError):
            verify_formal_artifacts(self.run, self.root)
