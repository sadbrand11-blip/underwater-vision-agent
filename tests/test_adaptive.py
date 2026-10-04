import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from optical_agent.adaptive_fixture import AdaptiveScriptedClient
from optical_agent.adaptive_goals import Goal, parse_goal
from optical_agent.adaptive_runtime import AdaptiveRuntime
from optical_agent.adaptive_tools import AdaptiveState, execute, pixel_hash, scope_id, validate_finish
from optical_agent.state import Session
from optical_agent.tools import ToolContext
from test_tasks import WireQueue


class Detector:
    dataset, threshold, calibration = 'sodd', .33, True

    def __init__(self, decline=False, empty=False):
        self.calls, self.decline, self.empty = 0, decline, empty

    def predict(self, image):
        self.calls += 1
        if self.empty:
            return []
        score = .4 if self.decline and image.mean() > 70 else .9
        return [{'class_name': name, 'class_id': cid, 'score': score, 'estimated_box_precision': score,
                 'box': [12, 12, 28, 28]} for name, cid in [('pipe', 6), ('qr_codes', 5)]]


def session(image=None, detector=None):
    return Session(ToolContext(detector or Detector(), np.full((64, 64, 3), 100, np.uint8) if image is None else image))


def response(obj):
    return {'message': {'content': json.dumps(obj, ensure_ascii=False)}, 'http_attempts': 0}


def actions(*items):
    return {'message': {'content': None, 'tool_calls': [
        {'id': f'a{i}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}
        for i, (name, args) in enumerate(items)]}, 'http_attempts': 0}


def goal(kind='analysis', targets=None, policy='if_needed', ref='original', reliable=True):
    return {'task_type': kind, 'image_reference': ref, 'target_classes': ['管道'] if targets is None else targets,
            'correction_policy': policy, 'require_reliability': reliable, 'question': ''}


PLAN = response({'steps': [{'tool': 'assess_image_quality', 'purpose': '先检查实际曝光'}]})


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        import app
        isolation = patch.object(app, '_memory_store', None)
        isolation.start()
        self.addCleanup(isolation.stop)

    def test_class_filter_counts_images_and_raw_cache_are_consistent(self):
        s = session()
        result = AdaptiveRuntime(AdaptiveScriptedClient()).run(s, '只识别管道')
        self.assertEqual(result['task_status'], 'completed')
        self.assertEqual(result['target_count'], 1)
        self.assertEqual([b['class_name'] for b in result['detections']], ['pipe'])
        self.assertEqual(len(s.adaptive.raw_detections['original']), 2)
        calls = s.context.detector.calls
        qr = AdaptiveRuntime(AdaptiveScriptedClient()).run(s, '只识别二维码')
        self.assertEqual(qr['detections'][0]['class_name'], 'qr_codes')
        self.assertEqual(s.context.detector.calls, calls)
        self.assertEqual(qr['computation_counts']['detection'], 0)
        self.assertTrue(any(e.get('cached') for e in qr['trace'] if e['type'] == 'tool'))

    def test_three_input_paths_and_original_unchanged(self):
        routes = []
        for value in [100, 45, 0]:
            s = session(np.full((64, 64, 3), value, np.uint8))
            before = pixel_hash(s.context.images['original'])
            result = AdaptiveRuntime(AdaptiveScriptedClient()).run(s, '检查曝光，必要时校正，然后只识别管道，说明是否可靠')
            self.assertEqual(result['task_status'], 'completed', result['answer'])
            self.assertEqual(pixel_hash(s.adaptive.images['original']), before)
            tools = [e['tool'] for e in result['trace'] if e['type'] == 'tool']
            routes.append(tools)
            if value == 0:
                self.assertNotIn('generate_exposure_candidate', tools)
                self.assertNotIn('detect_objects', tools)
                self.assertEqual(result['vision_status'], 'quality_failure')
            elif value == 100:
                self.assertNotIn('generate_exposure_candidate', tools)
                self.assertIn('assess_reliability', tools)
                self.assertIn('single_image', result['vision_result']['evidence_scope'])
            else:
                self.assertIn('compare_candidates', tools)
        self.assertNotEqual(routes[0], routes[1])
        self.assertNotEqual(routes[1], routes[2])

    def test_plan_text_alone_cannot_finish_and_budget_counts_planning(self):
        client = WireQueue([response(goal()), PLAN, response({'selected_image_id': 'original', 'evidence_ids': [], 'citation_ids': []})])
        result = AdaptiveRuntime(client, max_model_calls=3).run(session(), '分析并识别管道')
        self.assertEqual(result['task_status'], 'incomplete')
        self.assertEqual(result['tool_calls'], 0)
        self.assertFalse(result['task_validation']['passed'])
        self.assertEqual(result['model_calls'], 3)
        self.assertFalse(next(e for e in result['trace'] if e['type'] == 'initial_plan')['executed'])

    def test_declining_candidate_replans_using_real_observation_and_stops(self):
        s = session(np.full((64, 64, 3), 45, np.uint8), Detector(decline=True))
        targets = Goal('analysis', 'original', ('pipe_type2', 'pipe'), 'if_needed', True)
        suffix = scope_id(targets)
        cmp1 = 'comparison:candidate_gamma:' + suffix
        cmp2 = 'comparison:candidate_gamma+candidate_local:' + suffix
        revision = {'reason': '候选的检测证据下降，尝试局部方法再比较', 'observation_ids': [cmp1],
                    'steps': [{'tool': 'generate_exposure_candidate', 'purpose': '从原图生成另一种候选'},
                              {'tool': 'compare_candidates', 'purpose': '按实际证据比较'}]}
        client = WireQueue([response(goal()), PLAN,
            actions(('assess_image_quality', {'image_id': 'original'})),
            actions(('generate_exposure_candidate', {'image_id': 'original', 'method': 'gamma_only'}), ('detect_objects', {'image_id': 'original'})),
            actions(('assess_image_quality', {'image_id': 'candidate_gamma'}), ('detect_objects', {'image_id': 'candidate_gamma'})),
            actions(('compare_candidates', {'candidate_image_ids': ['candidate_gamma']})),
            actions(('revise_plan', revision), ('generate_exposure_candidate', {'image_id': 'original', 'method': 'local_bounded'}),
                    ('assess_image_quality', {'image_id': 'candidate_local'}), ('detect_objects', {'image_id': 'candidate_local'}),
                    ('compare_candidates', {'candidate_image_ids': ['candidate_gamma', 'candidate_local']})),
            response({'selected_image_id': 'original', 'evidence_ids': ['quality:original', 'detections:original:' + suffix, cmp2], 'citation_ids': []})])
        result = AdaptiveRuntime(client).run(s, '校正没有帮助则保留原图，只分析管道')
        self.assertEqual(result['task_status'], 'completed', result['answer'])
        self.assertEqual(len(result['plan_revisions']), 1)
        self.assertEqual(result['selected_image_id'], 'original')
        self.assertEqual(result['model_calls'], 8)
        self.assertEqual(result['tool_calls'], 11)
        self.assertEqual(result['computation_counts']['correction'], 2)
        self.assertTrue(s.adaptive.observations[cmp1]['data']['reports'][0]['effect_degraded'])

    def test_candidate_limit_cache_and_fabricated_replanning(self):
        state = AdaptiveState(session(np.full((64, 64, 3), 45, np.uint8)).context)
        state.begin(Goal('analysis', 'original', ('pipe',), 'if_needed', True))
        execute(state, 'assess_image_quality', {'image_id': 'original'})
        for method in ('gamma_only', 'local_bounded', 'gamma_only'):
            self.assertTrue(execute(state, 'generate_exposure_candidate', {'image_id': 'original', 'method': method})['ok'])
        self.assertEqual(state.counts['correction'], 2)
        self.assertTrue(state.events[-1]['cached'])
        self.assertFalse(execute(state, 'generate_exposure_candidate', {'image_id': 'candidate_gamma', 'method': 'gamma_only'})['ok'])
        self.assertFalse(execute(state, 'revise_plan', {'reason': '虚构观察', 'observation_ids': ['not_real'], 'steps': [{'tool': 'detect_objects', 'purpose': '识别'}]})['ok'])
        for _ in range(2):
            self.assertTrue(execute(state, 'revise_plan', {'reason': '根据质量证据', 'observation_ids': ['quality:original'], 'steps': [{'tool': 'detect_objects', 'purpose': '识别'}]})['ok'])
        self.assertFalse(execute(state, 'revise_plan', {'reason': '继续', 'observation_ids': ['quality:original'], 'steps': [{'tool': 'detect_objects', 'purpose': '识别'}]})['ok'])

    def test_old_target_evidence_and_wrong_image_cannot_finish(self):
        s = session()
        AdaptiveRuntime(AdaptiveScriptedClient()).run(s, '只识别管道')
        old = next(oid for oid in s.adaptive.observations if oid.startswith('detections:'))
        s.adaptive.begin(Goal('detection', 'original', ('qr_codes',), 'never', False))
        selection = {'selected_image_id': 'original', 'evidence_ids': [old], 'citation_ids': []}
        self.assertFalse(validate_finish(s.adaptive, selection)['passed'])
        s.adaptive.begin(Goal('quality', 'original', ('pipe',), 'never', False))
        self.assertFalse(validate_finish(s.adaptive, selection)['passed'])

    def test_empty_detection_unsupported_and_quality_do_not_invent_detection(self):
        s = session(detector=Detector(empty=True))
        empty = AdaptiveRuntime(AdaptiveScriptedClient()).run(s, '只识别管道')
        self.assertEqual(empty['target_count'], 0)
        self.assertEqual(empty['task_status'], 'completed')
        quality = AdaptiveRuntime(AdaptiveScriptedClient()).run(s, '只检查曝光')
        self.assertIsNone(quality['target_count'])
        self.assertEqual(quality['detections'], [])
        unsupported = AdaptiveRuntime(AdaptiveScriptedClient()).run(s, '识别鱼')
        self.assertEqual(unsupported['task_status'], 'unsupported')
        self.assertEqual(unsupported['tool_calls'], 0)
        self.assertEqual(unsupported['plan_revisions'], [])

    def test_corrected_followup_pending_and_session_isolation(self):
        s = session(np.full((64, 64, 3), 45, np.uint8))
        first = AdaptiveRuntime(AdaptiveScriptedClient()).run(s, '识别校正后的图')
        self.assertEqual(first['task_status'], 'needs_clarification')
        AdaptiveRuntime(AdaptiveScriptedClient()).run(s, '先校正然后识别管道')
        follow = AdaptiveRuntime(AdaptiveScriptedClient()).run(s, '只识别校正后的图中的二维码')
        self.assertEqual(follow['goal_contract']['image_id'], 'candidate_gamma')
        self.assertEqual(follow['selected_image_id'], 'candidate_gamma')
        fresh = AdaptiveRuntime(AdaptiveScriptedClient()).run(session(), '识别校正后的图')
        self.assertEqual(fresh['task_status'], 'needs_clarification')

    def test_unknown_tool_bad_parameters_and_secret_safe_log(self):
        s = session()
        client = WireQueue([response(goal()), PLAN, actions(('not_registered', {})), actions(('detect_objects', {'image_id': 'other'}))])
        client.api_key = 'sk-secret-test-value-012345'
        with tempfile.TemporaryDirectory() as folder:
            result = AdaptiveRuntime(client, log_dir=folder).run(s, '分析管道')
            self.assertEqual(result['execution_status'], 'tool_errors')
            self.assertFalse(any(e.get('executed') for e in result['trace'] if e.get('tool') == 'not_registered'))
            self.assertNotIn(client.api_key, next(Path(folder).glob('*.jsonl')).read_text(encoding='utf-8'))

    def test_web_opt_in_and_legacy_compatibility(self):
        import app
        from agent import OpticalAgent
        app._agent = OpticalAgent(Detector())
        client = app.app.test_client()
        encoded = cv2.imencode('.png', np.full((64, 64, 3), 100, np.uint8))[1].tobytes()
        sid = client.post('/api/sessions', data={'image': (io.BytesIO(encoded), 'fixture.png')}).json['session_id']
        new = client.post('/api/chat', json={'session_id': sid, 'message': '只识别管道', 'mode': 'scripted', 'agent_mode': 'adaptive'})
        self.assertEqual(new.status_code, 200)
        self.assertEqual(new.json['target_count'], 1)
        image_ref = next(url for key, url in new.json['image_refs'].items() if key.startswith('adaptive_boxes'))
        self.assertEqual(client.get(image_ref).status_code, 200)
        old = client.post('/api/chat', json={'session_id': sid, 'message': '只检查曝光', 'mode': 'scripted'})
        self.assertEqual(old.json['agent_mode'], 'legacy')
        self.assertNotIn('goal_contract', old.json)
