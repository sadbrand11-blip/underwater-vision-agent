import copy
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from optical_agent.adaptive_fixture import AdaptiveScriptedClient
from optical_agent.adaptive_goals import parse_goal
from optical_agent.adaptive_runtime import AdaptiveRuntime
from optical_agent.adaptive_tools import AdaptiveState, validate_finish
from optical_agent.llm import ScriptedClient
from optical_agent.memory import MemoryStore, MemoryTurn, MemoryUnavailable, image_hash, versions_for
from optical_agent.runtime import AgentRuntime
from optical_agent.state import Session, SessionStore
from optical_agent.tools import ToolContext
from test_adaptive import Detector, session
from test_tasks import WireQueue


class VersionedDetector(Detector):
    model_sha256 = 'fixture_weight_v1_not_a_real_model'
    classes = ('background', 'propeller', 'pipe_type2', 'red_fin', 'net', 'qr_codes', 'pipe')


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.store = MemoryStore(Path(self.folder.name) / 'memory.sqlite3')
        self.prefs = {'enabled': True, 'target_classes': ['管道'], 'display_detail': 'expanded'}
        self.store.set_preferences(self.prefs)

    def run_task(self, message='只识别目标', value=100, store=None, detector=None):
        s = session(np.full((64, 64, 3), value, np.uint8), detector or VersionedDetector())
        result = AdaptiveRuntime(AdaptiveScriptedClient(), memory_store=store or self.store).run(s, message)
        self.assertEqual(result['task_status'], 'completed', result['answer'])
        return s, result

    def test_restart_persists_preferences_and_facts_and_new_image_computes(self):
        first, r1 = self.run_task()
        reopened = MemoryStore(self.store.path)
        self.assertEqual(reopened.preferences()['target_classes'], ['pipe_type2', 'pipe'])
        second, r2 = self.run_task(value=110, store=reopened)
        self.assertEqual(len(r2['memory_context']['history_refs']), 1)
        self.assertEqual(second.context.detector.calls, 1)
        self.assertNotEqual(image_hash(first.context.images['original']), image_hash(second.context.images['original']))
        self.assertTrue(all(not x['evidence_id'].startswith('memory:') for x in r2['evidence']))
        self.assertEqual(len(reopened.list_history()), 2)

    def test_explicit_all_and_quality_override_default_classes(self):
        _, r = self.run_task('只识别二维码')
        self.assertEqual(r['goal_contract']['target_classes'], ['qr_codes'])
        self.assertNotIn('target_classes', r['memory_context']['applied_preferences'])
        _, r = self.run_task('识别全部类别')
        self.assertEqual(len(r['goal_contract']['target_classes']), 6)
        s, r = self.run_task('只检查曝光')
        self.assertEqual(s.context.detector.calls, 0)
        self.assertNotIn('target_classes', r['memory_context']['applied_preferences'])
        self.assertEqual([x['tool'] for x in r['trace'] if x['type'] == 'tool'], ['assess_image_quality'])
        self.assertEqual(self.store.list_history()[0]['target_classes'], [])

    def test_unavailable_preference_clarifies_instead_of_silently_dropping(self):
        self.store.set_preferences(dict(self.prefs, target_classes=['underwater_robot']))
        s = session(detector=VersionedDetector())
        r = AdaptiveRuntime(AdaptiveScriptedClient(), memory_store=self.store).run(s, '只识别目标')
        self.assertEqual(r['task_status'], 'needs_clarification')
        self.assertIn('默认类别在当前模型中不可用', r['answer'])
        self.assertEqual(s.context.detector.calls, 0)
        self.assertEqual(self.store.list_history(), [])

    def test_empty_detection_and_quality_failure_are_honest_completed_history(self):
        _, empty = self.run_task(detector=VersionedDetector(empty=True))
        row = self.store.list_history()[0]
        self.assertEqual(row['facts']['selected_candidate_count'], 0)
        self.assertIsNone(row['facts']['visual_status'])
        s, r = self.run_task('检查曝光，必要时校正并识别目标，说明是否可靠', value=0)
        self.assertEqual(r['vision_status'], 'quality_failure')
        self.assertEqual(s.context.detector.calls, 0)
        self.assertEqual(self.store.list_history()[0]['facts']['visual_status'], 'quality_failure')
        self.assertIsNone(self.store.list_history()[0]['facts']['selected_candidate_count'])

    def test_unfinished_and_plan_only_do_not_save(self):
        r = AdaptiveRuntime(AdaptiveScriptedClient(), max_model_calls=2, memory_store=self.store).run(
            session(detector=VersionedDetector()), '只识别目标')
        self.assertEqual(r['task_status'], 'incomplete')
        self.assertEqual(r['memory_context']['save_status'], 'not_completed')
        self.assertEqual(self.store.list_history(), [])

    def test_duplicate_is_reused_and_max_200_eviction_leaves_preferences(self):
        _, r1 = self.run_task()
        _, r2 = self.run_task()
        self.assertEqual(r2['memory_context']['save_status'], 'reused')
        self.assertEqual(r1['memory_context']['saved_memory_id'], r2['memory_context']['saved_memory_id'])
        small = MemoryStore(self.store.path, limit=2)
        self.run_task(value=110, store=small)
        self.run_task(value=120, store=small)
        self.assertEqual(len(small.list_history()), 2)
        self.assertEqual(small.preferences()['target_classes'], ['pipe_type2', 'pipe'])

    def test_version_and_scope_mismatch_do_not_reference_old_results(self):
        self.run_task()
        d = VersionedDetector(); d.model_sha256 = 'different_weight'
        _, r = self.run_task(value=111, detector=d)
        self.assertEqual(r['memory_context']['history_refs'], [])
        _, r = self.run_task('只识别二维码', value=112)
        self.assertEqual(r['memory_context']['history_refs'], [])
        s = session(detector=VersionedDetector())
        s.rag_metadata = {'mode': 'hybrid', 'corpus_hash': 'different_corpus'}
        r = AdaptiveRuntime(AdaptiveScriptedClient(), memory_store=self.store).run(s, '只识别目标')
        self.assertEqual(r['memory_context']['history_refs'], [])

    def test_missing_stable_model_hash_does_not_enable_history_reuse(self):
        self.run_task()
        _, r = self.run_task(detector=Detector())
        self.assertFalse(self.store.list_history()[0]['versions']['model_identity_verified'])
        self.assertEqual(r['memory_context']['history_refs'], [])

    def test_preferences_and_history_are_pinned_at_task_start(self):
        self.run_task()
        s = session(detector=VersionedDetector())
        turn = MemoryTurn(self.store, s, 'adaptive')
        self.store.set_preferences(dict(self.prefs, target_classes=['qr_codes'], display_detail='compact'))
        self.store.delete()
        state = AdaptiveState(s.context)
        goal_json = {'task_type':'detection','image_reference':'original','target_classes':[],
                     'correction_policy':'never','require_reliability':False,'question':'','target_scope':'default'}
        g = parse_goal(json.dumps(goal_json), state, 'original', turn.goal_preferences)
        turn.retrieve(g.as_dict(), '只识别目标', 'default')
        self.assertEqual(g.target_classes, ('pipe_type2', 'pipe'))
        self.assertEqual(len(turn.context['history_refs']), 1)
        self.assertEqual(turn.context['preferences']['display_detail'], 'expanded')

    def test_history_only_enters_planning_background_and_cannot_finish_task(self):
        self.run_task()
        captured = []
        client = AdaptiveScriptedClient()
        original = client.complete
        def complete(messages, *args, **kwargs):
            captured.append(copy.deepcopy(messages))
            return original(messages, *args, **kwargs)
        client.complete = complete
        s = session(detector=VersionedDetector())
        r = AdaptiveRuntime(client, memory_store=self.store).run(s, '只识别目标')
        self.assertNotIn('historical_summaries', json.dumps(captured[0]))
        self.assertIn('historical_summaries', json.dumps(captured[1]))
        self.assertEqual(r['tool_calls'], 1)
        memory_id = r['memory_context']['history_refs'][0]['memory_id']
        with self.assertRaises(ValueError):
            validate_finish(s.adaptive, {'selected_image_id':'original','evidence_ids':[memory_id],'citation_ids':[]})
        self.assertLessEqual(sum(len(x['summary']) for x in r['memory_context']['history_refs']), 1200)

    def test_disable_preserves_history_and_old_behavior(self):
        self.run_task()
        self.store.set_preferences(dict(self.prefs, enabled=False))
        s = session(detector=VersionedDetector())
        r = AdaptiveRuntime(AdaptiveScriptedClient(), memory_store=self.store).run(s, '识别全部类别')
        baseline = AdaptiveRuntime(AdaptiveScriptedClient()).run(session(detector=VersionedDetector()), '识别全部类别')
        for key in ('goal_contract','detections','tool_calls','model_calls','task_validation','vision_status'):
            self.assertEqual(r[key], baseline[key])
        self.assertEqual(len(self.store.list_history()), 1)
        self.assertEqual(r['memory_context']['save_status'], 'disabled')

    def test_database_failure_read_and_write_never_fails_visual_task(self):
        broken = Path(self.folder.name) / 'broken.sqlite3'; broken.write_bytes(b'not a sqlite database')
        _, r = self.run_task('只识别管道', store=MemoryStore(broken))
        self.assertFalse(r['memory_context']['available'])
        self.assertEqual(r['target_count'], 1)
        with patch.object(self.store, 'save', side_effect=MemoryUnavailable('write failed')):
            _, r = self.run_task()
        self.assertEqual(r['memory_context']['save_status'], 'unavailable')
        self.assertEqual(r['task_status'], 'completed')

    def test_concurrent_dedup_delete_search_and_clear(self):
        self.run_task()
        payload = self.store.list_history()[0]
        payload = {k:v for k,v in payload.items() if k not in {'memory_id','created_at'}}
        self.store.delete()  # Exercise simultaneous first inserts, not an existing-row lookup.
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _: self.store.save(payload), range(8)))
        self.assertEqual(len({x['memory_id'] for x in results}), 1)
        self.assertEqual(len(self.store.list_history('管道')), 1)
        self.assertEqual(self.store.list_history('ZZUNRELATEDZZ'), [])
        self.assertEqual(self.store.delete(results[0]['memory_id']), 1)
        self.assertEqual(self.store.delete(results[0]['memory_id']), 0)
        self.run_task(); self.assertEqual(self.store.delete(), 1)
        self.assertEqual(self.store.preferences()['target_classes'], ['pipe_type2','pipe'])

    def test_corrupt_and_future_schema_are_not_reset(self):
        original = self.store.preferences()
        with closing(sqlite3.connect(str(self.store.path))) as conn, conn:
            conn.execute('PRAGMA user_version=2')
        with self.assertRaises(MemoryUnavailable): self.store.preferences()
        with closing(sqlite3.connect(str(self.store.path))) as conn, conn:
            self.assertEqual(conn.execute('PRAGMA user_version').fetchone()[0], 2)
            self.assertEqual(json.loads(conn.execute('SELECT payload FROM preferences').fetchone()[0]), original)

    def test_visual_config_mismatch_and_stable_array_calibration(self):
        detector = VersionedDetector(); detector.calibration = {'x': np.array([0.,1.]), 'y': np.array([0.,.9])}
        self.run_task(detector=detector)
        clone = VersionedDetector(); clone.calibration = {'x': np.array([0.,1.]), 'y': np.array([0.,.9])}
        _, r = self.run_task(value=120, detector=clone)
        self.assertTrue(r['memory_context']['history_refs'])
        changed = VersionedDetector(); changed.threshold = .8; changed.calibration = clone.calibration
        _, r = self.run_task(value=121, detector=changed)
        self.assertEqual(r['memory_context']['history_refs'], [])

    def test_same_counts_but_different_real_boxes_are_not_deduplicated(self):
        self.run_task()
        d = VersionedDetector()
        original = d.predict
        def predict(image):
            boxes = original(image)
            for box in boxes: box['box'] = [20, 20, 40, 40]
            return boxes
        d.predict = predict
        _, r = self.run_task(detector=d)
        self.assertEqual(r['memory_context']['save_status'], 'saved')
        self.assertEqual(len(self.store.list_history()), 2)

    def test_missing_robot_branch_preference_clarifies(self):
        from optical_agent.vision_router import DetectorRouter
        self.store.set_preferences(dict(self.prefs, target_classes=['underwater_robot']))
        s = session(detector=DetectorRouter(VersionedDetector(), None))
        r = AdaptiveRuntime(AdaptiveScriptedClient(), memory_store=self.store).run(s, '只识别目标')
        self.assertEqual(r['task_status'], 'needs_clarification')
        self.assertEqual(r['tool_calls'], 0)

    def test_session_expiry_does_not_delete_memory(self):
        self.run_task()
        clock = [0]
        sessions = SessionStore(clock=lambda:clock[0])
        s = sessions.create(VersionedDetector(), np.full((64,64,3),100,np.uint8))
        clock[0] = 1801
        with self.assertRaises(KeyError): sessions.get(s.id)
        self.assertEqual(len(self.store.list_history()), 1)

    def test_legacy_archives_actual_comparison_but_ignores_target_preferences(self):
        s = session(detector=VersionedDetector())
        r = AgentRuntime(ScriptedClient(), memory_store=self.store).run(s, '比较校正前后')
        self.assertEqual(r['task_status'], 'completed', r['answer'])
        self.assertEqual(r['memory_context']['applied_preferences'], {})
        self.assertEqual(r['memory_context']['history_refs'], [])
        self.assertEqual(self.store.list_history()[0]['task_type'], 'comparison')
        self.assertTrue(self.store.list_history()[0]['facts']['qualities'])

    def test_records_contain_no_pixels_boxes_full_chat_or_secret(self):
        client = AdaptiveScriptedClient(); client.api_key = 'unit-test-key'
        s = session(detector=VersionedDetector())
        s.input_provenance = {'kind':'simulated_exposure_perturbation','variant':'unit-test-key'}
        r = AdaptiveRuntime(client, memory_store=self.store).run(s, '只识别目标；PRIVATE_FULL_CHAT_MARKER')
        serialized = json.dumps(self.store.list_history())
        self.assertNotIn('unit-test-key', serialized)
        self.assertNotIn('PRIVATE_FULL_CHAT_MARKER', serialized)
        self.assertNotIn('"box"', serialized)
        self.assertNotIn('"messages"', serialized)
        self.assertNotIn('unit-test-key', json.dumps(r))


class MemoryWebTests(unittest.TestCase):
    def setUp(self):
        import app as web
        self.web = web
        self.folder = tempfile.TemporaryDirectory(); self.addCleanup(self.folder.cleanup)
        self.store = MemoryStore(Path(self.folder.name)/'memory.sqlite3')
        for name,value in [('_memory_store',self.store),('_sessions',SessionStore())]:
            p = patch.object(web,name,value); p.start(); self.addCleanup(p.stop)
        self.client = web.app.test_client()

    def test_preferences_validation_history_routes_and_local_access(self):
        self.assertEqual(self.client.get('/memory').status_code, 200)
        for payload in ({'enabled':True}, {'enabled':1,'target_classes':[],'display_detail':'compact'},
                        {'enabled':True,'target_classes':['fish'],'display_detail':'compact'},
                        {'enabled':True,'target_classes':[],'display_detail':{}}):
            self.assertEqual(self.client.put('/api/memory/preferences',json=payload).status_code,400)
        self.assertEqual(self.client.put('/api/memory/preferences',json={'enabled':True,'target_classes':['管道'],'display_detail':'compact'}).status_code,200)
        self.assertEqual(self.client.get('/api/memory/preferences').json['preferences']['target_classes'],['pipe_type2','pipe'])
        for method,url in [('get','/api/memory/history'),('put','/api/memory/preferences'),('delete','/api/memory/history')]:
            self.assertEqual(getattr(self.client,method)(url,headers={'Origin':'https://other.example'}).status_code,403)
        self.assertEqual(self.client.get('/api/memory/history?q='+'x'*401).status_code,400)
        self.assertEqual(self.client.delete('/api/memory/history/invalid').status_code,400)
        self.assertEqual(self.client.delete('/api/memory/history/memory:unknown').status_code,404)
        self.assertEqual(self.client.delete('/api/memory/history').status_code,200)

    def test_api_chat_defaults_to_memory_without_paid_requests(self):
        from agent import OpticalAgent
        self.store.set_preferences({'enabled':True,'target_classes':['管道'],'display_detail':'compact'})
        with patch.object(self.web,'get_vision_agent',return_value=OpticalAgent(VersionedDetector())),\
             patch.object(self.web,'get_retriever',return_value=None):
            encoded = cv2.imencode('.png',np.full((64,64,3),100,np.uint8))[1].tobytes()
            sid = self.client.post('/api/sessions',data={'image':(io.BytesIO(encoded),'image.png')}).json['session_id']
        r = self.client.post('/api/chat',json={'session_id':sid,'mode':'scripted','agent_mode':'adaptive','message':'只识别目标'})
        self.assertEqual(r.status_code,200)
        self.assertEqual(r.json['memory_context']['save_status'],'saved')
        self.assertEqual(r.json['request_attempts'],0)
        self.assertEqual(len(self.client.get('/api/memory/history').json['history']),1)
        memory_id = r.json['memory_context']['saved_memory_id']
        self.assertEqual(self.client.delete('/api/memory/history/'+memory_id).json['deleted'],1)


if __name__ == '__main__': unittest.main()
