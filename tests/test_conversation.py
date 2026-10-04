import io
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock, patch

import cv2
import numpy as np

import app as web
from agent import OpticalAgent
from optical_agent.rag import KnowledgeRetriever
from optical_agent.state import SessionStore
from optical_agent.llm import CloudClient
from optical_agent.tasks import scripted_intent
from test_agent_tools import FakeDetector


class ConversationTests(unittest.TestCase):
    def setUp(self):
        isolation = patch.object(web, '_memory_store', None)
        isolation.start()
        self.addCleanup(isolation.stop)
        web._agent = OpticalAgent(FakeDetector())
        web._sessions = SessionStore()
        self.client = web.app.test_client()

    def upload(self, value=100):
        _, encoded = cv2.imencode('.png', np.full((64, 64, 3), value, np.uint8))
        return self.client.post('/api/sessions', data={'image': (io.BytesIO(encoded.tobytes()), 'test.png')})

    def test_rag_sources_and_insufficient_evidence(self):
        retriever = KnowledgeRetriever()
        matches = retriever.search('曝光校正为什么失败，信息丢失与拒识原因')
        self.assertTrue(matches)
        self.assertLessEqual(len(matches), 3)
        self.assertIn('verified_at', matches[0])
        self.assertEqual(retriever.search('ΩΩΩΨΨΨ'), [])

    def test_external_origin_and_host_cannot_access_local_api(self):
        for headers in ({'Origin':'http://attacker.example'}, {'Host':'attacker.example'},
                        {'Host':'attacker.example','Origin':'http://attacker.example'}):
            self.assertEqual(self.client.post('/api/sessions',headers=headers).status_code,403)
            self.assertEqual(self.client.post('/api/chat',headers=headers,json={}).status_code,403)
        self.assertEqual(self.client.get('/health',headers={'Origin':'http://localhost'}).status_code,200)

    def test_upload_chat_followup_and_preview(self):
        response = self.upload()
        self.assertEqual(response.status_code, 201)
        sid = response.json['session_id']
        answer = self.client.post('/api/chat', json={'session_id': sid, 'message': '比较校正前后识别改善', 'mode': 'scripted'})
        self.assertEqual(answer.json['execution_status'], 'completed')
        self.assertEqual(answer.json['task_contract']['task_type'], 'comparison')
        self.assertEqual(answer.json['task_status'], 'completed')
        self.assertTrue(answer.json['task_validation']['passed'])
        self.assertIn('corrected', answer.json['image_refs'])
        why = self.client.post('/api/chat', json={'session_id': sid, 'message': '解释不能接受的原因', 'mode': 'scripted'})
        self.assertEqual(why.json['execution_status'], 'completed')
        self.assertTrue(why.json['citations'])
        preview = self.client.get(answer.json['image_refs']['original'])
        self.assertEqual(preview.mimetype, 'image/jpeg')

    def test_sessions_and_invalid_inputs(self):
        s1, s2 = self.upload(100).json['session_id'], self.upload(200).json['session_id']
        self.assertNotEqual(self.client.get(f'/api/sessions/{s1}/images/original').data,
                            self.client.get(f'/api/sessions/{s2}/images/original').data)
        self.assertEqual(self.client.get(f'/api/sessions/{s1}/images/corrected').status_code, 404)
        self.assertEqual(self.client.post('/api/chat', json={'session_id': 'bad', 'message': '曝光'}).status_code, 404)
        self.assertEqual(self.client.post('/api/chat', json=[]).status_code, 400)
        self.assertEqual(self.client.get('/agent').status_code, 200)

    def test_simultaneous_preview_encoding_does_not_block_other_images(self):
        sid = self.upload().json['session_id']
        encoding_started, release = Event(), Event()
        original_encode = cv2.imencode
        def slow_encode(*args, **kwargs):
            encoding_started.set()
            if not release.wait(3):
                raise RuntimeError('Test preview encoding was not released')
            return original_encode(*args, **kwargs)
        def request_image():
            with web.app.test_client() as client:
                return client.get(f'/api/sessions/{sid}/images/original').status_code
        with ThreadPoolExecutor(max_workers=1) as pool:
            with patch.object(web.cv2, 'imencode', slow_encode):
                future = pool.submit(request_image)
                self.assertTrue(encoding_started.wait(3))
                session = web._sessions.get(sid)
                free = session.lock.acquire(blocking=False)
                if free:
                    session.lock.release()
                release.set()
                self.assertEqual(future.result(), 200)
                self.assertTrue(free, 'JPEG encoding must not retain the session lock')

    def test_cloud_error_details_show_upstream_code_despite_local_200(self):
        sid = self.upload().json['session_id']
        for code in (401, 402):
            with self.subTest(code=code), patch('app.CloudClient', return_value=CloudClient(api_key='fixture-key')):
                with patch('optical_agent.llm.requests.post', return_value=Mock(status_code=code, headers={})) as post:
                    response = self.client.post('/api/chat', json={'session_id': sid, 'message': '只检查曝光', 'mode': 'cloud'})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json['error_details']['http_status'], code)
                    self.assertEqual(response.json['request_attempts'], 1)
                    self.assertEqual(response.json['model_calls'], 0)
                    self.assertEqual(response.json['tool_calls'], 0)
                    self.assertEqual(post.call_count, 1)

    def test_cloud_200_invalid_finish_keeps_report_validation_stage(self):
        sid = self.upload().json['session_id']
        wire = Mock(status_code=200, headers={}, json=Mock(return_value={
            'choices': [{'finish_reason': 'stop', 'message': {
                'content': '{"evidence_ids":["quality:original"],"citation_ids":[]}'}}]}))
        with patch('app.CloudClient', return_value=CloudClient(api_key='fixture-key')):
            intent = Mock(status_code=200, headers={}, json=Mock(return_value={
                'choices': [{'message': {'content': json.dumps(scripted_intent('只检查曝光'))}}]}))
            with patch('optical_agent.llm.requests.post', side_effect=[intent, wire, wire, wire]) as post:
                response = self.client.post('/api/chat', json={'session_id': sid, 'message': '只检查曝光', 'mode': 'cloud'})
        result = response.json
        self.assertEqual(result['execution_status'], 'invalid_response')
        self.assertEqual(result['error_details']['http_status'], 200)
        self.assertEqual(result['error_details']['stage'], 'model_finish_validation')
        self.assertEqual(result['request_attempts'], 4)
        self.assertEqual(result['format_repairs'], 2)
        self.assertEqual(result['tool_calls'], 0)
        self.assertEqual(post.call_count, 4)
        self.assertNotIn('fixture-key', response.get_data(as_text=True))

    def test_api_clarification_then_correction_and_corrected_detection(self):
        sid = self.upload().json['session_id']
        def ask(message):
            return self.client.post('/api/chat', json={'session_id': sid, 'message': message, 'mode': 'scripted'}).json
        missing = ask('识别校正后的图')
        self.assertEqual(missing['task_status'], 'needs_clarification')
        self.assertEqual(missing['tool_calls'], 0)
        corrected = ask('先校正原图')
        self.assertEqual(corrected['task_status'], 'completed')
        self.assertEqual(corrected['task_contract']['task_type'], 'correction')
        self.assertIn('corrected', corrected['image_refs'])
        detections = ask('识别校正后的图')
        self.assertEqual(detections['task_status'], 'completed')
        self.assertEqual(detections['evidence'][0]['evidence_id'], 'detections:corrected')
        unsupported = ask('识别图中的珊瑚')
        self.assertEqual(unsupported['task_status'], 'unsupported')
        self.assertEqual(unsupported['tool_calls'], 0)
