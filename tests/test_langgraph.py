"""Independent parity checks plus the existing behavioral contract on both engines."""
import copy
import io
import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import test_adaptive as fixture
import test_memory as memory_fixture
from optical_agent.adaptive_fixture import AdaptiveScriptedClient
from optical_agent.adaptive_runtime import AdaptiveRuntime
from optical_agent.langgraph_runtime import LangGraphRuntime
from optical_agent.llm import LLMError
from optical_agent.memory import MemoryStore, MemoryTurn
from test_tasks import WireQueue

AVAILABLE = find_spec('langgraph') is not None


from optical_agent.runtime_compare import core, without_timings


class Recorder(AdaptiveScriptedClient):
    def __init__(self):
        self.requests = []

    def complete(self, messages, tools=None, timeout=30, **kwargs):
        self.requests.append(copy.deepcopy((messages, tools, kwargs)))
        return super().complete(messages, tools, timeout, **kwargs)


@unittest.skipUnless(AVAILABLE, 'optional LangGraph dependency not installed')
class GraphAdaptiveContractTests(fixture.AdaptiveTests):
    def setUp(self):
        super().setUp()
        p = patch.object(fixture, 'AdaptiveRuntime', LangGraphRuntime)
        p.start()
        self.addCleanup(p.stop)


@unittest.skipUnless(AVAILABLE, 'optional LangGraph dependency not installed')
class GraphMemoryContractTests(memory_fixture.MemoryTests):
    def setUp(self):
        super().setUp()
        p = patch.object(memory_fixture, 'AdaptiveRuntime', LangGraphRuntime)
        p.start()
        self.addCleanup(p.stop)


@unittest.skipUnless(AVAILABLE, 'optional LangGraph dependency not installed')
class ParityTests(unittest.TestCase):
    def test_same_requests_and_outputs_for_independent_sessions(self):
        cases = [(100, ['只检查曝光'], False), (45, ['只校正曝光'], False),
                 (100, ['只识别管道'], False),
                 (100, ['检查这张水下图，必要时校正曝光，然后只识别管道，说明结果是否可靠。'], False),
                 (45, ['检查这张水下图，必要时校正曝光，然后只识别管道，说明结果是否可靠。'], False),
                 (0, ['检查这张水下图，必要时校正曝光，然后只识别管道，说明结果是否可靠。'], False),
                 (45, ['识别校正后的管道', '先校正再检测管道', '只识别校正后的管道'], False),
                 (45, ['比较校正前后管道', '比较校正前后管道'], True)]
        for value, messages, decline in cases:
            with self.subTest(value=value, messages=messages):
                a, b = Recorder(), Recorder()
                sa = fixture.session(np.full((64,64,3), value, np.uint8), fixture.Detector(decline=decline))
                sb = fixture.session(np.full((64,64,3), value, np.uint8), fixture.Detector(decline=decline))
                for message in messages:
                    native = AdaptiveRuntime(a).run(sa, message)
                    graph = LangGraphRuntime(b).run(sb, message)
                    self.assertEqual(core(native), core(graph))
                    self.assertGreater(len(graph['graph_trace']), 0)
                    self.assertEqual(without_timings(a.requests, parse_json=True), without_timings(b.requests, parse_json=True))
                    self.assertFalse(sb.lock.locked())

    def test_premature_finish_recovers_and_real_evidence_is_required(self):
        responses = [fixture.response(fixture.goal('quality', [], 'never', reliable=False)), fixture.PLAN,
                     fixture.response({'selected_image_id':'original','evidence_ids':['quality:original'],'citation_ids':[]}),
                     fixture.actions(('assess_image_quality', {'image_id':'original'})),
                     fixture.response({'selected_image_id':'original','evidence_ids':['quality:original'],'citation_ids':[]})]
        a = AdaptiveRuntime(WireQueue(copy.deepcopy(responses))).run(fixture.session(), '只检查曝光')
        b = LangGraphRuntime(WireQueue(copy.deepcopy(responses))).run(fixture.session(), '只检查曝光')
        self.assertEqual(core(a), core(b))
        self.assertEqual(b['task_status'], 'completed')
        self.assertEqual(b['format_repairs'], 1)
        self.assertIn('repair', [x['node'] for x in b['graph_trace']])

    def test_fabrication_and_continuous_format_errors_stop_within_budget(self):
        for final in [fixture.response({'selected_image_id':'original','evidence_ids':['made_up'],'citation_ids':[]}),
                      {'message':{'content':'not json'}}]:
            responses=[fixture.response(fixture.goal('quality', [], 'never', reliable=False)),fixture.PLAN,final,final,final]
            a=AdaptiveRuntime(WireQueue(copy.deepcopy(responses))).run(fixture.session(),'曝光')
            b=LangGraphRuntime(WireQueue(copy.deepcopy(responses))).run(fixture.session(),'曝光')
            self.assertEqual(core(a),core(b))
            self.assertEqual(b['format_repairs'],2)
            self.assertEqual(b['tool_calls'],0)
            self.assertEqual(b['task_status'],'incomplete')

    def test_limits_and_api_error_attempt_counts(self):
        for opts in [{'max_model_calls':2},{'max_tool_calls':1}]:
            a=AdaptiveRuntime(AdaptiveScriptedClient(),**opts).run(fixture.session(),'只检查曝光')
            b=LangGraphRuntime(AdaptiveScriptedClient(),**opts).run(fixture.session(),'只检查曝光')
            self.assertEqual(core(a),core(b))
        class Failed(AdaptiveScriptedClient):
            def complete(self,*args,**kwargs):
                raise LLMError('network_error','offline simulated failure',http_trace=[{'http_status':None},{'http_status':503}])
        a=AdaptiveRuntime(Failed()).run(fixture.session(),'曝光')
        b=LangGraphRuntime(Failed()).run(fixture.session(),'曝光')
        self.assertEqual(core(a),core(b))
        self.assertEqual(b['request_attempts'],2)

    def test_graph_step_limit_and_tool_crash_preserve_partial_results_and_unlock(self):
        s=fixture.session()
        r=LangGraphRuntime(AdaptiveScriptedClient(),max_graph_steps=5).run(s,'只检查曝光')
        self.assertEqual(r['execution_status'],'graph_step_limit')
        self.assertFalse(s.lock.locked())
        self.assertTrue(r['evidence'])
        with patch('optical_agent.langgraph_runtime.execute',side_effect=RuntimeError('sk-do-not-log-me')):
            s=fixture.session()
            r=LangGraphRuntime(AdaptiveScriptedClient()).run(s,'只检查曝光')
        self.assertEqual(r['execution_status'],'graph_error')
        self.assertNotIn('sk-do-not-log-me',json.dumps(r))
        self.assertFalse(s.lock.locked())

    def test_memory_finish_not_repeated_after_exception(self):
        with tempfile.TemporaryDirectory() as folder:
            store=MemoryStore(Path(folder)/'m.sqlite3')
            with patch.object(MemoryTurn,'finish',side_effect=RuntimeError('simulated persistence boundary')) as finish:
                s=fixture.session()
                r=LangGraphRuntime(AdaptiveScriptedClient(),memory_store=store).run(s,'只检查曝光')
            self.assertEqual(finish.call_count,1)
            self.assertEqual(r['task_status'],'incomplete')
            self.assertFalse(s.lock.locked())

    def test_no_cloud_tracing_even_when_ambient_tracing_enabled(self):
        import os
        with patch.dict(os.environ,{'LANGSMITH_TRACING':'true','LANGCHAIN_TRACING_V2':'true'}), \
             patch('requests.sessions.Session.request',side_effect=AssertionError('network forbidden')):
            r=LangGraphRuntime(AdaptiveScriptedClient()).run(fixture.session(),'只检查曝光')
        self.assertEqual(r['task_status'],'completed')

    def test_web_selection_fixed_session_and_legacy_rejected(self):
        import app
        from agent import OpticalAgent
        with patch.object(app,'_agent',OpticalAgent(fixture.Detector())),patch.object(app,'_memory_store',None):
            client=app.app.test_client()
            image=cv2.imencode('.png',np.full((64,64,3),100,np.uint8))[1].tobytes()
            sid=client.post('/api/sessions',data={'image':(io.BytesIO(image),'fixture.png')}).json['session_id']
            payload={'session_id':sid,'message':'只检查曝光','mode':'scripted','agent_mode':'adaptive','runtime_engine':'langgraph'}
            with patch.object(LangGraphRuntime,'__init__',autospec=True) as init:
                # Check the public constructor path without writing production logs.
                init.side_effect=lambda self,client,**kw: AdaptiveRuntime.__init__(self,client,memory_store=None)
                r=client.post('/api/chat',json=payload)
            self.assertEqual(r.status_code,200,r.json)
            self.assertEqual(r.json['runtime_engine'],'langgraph')
            self.assertTrue(r.json['graph_trace'])
            self.assertEqual(client.post('/api/chat',json=dict(payload,runtime_engine='native')).status_code,400)
            self.assertEqual(client.post('/api/chat',json=dict(payload,agent_mode='legacy')).status_code,400)


class OptionalDependencyTests(unittest.TestCase):
    def test_native_default_and_graph_import_remain_optional(self):
        # Fresh process imports of the app are checked by the base-environment run.
        import app
        from agent import OpticalAgent
        with patch.object(app,'_agent',OpticalAgent(fixture.Detector())),patch.object(app,'_memory_store',None):
            client=app.app.test_client()
            image=cv2.imencode('.png',np.full((64,64,3),100,np.uint8))[1].tobytes()
            sid=client.post('/api/sessions',data={'image':(io.BytesIO(image),'fixture.png')}).json['session_id']
            result=client.post('/api/chat',json={'session_id':sid,'message':'只检查曝光','mode':'scripted','agent_mode':'adaptive'})
            self.assertEqual(result.status_code,200)
            self.assertEqual(result.json['runtime_engine'],'native')
            self.assertEqual(result.json['graph_trace'],[])
