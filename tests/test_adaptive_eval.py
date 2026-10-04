import copy
from dataclasses import replace
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from optical_agent.adaptive_eval import fixed_workflow, score, summarize
from optical_agent.adaptive_fixture import AdaptiveScriptedClient
from optical_agent.adaptive_goals import Goal, parse_goal, validate_plan
from optical_agent.adaptive_runtime import AdaptiveRuntime
from optical_agent.adaptive_tools import AdaptiveState, execute, pixel_hash, validate_finish
from optical_agent.llm import CloudClient, LLMError
from optical_agent.request_budget import RequestBudget
from test_adaptive import session, response, goal, PLAN, actions
from test_tasks import WireQueue


QUALITY = {'types':['quality'],'classes':'all','policy':'never','image':'original','status':'completed','evidence':'quality'}


class AdaptiveEvaluationTests(unittest.TestCase):
    def test_network_stops_span_disjoint_invocations_and_publisher_rejects_changes(self):
        from pathlib import Path
        from optical_agent.experiments.evaluate_adaptive import run_history
        from optical_agent.experiments.report_adaptive import verify_pins
        from optical_agent.independent_eval import digest
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'before').mkdir()
            def record(stamp,status):
                return {'started_at':stamp,'turns':[{'adaptive':{'result':{'execution_status':status}}}]}
            first=root/'before/dev_a_0.json'
            first.write_text(json.dumps(record('1','network_error')),encoding='utf8')
            self.assertEqual(run_history(root),(1,None))
            (root/'fix1').mkdir()
            (root/'fix1/demo_b_0.json').write_text(json.dumps(record('2','network_error')),encoding='utf8')
            self.assertEqual(run_history(root),(2,'two_consecutive_connection_failures'))
            pins={'before/dev_a_0.json':digest(first.read_bytes())}
            verify_pins(root,pins,pins)
            first.write_text(json.dumps(record('1','completed')),encoding='utf8')
            with self.assertRaises(ValueError):
                verify_pins(root,pins,pins)
            with self.assertRaises(ValueError):
                verify_pins(root,{},pins)
            second=root/'fix1/demo_b_0.json'
            for fatal in ('authentication_error','insufficient_balance','request_budget_exceeded'):
                interrupted=dict(record('2',fatal),status='started')
                second.write_text(json.dumps(interrupted),encoding='utf8')
                self.assertEqual(run_history(root),(0,fatal))
            second.write_text(json.dumps(record('2','budget_exceeded')),encoding='utf8')
            self.assertEqual(run_history(root),(0,None))

    def test_audit_guards_task_scoped_revision_and_never_original_binding(self):
        s = session(np.full((64,64,3),45,np.uint8))
        AdaptiveRuntime(AdaptiveScriptedClient()).run(s,'只校正曝光')
        state=s.adaptive
        invalid=execute(state,'revise_plan',{'reason':'比较','observation_ids':['quality:original'],
            'steps':[{'tool':'compare_candidates','purpose':'超出纯校正范围'}]})
        self.assertFalse(invalid['ok'])
        self.assertEqual(state.revisions,[])
        state.begin(Goal('analysis','original',('pipe',),'never',True))
        for name,args in [('detect_objects',{'image_id':'original'}),('assess_image_quality',{'image_id':'candidate_gamma'}),
                          ('detect_objects',{'image_id':'candidate_gamma'}),('compare_candidates',{'candidate_image_ids':['candidate_gamma']})]:
            self.assertTrue(execute(state,name,args)['ok'])
        proposed={'selected_image_id':'candidate_gamma','evidence_ids':list(state.observations),'citation_ids':[]}
        self.assertFalse(validate_finish(state,proposed)['passed'])

    def test_run_wide_terminal_stop_applies_to_disjoint_split_before_budget(self):
        import io
        from pathlib import Path
        from optical_agent.experiments import evaluate_adaptive
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'eval').mkdir()
            (root/'eval/adaptive_tasks.json').write_text('{"cases":[]}',encoding='utf8')
            stopped=root/'runs/adaptive_eval/stopped'
            stopped.mkdir(parents=True)
            (stopped/'terminal_stop.json').write_text('{"reason":"authentication_error"}',encoding='utf8')
            with (patch.object(evaluate_adaptive,'PROJECT_ROOT',root),
                 patch('sys.argv',['optical_agent/experiments/evaluate_adaptive.py','--backend','live','--run-id','stopped','--phase','fix1','--split','demo']),
                 patch.object(evaluate_adaptive,'inputs',return_value=({},{})),
                 patch.object(evaluate_adaptive,'freeze',return_value={}),
                 patch.object(evaluate_adaptive,'load_local_env'),
                 patch.object(evaluate_adaptive,'CloudClient') as cloud,
                 patch.object(evaluate_adaptive,'RequestBudget') as budget,
                 patch('sys.stdout',new_callable=io.StringIO) as out):
                cloud.return_value.configured=True
                evaluate_adaptive.main()
                budget.assert_not_called()
                cloud.return_value.complete.assert_not_called()
                self.assertEqual(json.loads(out.getvalue())['http_attempts_this_invocation'],0)

    def test_independent_oracle_rejects_self_validated_wrong_intent_and_wrong_scope(self):
        s = session()
        result = AdaptiveRuntime(AdaptiveScriptedClient()).run(s, '只检查曝光')
        self.assertTrue(result['task_validation']['passed'])
        self.assertTrue(score(QUALITY, result, s.adaptive, original_hash=pixel_hash(s.adaptive.images['original']))['passed'])
        expected = dict(QUALITY, types=['analysis'], evidence='analysis', classes=['pipe','pipe_type2'], policy='if_needed')
        self.assertFalse(score(expected, result, s.adaptive)['passed'])
        detected = AdaptiveRuntime(AdaptiveScriptedClient()).run(s, '只识别管道')
        oracle = dict(QUALITY, types=['detection'], evidence='detections', classes=['qr_codes'])
        self.assertFalse(score(oracle, detected, s.adaptive)['checks']['classes'])

    def test_missing_or_mutated_real_evidence_cannot_pass(self):
        s = session()
        result = AdaptiveRuntime(AdaptiveScriptedClient()).run(s, '只检查曝光')
        missing = copy.deepcopy(result)
        missing['evidence'] = []
        self.assertFalse(score(QUALITY, missing, s.adaptive)['checks']['evidence'])
        mutated = copy.deepcopy(result)
        mutated['evidence'][0]['data']['quality']['exposure_state'] = 'imagined'
        self.assertFalse(score(QUALITY, mutated, s.adaptive)['checks']['facts'])

    def test_candidates_config_and_original_are_protected_at_finalization(self):
        s = session(np.full((64,64,3),45,np.uint8))
        AdaptiveRuntime(AdaptiveScriptedClient()).run(s,'只校正曝光')
        state = s.adaptive
        with self.assertRaises(ValueError):
            state.images['candidate_gamma'][0,0,0] = 255
        state.config = replace(state.config, dark_level=13)
        self.assertFalse(execute(state,'assess_image_quality',{'image_id':'original'})['ok'])
        with self.assertRaises(ValueError):
            validate_finish(state, {'selected_image_id':'original','evidence_ids':['quality:original'],'citation_ids':[]})

    def test_terminal_comparison_malformed_goal_invalid_plan_and_fake_citation(self):
        s = session(np.zeros((64,64,3),np.uint8))
        result = AdaptiveRuntime(AdaptiveScriptedClient()).run(s,'比较校正前后')
        self.assertEqual(result['task_status'],'completed')
        self.assertEqual(result['vision_status'],'quality_failure')
        with self.assertRaises(ValueError):
            parse_goal(json.dumps(dict(goal(),task_type=[])),s.adaptive,'original')
        with self.assertRaises(ValueError):
            validate_plan({'steps':[{'tool':'imaginary','purpose':'执行'}]},['detect_objects'])
        with self.assertRaises(ValueError):
            validate_finish(s.adaptive,{'selected_image_id':'original','evidence_ids':['quality:original'],'citation_ids':['fake']})

    def test_tool_time_and_format_limits_are_bounded(self):
        queue = WireQueue([response(goal()),PLAN,actions(('assess_image_quality',{'image_id':'original'}),('detect_objects',{'image_id':'original'}))])
        limited = AdaptiveRuntime(queue,max_tool_calls=1).run(session(),'检查并识别管道')
        self.assertEqual(limited['tool_calls'],1)
        self.assertEqual(limited['task_status'],'incomplete')
        queue = WireQueue([response({'wrong':1})]*3)
        broken = AdaptiveRuntime(queue).run(session(),'检查管道')
        self.assertEqual(broken['format_repairs'],2)
        self.assertEqual(broken['model_calls'],3)
        ticks = iter([0,181,181,181])
        timeout = AdaptiveRuntime(WireQueue([]),clock=lambda: next(ticks,181)).run(session(),'检查管道')
        self.assertEqual(timeout['model_calls'],0)
        self.assertEqual(timeout['task_status'],'incomplete')

    def test_fixed_workflow_is_same_tools_and_has_zero_llm_cost(self):
        s = session(np.full((64,64,3),45,np.uint8))
        result = fixed_workflow(s,'检查这张图，必要时校正，只识别管道，说明是否可靠')
        expected = dict(QUALITY, types=['analysis'],classes=['pipe','pipe_type2'],policy='if_needed',evidence='analysis',candidate_required=True)
        self.assertTrue(score(expected,result,s.adaptive,original_hash=pixel_hash(s.adaptive.images['original']))['passed'])
        self.assertEqual(result['computation_counts']['correction'],2)
        self.assertEqual(result['model_calls'],0)
        self.assertEqual(result['usage']['total_tokens'],0)

    def test_labels_never_enter_runtime_requests_and_partial_denominator_remains(self):
        class Spy(AdaptiveScriptedClient):
            def complete(self,messages,*args,**kwargs):
                wire = json.dumps(messages,ensure_ascii=False)
                for key in ['"expected"','"candidate_required"','"no_detection"']:
                    assert key not in wire
                return super().complete(messages,*args,**kwargs)
        s = session()
        r = AdaptiveRuntime(Spy()).run(s,'只检查曝光')
        scoring = score(QUALITY,r,s.adaptive)
        summary = summarize([{'status':'recorded','turns':[{'case':'one','repeat':0,'turn':0,'adaptive':{'result':r,'score':scoring}}]}],2,{'used':0})
        self.assertEqual(summary['case_completion_rate'],.5)
        self.assertFalse(summary['formal_acceptance'])

    def test_120_attempt_cap_and_retry_resume_cannot_bypass(self):
        import requests
        with tempfile.TemporaryDirectory() as folder:
            from pathlib import Path
            budget = RequestBudget(Path(folder)/'budget.json',120)
            for _ in range(119):
                budget.reserve()
            cloud = CloudClient(api_key='test-secret',request_budget=budget)
            with patch('optical_agent.llm.requests.post',side_effect=requests.ConnectionError('fixture')),patch('optical_agent.llm.time.sleep'):
                with self.assertRaises(LLMError) as caught:
                    cloud.complete([{'role':'user','content':'test'}])
            self.assertEqual(caught.exception.code,'request_budget_exceeded')
            self.assertEqual(budget.snapshot()['used'],120)
            resumed = RequestBudget(budget.path,120,must_exist=True)
            cloud.request_budget = resumed
            with patch('optical_agent.llm.requests.post') as post:
                with self.assertRaises(LLMError):
                    cloud.complete([{'role':'user','content':'test'}])
                post.assert_not_called()
