import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from metrics import match_counts
from optical_agent.benchmark import FixtureDetector, run_benchmark, score_task
from optical_agent.rag import KnowledgeRetriever
from optical_agent.runtime import AgentRuntime


class BenchmarkTests(unittest.TestCase):
    def test_frozen_tasks_and_scripted_fixture_qualification(self):
        root=Path(__file__).resolve().parents[1]
        tasks=json.loads((root/'eval'/'agent_tasks.json').read_text(encoding='utf-8'))
        self.assertEqual(len(tasks),50)
        self.assertEqual(sum(t['split']=='test' for t in tasks),20)
        selected=[t for t in tasks if t['split']=='test' and t['id'].endswith('07')]
        result=run_benchmark(selected,FixtureDetector(),lambda t:(np.full((64,64,3),100,np.uint8),None),
                             KnowledgeRetriever(root),systems=('fine',),repeats=1)
        self.assertEqual(result['backend'],'scripted')
        self.assertFalse(result['systems']['fine']['metrics']['live_targets_met'])
        self.assertEqual(result['systems']['fine']['metrics']['reliability_overrides'],0)
        self.assertEqual(result['systems']['fine']['metrics']['task_completion_rate'],1)
        followup=next(r for r in result['systems']['fine']['tasks'] if r['kind']=='followup')
        self.assertEqual(followup['tool_calls'],7)
        self.assertEqual(followup['final_turn_tool_calls'],1)
        self.assertEqual(followup['model_calls'],11)

    def test_duplicate_predictions_count_as_false_positive(self):
        prediction={'box':[0,0,10,10],'class_id':1,'score':.9}
        result=match_counts([[prediction,prediction]],[[{'box':[0,0,10,10],'class_id':1}]],.33)
        self.assertEqual(result['tp'],1)
        self.assertEqual(result['fp'],1)

    def test_failed_setup_turn_cannot_pass_followup(self):
        task={'id':'followup', 'kind':'followup','split':'test','source_index':0,'image_variant':'none',
              'setup':['failing setup'], 'message':'只检查原图曝光',
              'required_evidence':'quality:original','allowed_tools':['assess_image_quality'], 'requires_citations':False}
        original_run=AgentRuntime.run
        def run(runtime,session,message):
            if message=='failing setup':
                return {'execution_status':'network_error','latency_ms':1}
            return original_run(runtime,session,message)
        with patch.object(AgentRuntime,'run',run):
            result=run_benchmark([task],FixtureDetector(),lambda t:(np.full((64,64,3),100,np.uint8),None),
                                 None,systems=('fine',),repeats=1)
        self.assertEqual(result['systems']['fine']['tasks'][0]['execution_status'],'completed')
        self.assertEqual(result['systems']['fine']['metrics']['task_completion_rate'],0)
