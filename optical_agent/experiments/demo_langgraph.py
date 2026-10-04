"""Offline native / real StateGraph comparison. No CloudClient is constructed."""
from optical_agent.paths import PROJECT_ROOT
import argparse
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
from optical_agent import __version__
from optical_agent.adaptive_runtime import AdaptiveRuntime
from optical_agent.adaptive_fixture import AdaptiveScriptedClient
from optical_agent.langgraph_runtime import LangGraphRuntime
from optical_agent.runtime_compare import core
from optical_agent.state import Session
from optical_agent.tools import ToolContext

TASK = '检查这张水下图，必要时校正曝光，然后只识别管道，说明结果是否可靠。'


class FixtureDetector:
    dataset, threshold, calibration = 'sodd', .33, True
    def predict(self, image):
        return [{'class_name':'pipe','class_id':6,'score':.9,'estimated_box_precision':.9,'box':[12,12,28,28]}]


class QueueClient:
    provider, model_name = 'scripted', 'offline_recovery_wire_fixture'
    def __init__(self):
        def reply(data):
            return {'message':{'content':json.dumps(data,ensure_ascii=False)},'http_attempts':0}
        goal={'task_type':'quality','image_reference':'original','target_classes':[], 'correction_policy':'never','require_reliability':False,'question':''}
        final={'selected_image_id':'original','evidence_ids':['quality:original'],'citation_ids':[]}
        self.responses=iter([reply(goal),reply({'steps':[{'tool':'assess_image_quality','purpose':'检查曝光'}]}),
            reply(final),{'message':{'tool_calls':[{'id':'q1','type':'function','function':{'name':'assess_image_quality','arguments':'{"image_id":"original"}'}}]},'http_attempts':0},reply(final)])
    def complete(self,*args,**kwargs):
        return next(self.responses)


def run_pair(name, image, detector, task=TASK, client_factory=AdaptiveScriptedClient, provenance=None):
    outputs={}
    for engine,runner in [('native',AdaptiveRuntime),('langgraph',LangGraphRuntime)]:
        session=Session(ToolContext(detector,image.copy()))
        original=image.copy()
        result=runner(client_factory()).run(session,task)
        if not np.array_equal(session.context.images['original'],original):
            raise AssertionError('Original image changed')
        outputs[engine]=result
    matching=core(outputs['native'])==core(outputs['langgraph'])
    return {'case':name,'input':provenance or {'kind':'synthetic_fixture'},'task':task,'core_equal':matching,
            'outputs':outputs,'original_preserved':True,
            'actual_http_requests':sum(r['request_attempts'] for r in outputs.values())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real-images',action='store_true',help='Use existing SODD detector and indices 6–8')
    p.add_argument('--output',type=Path,default=Path(r'D:\CodexData\optical_agent\langgraph\runs\equivalence_v052.json'))
    args=p.parse_args()
    cases=[]
    for name,value in [('normal_fixture',100),('dark_fixture',45),('lost_fixture',0)]:
        cases.append(run_pair(name,np.full((64,64,3),value,np.uint8),FixtureDetector()))
    cases.append(run_pair('validation_recovery',np.full((64,64,3),100,np.uint8),FixtureDetector(),'只检查曝光',QueueClient))
    if args.real_images:
        from detector import TorchDetector
        from data import SODDDataset
        detector=TorchDetector(PROJECT_ROOT/'models'/'sodd_detector.pt')
        dataset=SODDDataset(Path(r'D:\CodexData\optical_agent\sodd\SODD\data'),'test')
        for variant,index in [('normal',6),('dark',7),('lost',8)]:
            tensor,_,source=dataset[index]
            image=np.uint8(np.round(tensor.permute(1,2,0).numpy()*255))
            original_hash=hashlib.sha256(image.tobytes()).hexdigest()
            if variant=='dark':
                image=np.uint8(image.astype(float)*.6)
            elif variant=='lost':
                image=np.zeros_like(image)
            cases.append(run_pair('sodd_'+variant,image,detector,provenance={
                'kind':'real_sodd' if variant=='normal' else 'simulated_exposure_perturbation',
                'source_index':index,'source_pixels_sha256':original_hash,'variant':variant,
                'source_file':Path(source).name,
                'model_sha256':detector.model_sha256}))
    base=PROJECT_ROOT
    record={'app_version':__version__,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'langgraph_version':version('langgraph'),'native_runtime_sha256':hashlib.sha256((base/'optical_agent'/'adaptive_runtime.py').read_bytes()).hexdigest(),
            'graph_runtime_sha256':hashlib.sha256((base/'optical_agent'/'langgraph_runtime.py').read_bytes()).hexdigest(),
            'all_core_equal':all(c['core_equal'] for c in cases),'case_count':len(cases),
            'actual_http_requests':sum(c['actual_http_requests'] for c in cases),'cases':cases,
            'scope':'Offline deterministic clients. Not evidence of live LLM quality or improved detection accuracy.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:record[k] for k in ('app_version','case_count','all_core_equal','actual_http_requests')},ensure_ascii=False))
    if not record['all_core_equal']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
