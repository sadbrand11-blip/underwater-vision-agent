"""Read/write local test sessions only; explicitly scripted, never a cloud request."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import requests

from optical_agent.vision_data import ROOT, save_json


def main():
    endpoint = 'http://127.0.0.1:7860'
    client = requests.Session()
    client.trust_env = False
    health = client.get(endpoint+'/health', timeout=10).json()
    assert health['app_version'] == '0.4.0'
    image = ROOT/'uiis10k/data/UIIS10K/img/test_0352.jpg'
    cases = [
        ('legacy_exposure', 'legacy', '只检查这张图的曝光', image.read_bytes(), False),
        ('candidate_pipe', 'experiment_candidate', '只识别管道', image.read_bytes(), False),
        ('candidate_robot', 'experiment_candidate', '只识别水下机器人', image.read_bytes(), False),
        ('joint_targets', 'experiment_candidate', '只识别管道和水下机器人', image.read_bytes(), False),
        ('correction_robot', 'experiment_candidate', '必要时校正曝光，然后只识别水下机器人，说明结果是否可靠。', image.read_bytes(), False),
        ('lost_robot', 'experiment_candidate', '检查这张水下图，必要时校正曝光，然后只识别水下机器人，说明结果是否可靠。',
         cv2.imencode('.png',np.zeros((480,640,3),np.uint8))[1].tobytes(), True),
    ]
    run_id=datetime.now(timezone.utc).strftime('v040_delivery_r1_%Y%m%dT%H%M%S%fZ')
    output = ROOT/'vision_v040/web_checks'/run_id
    output.mkdir(parents=True, exist_ok=False)
    summaries=[]
    for name, visual_mode, question, content, simulated in cases:
        response = client.post(endpoint+'/api/sessions',data={'vision_mode':visual_mode},
                               files={'image':('example.png' if simulated else image.name,content)},timeout=60)
        response.raise_for_status()
        session = response.json()
        answer = client.post(endpoint+'/api/chat',json={'session_id':session['session_id'],
                             'message':question,'agent_mode':'adaptive','mode':'scripted'},timeout=180)
        answer.raise_for_status()
        result=answer.json()
        save_json(output/(name+'.json'),{'session':session,'result':result,'simulated':simulated})
        assert result['task_status'] == 'completed', (name,result['task_status'])
        assert result.get('request_attempts',0) == 0, 'Unexpected HTTP LLM request'
        boxes=result.get('detections',[])
        if name=='legacy_exposure':
            assert result['computation_counts'].get('detection',0)==0
        elif name=='candidate_pipe':
            assert set(result['goal_contract']['target_classes'])=={'pipe','pipe_type2'}
            assert all(box['class_name'] in {'pipe','pipe_type2'} and box['model_branch']=='facilities' for box in boxes)
        elif name in {'candidate_robot','correction_robot'}:
            assert result['goal_contract']['target_classes']==['underwater_robot']
            assert all(box['class_name']=='underwater_robot' and box['class_id']==7 and box['model_branch']=='robot' for box in boxes)
            if name=='correction_robot':
                assert result['computation_counts']['correction']==1
                assert result['selected_image_id'] in {'original','candidate_gamma','candidate_local'}
                assert result['computation_counts']['comparison']>=1
        elif name=='joint_targets':
            assert set(result['goal_contract']['target_classes'])=={'pipe','pipe_type2','underwater_robot'}
            assert result['computation_counts']['detection']==2
        else:
            assert result['computation_counts'].get('correction',0)==0
            assert result['vision_status']=='quality_failure',result['vision_status']
        summaries.append({'case':name,'task_status':result['task_status'],'vision_status':result['vision_status'],
                          'target_count':result['target_count'],'actual_compute':result['computation_counts'],
                          'llm_request_attempts':result.get('request_attempts',0),'simulated':simulated,
                          'goal_classes':result['goal_contract']['target_classes']})
    test_path=ROOT/'tests_v040_release.log'
    if not test_path.exists():
        test_path=ROOT/'tests_v040_final_delivery.log'
    if not test_path.exists():
        test_path=ROOT/'tests_v040_post_audit.log'
    if not test_path.exists():
        test_path=ROOT/'tests_v040_delivery.log'
    test_log=test_path.read_text(encoding='utf8')
    test_result=re.search(r'(\d+) passed in ([\d.]+)s',test_log)
    if test_result is None:
        raise ValueError('A completed actual Python test log is required')
    public={'version':health['app_version'],'runtime_revision':health['runtime_revision'],'run_id':run_id,'raw_records':str(output),
            'python_tests':{'passed':int(test_result[1]),'seconds':float(test_result[2]),'source_log':str(test_path)},
            'web_checks':summaries,'mode':'scripted with real local vision models','paid_llm_requests':0}
    save_json(output/'summary.json',public)
    save_json(Path(__file__).parent/'VISION_VERIFICATION.json',public)
    print(json.dumps(public,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
