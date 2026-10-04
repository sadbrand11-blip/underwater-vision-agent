"""Freeze first, evaluate held-out data once; never select a test threshold."""
from optical_agent.paths import PROJECT_ROOT
from dataclasses import asdict
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch

from agent import OpticalAgent
from detector import TorchDetector
from metrics import iou
from quality import ExposureProfile
from optical_agent.vision_data import ROOT, digest, read_rgb, load_manifest, subset, save_json
from optical_agent.experiments.train_vision_detectors import fixed_report

BASE = PROJECT_ROOT
FOLDER = ROOT/'vision_v040'


def frozen_inputs():
    files=[FOLDER/'manifest.json',FOLDER/'source_annotation_audit.json',FOLDER/'exposure/selection.json',
        *[FOLDER/'models'/f for f in ['facilities_control.pt','facilities_improved.pt','robot_improved.pt']],
        *[BASE/f for f in ['quality.py','detector.py','agent.py','metrics.py','optical_agent/experiments/train_vision_detectors.py',
                          'optical_agent/experiments/evaluate_vision_exposure.py','optical_agent/experiments/evaluate_vision_detectors.py','optical_agent/vision_data.py','optical_agent/vision_router.py']]]
    return {str(p):digest(p) for p in files}


def verify_dataset_sources(manifest):
    hashes={}
    for row in manifest['records']:
        for path,expected in [(row['path'],row['file_sha256']),
            *([(row['reference'],row['reference_sha256'])] if row.get('reference') else []),
            *([(row['label'],row['label_sha256'])] if row.get('label') else [])]:
            actual=digest(path)
            if actual!=expected:
                raise ValueError('Dataset source changed after preparation: '+path)
            hashes[path]=actual
    for path,expected in manifest['annotation_hashes'].items():
        actual=digest(path)
        if actual!=expected:
            raise ValueError('COCO annotation source changed after preparation: '+path)
        hashes[path]=actual
    return hashes


def freeze():
    for name in ['facilities_control','facilities_improved','robot_improved']:
        history=json.loads((FOLDER/'models'/(name+'.history.json')).read_text())
        if history['status']!='complete':
            raise ValueError('Cannot freeze while training/calibration is incomplete')
    inputs=frozen_inputs(); path=FOLDER/'freeze.json'
    manifest=load_manifest(FOLDER/'manifest.json')
    inputs.update(verify_dataset_sources(manifest))
    if path.exists():
        frozen=json.loads(path.read_text())
        if frozen['sha256']!=inputs:
            raise ValueError('Frozen inputs changed; cannot resume the same test run')
        return frozen
    value={'phase':'frozen_before_heldout_test','sha256':inputs,'seed':20261001,
        'selection_policy':'validation selection only; heldout calibration only; no test tuning',
        'metric_policy':'IoU0.5 one-to-one, COCO101-point AP50; project split, not official benchmark leaderboard'}
    save_json(path,value)
    return value


def matched_fp(pred,truth):
    used=set(); false=[]
    for p in sorted(pred,key=lambda x:-x['score']):
        options=[(iou(p['box'],t['box']),j) for j,t in enumerate(truth) if j not in used and p['class_id']==t['class_id']]
        if options and max(options)[0]>=.5:
            used.add(max(options)[1])
        else:
            false.append(p)
    return false


def new_fp(original,corrected,truth):
    old,new=matched_fp(original,truth),matched_fp(corrected,truth)
    used=set(); count=0
    for p in new:
        options=[(iou(p['box'],q['box']),j) for j,q in enumerate(old) if j not in used and p['class_id']==q['class_id']]
        if options and max(options)[0]>=.5:
            used.add(max(options)[1])
        else:
            count+=1
    return count


def eval_one(name,rows,profile):
    checkpoint=FOLDER/'models'/(name+'.pt'); folder=FOLDER/'detection_test'/name
    detector=TorchDetector(checkpoint); detector.exposure_profile=profile; agent=OpticalAgent(detector)
    records=[]; folder.mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(rows,1):
        path=folder/(row['id'].replace(':','_')+'.json')
        signature={'image_sha256':row['file_sha256'],'model_sha256':detector.model_sha256,'exposure':asdict(profile)}
        if path.exists():
            result=json.loads(path.read_text())
            if result['signature']!=signature:
                raise ValueError('Test result signature changed')
        else:
            rgb=read_rgb(row['path']); start=time.perf_counter()
            original=detector.predict(rgb,.01)
            under=detector.predict(np.uint8(rgb.astype(float)*.6),.01) if name.startswith('facilities') else None
            report,_=agent.inspect([rgb])
            truth=row['boxes']
            raw=report['detections_original']; corrected=report['detections_corrected']
            result={'id':row['id'],'group_id':row['group_id'],'signature':signature,
                'original':original,'simulated_under':under,'truth':truth,'report':report,
                'new_correction_false_positives':new_fp(raw,corrected,truth),
                'seconds':time.perf_counter()-start}
            save_json(path,result)
        records.append(result)
        if i%100==0:
            print(json.dumps({'model':name,'test_completed':i,'total':len(rows)}),flush=True)
    truth=[r['truth'] for r in records]
    raw=fixed_report([r['original'] for r in records],truth,detector.classes,detector.threshold)
    output={'model':name,'original':raw,'model_sha256':detector.model_sha256,
        'corrected':fixed_report([r['report']['detections_corrected'] for r in records],truth,detector.classes,detector.threshold),
        'selected':fixed_report([r['report']['detections_selected'] for r in records],truth,detector.classes,detector.threshold),
        'accepted':fixed_report([r['report']['accepted_detections'] if r['report']['status']=='reliable' else [] for r in records],truth,detector.classes,detector.threshold),
        'quality_failures':sum(r['report']['status']=='quality_failure' for r in records),
        'unreliable_count':sum(r['report']['status']!='reliable' for r in records),
        'unreliable_ratio':sum(r['report']['status']!='reliable' for r in records)/max(1,len(rows)),
        'new_correction_false_positives':sum(r['new_correction_false_positives'] for r in records),
        'latency_mean_ms':float(np.mean([r['report']['latency_ms'] for r in records])),
        'latency_p95_ms':float(np.percentile([r['report']['latency_ms'] for r in records],95)),
        'positive_images':sum(bool(r['truth']) for r in records),'positive_groups':len({r['group_id'] for r in records if r['truth']}),
        'instances':sum(map(len,truth)),'calibration':detector.calibration_status}
    if name.startswith('facilities'):
        output['simulated_under']=fixed_report([r['simulated_under'] for r in records],truth,detector.classes,detector.threshold)
    output['exploratory_only']=name.startswith('robot') and (output['instances']<30 or output['positive_groups']<5)
    save_json(FOLDER/'detection_test'/(name+'.summary.json'),output)
    return output


def detector_gate(control,candidate):
    return {'macro_recall':candidate['macro_recall']>=control['macro_recall'],
        'class_recall':all(v['recall'] is not None and control['by_class'][c]['recall'] is not None
                          and v['recall']>=control['by_class'][c]['recall']-.05 for c,v in candidate['by_class'].items()),
        'fp':candidate['fp_per_image']<=control['fp_per_image']+.1}


def main():
    torch.set_num_threads(8); cv2.setNumThreads(1)
    frozen=freeze(); manifest=load_manifest(FOLDER/'manifest.json')
    chosen=json.loads((FOLDER/'exposure/selection.json').read_text())
    results={}
    for name in ['facilities_control','facilities_improved','robot_improved']:
        profile=ExposureProfile() if name=='facilities_control' else ExposureProfile(**chosen['selected'])
        results[name]=eval_one(name,subset(manifest,'sodd' if name.startswith('facilities') else 'uiis10k','test'),profile)
    histories={n:json.loads((FOLDER/'models'/(n+'.history.json')).read_text()) for n in results}
    validation={n:torch.load(FOLDER/'models'/(n+'.pt'),map_location='cpu',weights_only=False)['validation']['selection'] for n in results}
    val_gate=detector_gate(validation['facilities_control'],validation['facilities_improved'])
    test_gate=detector_gate(results['facilities_control']['original'],results['facilities_improved']['original'])
    output={'status':'complete','freeze_sha256':digest(FOLDER/'freeze.json'),'models':results,'validation':validation,
        'training':{n:{k:v for k,v in h.items() if k!='history'} for n,h in histories.items()},
        'facility_validation_gates':val_gate,'facility_test_gates':test_gate,
        'promote_facilities':all(val_gate.values()) and all(test_gate.values()),
        'default_policy':'Retain historical default unless the preregistered gates and independent audit support deployment; experimental modes are explicit.',
        'run_scope':'Single seed; project grouped split; no paid LLM calls; not a new formal Agent Eval'}
    save_json(FOLDER/'detection_test/summary.json',output)
    print(json.dumps(output),flush=True)


if __name__=='__main__':
    main()
