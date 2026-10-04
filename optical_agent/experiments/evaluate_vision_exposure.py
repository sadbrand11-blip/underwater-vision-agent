"""24 bounded profiles on train, selection on validation, frozen held-out testing."""
from optical_agent.paths import PROJECT_ROOT
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import time

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from quality import ExposureProfile, UNDERWATER_QUALITY_CONFIG, assess, correct_exposure
from optical_agent.vision_data import ROOT, read_rgb, digest, load_manifest, subset, save_json

MAX_SIDE = 768


def resized(rgb, max_side=MAX_SIDE):
    h,w=rgb.shape[:2]
    factor=min(1.0,max_side/max(h,w))
    return cv2.resize(rgb,(round(w*factor),round(h*factor)),interpolation=cv2.INTER_AREA) if factor<1 else rgb


def paired_metrics(image, reference):
    if image.shape!=reference.shape:
        raise ValueError('Aligned reference shape mismatch')
    mse=float(np.mean((image.astype(np.float64)-reference.astype(np.float64))**2))
    return {'psnr':float(10*np.log10(255**2/mse)) if mse else None,
            'identical_reference':mse==0,
            'ssim':float(structural_similarity(image,reference,data_range=255,channel_axis=2))}


def profiles():
    return [ExposureProfile(f'grid_{i:02}',median,lower,1.45,blend)
            for i,(median,lower,blend) in enumerate(itertools.product([.35,.40,.45,.50],[.65,.80],[0,.15,.25]))]


def process(row, candidates, folder, quality_after=False):
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/(row['id'].replace(':','_')+'.json')
    signature={'image_sha256':row['file_sha256'],'reference_sha256':row.get('reference_sha256'),
        'profiles':[asdict(p) for p in candidates], 'max_side':MAX_SIDE,
        'quality_config':asdict(UNDERWATER_QUALITY_CONFIG),'quality_after':quality_after,
        'quality_code_sha256':digest(PROJECT_ROOT/'quality.py')}
    if path.exists():
        old=json.loads(path.read_text(encoding='utf8'))
        if old['signature']==signature:
            return old
        raise ValueError('Cached exposure experiment changed; use new cache folder')
    rgb=resized(read_rgb(row['path'])); ref=resized(read_rgb(row['reference'])) if row.get('reference') else None
    before,_=assess(rgb,UNDERWATER_QUALITY_CONFIG)
    results={'original':paired_metrics(rgb,ref) if ref is not None else {}}
    results['original'].update(quality={k:before[k] for k in ['exposure_state','quality_pass','lost_tile_fraction','global']})
    for profile in candidates:
        start=time.perf_counter(); corrected,parameters=correct_exposure(rgb,before,profile)
        metrics=paired_metrics(corrected,ref) if ref is not None else {}
        metrics.update(parameters=parameters,elapsed_ms=(time.perf_counter()-start)*1000)
        if quality_after:
            after,_=assess(corrected,UNDERWATER_QUALITY_CONFIG)
            metrics['quality']={k:after[k] for k in ['exposure_state','quality_pass','lost_tile_fraction','global']}
        results[profile.name]=metrics
    output={'id':row['id'],'group_id':row['group_id'],'split':row['split'],'signature':signature,'results':results}
    save_json(path,output)
    return output


def aggregate(items, name):
    scores=[r['results'][name] for r in items]
    output={'images':len(scores)}
    for field in ['psnr','ssim','elapsed_ms']:
        values=[r[field] for r in scores if r.get(field) is not None]
        if values:
            output[field]=float(np.mean(values))
            output[field+'_p95']=float(np.percentile(values,95))
    qualities=[r['quality'] for r in scores if 'quality' in r]
    if qualities:
        output.update(quality_failures=sum(r['quality_pass'] is False for r in qualities),
            exposure_states={s:sum(r['exposure_state']==s for r in qualities) for s in sorted({r['exposure_state'] for r in qualities})})
        for field in ['dark_fraction','bright_fraction','noise_estimate']:
            output[field]=float(np.mean([q['global'][field] for q in qualities]))
    return output


def run_rows(rows,candidates,folder,after=False):
    cv2.setNumThreads(1)
    outputs=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,result in enumerate(pool.map(lambda row:process(row,candidates,folder,after),rows),1):
            outputs.append(result)
            if i%25==0:
                print(json.dumps({'phase':folder.name,'completed':i,'total':len(rows)}),flush=True)
    return outputs


def search(manifest_path):
    manifest=load_manifest(manifest_path); candidates=profiles(); folder=ROOT/'vision_v040/exposure'
    rows=subset(manifest,'uieb','train','training')
    outputs=run_rows(rows,candidates,folder/'train')
    scores={p.name:aggregate(outputs,p.name) for p in candidates}
    ranked=sorted(candidates,key=lambda p:scores[p.name]['ssim'],reverse=True)
    output={'phase':'train_search_complete','manifest_sha256':digest(manifest_path),'max_side':MAX_SIDE,
        'metric_policy':'RGB SSIM skimage 7x7, data_range=255; PSNR 255; both images resized identically, capped at768px',
        'grid':[asdict(p) for p in candidates],'scores':scores,'original':aggregate(outputs,'original'),
        'top3':[asdict(p) for p in ranked[:3]],'quality_code_sha256':digest(PROJECT_ROOT/'quality.py')}
    save_json(folder/'search.json',output)
    print(json.dumps(output),flush=True)


def select(manifest_path):
    from detector import TorchDetector
    from train_vision_detectors import fixed_report, SODD_CLASSES
    manifest=load_manifest(manifest_path); folder=ROOT/'vision_v040/exposure'
    searched=json.loads((folder/'search.json').read_text())
    if searched['manifest_sha256']!=digest(manifest_path):
        backup=Path(manifest_path).with_name('manifest.before_allclass_validation.json')
        if not backup.exists() or digest(backup)!=searched['manifest_sha256']:
            raise ValueError('Split changed after search')
        previous=json.loads(backup.read_text())
        if (subset(previous,'uieb','train')!=subset(manifest,'uieb','train')
                or subset(previous,'sodd','selection')!=subset(manifest,'sodd','selection')):
            raise ValueError('Exposure training or SODD selection changed after search')
    candidates=[ExposureProfile(),*[ExposureProfile(**p) for p in searched['top3']]]
    val=run_rows(subset(manifest,'uieb','selection','selection'),candidates,folder/'selection',True)
    scores={p.name:aggregate(val,p.name) for p in candidates}; baseline=scores['legacy_v030']
    control=ROOT/'vision_v040/models/facilities_control.pt'
    detector=TorchDetector(control)
    sodds=subset(manifest,'sodd','selection','selection')
    preds={p.name:[] for p in candidates}; truth=[]
    for row in sodds:
        rgb=read_rgb(row['path']); before,_=assess(rgb,UNDERWATER_QUALITY_CONFIG); truth.append(row['boxes'])
        for p in candidates:
            corrected,_=correct_exposure(rgb,before,p)
            preds[p.name].append(detector.predict(corrected,.01))
    detection={p.name:fixed_report(preds[p.name],truth,SODD_CLASSES,detector.threshold) for p in candidates}
    gates={}
    base_det=detection['legacy_v030']
    for p in candidates[1:]:
        metric, det=scores[p.name],detection[p.name]
        gates[p.name]={'ssim':metric['ssim']>baseline['ssim'], 'psnr':metric['psnr']>=baseline['psnr']-.2,
            'saturation':metric['bright_fraction']<=baseline['bright_fraction']+.01,
            'darkness':metric['dark_fraction']<=baseline['dark_fraction']+.01,
            'macro_recall':det['macro_recall']>=base_det['macro_recall'],
            'class_recall':all(m['recall'] is None or base_det['by_class'][c]['recall'] is None
                or m['recall']>=base_det['by_class'][c]['recall']-.05 for c,m in det['by_class'].items()),
            'fp':det['fp_per_image']<=base_det['fp_per_image']+.1}
    allowed=[p for p in candidates[1:] if all(gates[p.name].values())]
    selected=max(allowed,key=lambda p:scores[p.name]['ssim']) if allowed else candidates[0]
    output={'phase':'validation_frozen','manifest_sha256':digest(manifest_path),'max_side':MAX_SIDE,
        'selected':asdict(selected),'validation':scores,'original':aggregate(val,'original'),'gates':gates,
        'sodd_validation':detection,'sodd_control_sha256':digest(control),'quality_code_sha256':digest(PROJECT_ROOT/'quality.py'),
        'promote_exposure':bool(allowed),'test_not_used':True}
    path=folder/'selection.json'
    if path.exists() and json.loads(path.read_text())!=output:
        raise ValueError('Selection already frozen')
    save_json(path,output); print(json.dumps(output),flush=True)


def test(manifest_path):
    from evaluate_vision_detectors import freeze
    freeze()  # Shared complete-model/data/code freeze; never inspect a held-out image first.
    manifest=load_manifest(manifest_path); folder=ROOT/'vision_v040/exposure'
    selected=json.loads((folder/'selection.json').read_text())
    if selected['manifest_sha256']!=digest(manifest_path) or selected['quality_code_sha256']!=digest(PROJECT_ROOT/'quality.py'):
        raise ValueError('Frozen exposure configuration changed')
    candidates=[ExposureProfile(), ExposureProfile(**selected['selected']), ExposureProfile('gamma_reference',.45,.65,1.45,0)]
    candidates=list({p.name:p for p in candidates}.values())
    paired=run_rows(subset(manifest,'uieb','test'),candidates,folder/'test',True)
    challenge=run_rows(subset(manifest,'uieb_challenge','test'),candidates,folder/'challenge',True)
    # MOUD has varying illumination, not aligned exposure ground truth.
    moud_rows=[{'id':'moud:'+p.stem,'path':str(p),'file_sha256':digest(p),'group_id':p.stem.split('_')[0],
               'split':'diagnostic'} for p in sorted((ROOT/'moud/Scene_1/Images').rglob('*')) if p.suffix.lower() in {'.png','.jpg','.jpeg'}]
    moud=run_rows(moud_rows,candidates,folder/'moud',True) if moud_rows else []
    save_json(folder/'test_summary.json',{'manifest_sha256':digest(manifest_path),'max_side':MAX_SIDE,
        'paired':{n:aggregate(paired,n) for n in ['original',*[p.name for p in candidates]]},
        'challenge':{n:aggregate(challenge,n) for n in ['original',*[p.name for p in candidates]]},
        'moud':{n:aggregate(moud,n) for n in ['original',*[p.name for p in candidates]]} if moud else {},
        'moud_policy':'illumination diagnostic only; no unregistered PSNR or detection recall',
        'test_rows':paired, 'challenge_rows':challenge,'moud_rows':moud})


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=['search','select','test'])
    p.add_argument('--manifest',type=Path,default=ROOT/'vision_v040/manifest.json')
    a=p.parse_args(); globals()[a.phase](a.manifest)
