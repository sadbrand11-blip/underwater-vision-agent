"""Matched COCO initialization, grouped selection and separate calibration; test is inaccessible."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.isotonic import IsotonicRegression

from data import collate
from detector import build_model
from metrics import match_counts, average_precision, iou
from quality import UNDERWATER_QUALITY_CONFIG
from optical_agent.vision_data import ROOT, SEED, SODD_CLASSES, ROBOT_CLASSES, d_path, digest, load_manifest, subset, read_rgb, save_json
from train_detector import evaluate_model


class VisionDataset(Dataset):
    def __init__(self, rows, classes, augment=False, strategy='control', exposure='none'):
        self.rows, self.classes = rows, classes
        self.augment, self.strategy, self.exposure = augment, strategy, exposure

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = read_rgb(row['path'])
        boxes = np.asarray([x['box'] for x in row.get('boxes', [])], np.float32).reshape(-1,4)
        labels = torch.tensor([x['class_id'] for x in row.get('boxes', [])], dtype=torch.int64)
        if self.augment and random.random() < .5:
            image = np.ascontiguousarray(image[:, ::-1])
            if len(boxes):
                boxes[:, [0,2]] = image.shape[1] - boxes[:, [2,0]]
        if self.exposure == 'under':
            image = np.uint8(image.astype(float)*.6)
        elif self.augment:
            # The control matches the existing scalar-gain exposure augmentation.
            if self.strategy == 'control' and random.random() < .55:
                gain = random.uniform(.3,.9) if random.random() < .8 else random.uniform(1.1,1.7)
                image = np.uint8(np.clip(np.round(image.astype(np.float32)*gain),0,255))
            elif self.strategy == 'improved' and random.random() < .65:
                f = image.astype(np.float32)/255
                gamma, gain = random.uniform(.75,1.5), random.uniform(.65,1.3)
                h,w = image.shape[:2]
                # Modest spatial illumination variation; no test-time perturbations.
                spatial = np.linspace(random.uniform(.7,1),1,w)[None,:,None]
                f = (f**gamma)*gain*spatial
                f += np.random.normal(0,random.uniform(0,.006),f.shape)
                image = np.uint8(np.round(np.clip(f,0,1)*255))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        return torch.from_numpy(np.ascontiguousarray(image)).permute(2,0,1).float()/255, {
            'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([index]),
            'area': (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]),
            'iscrowd': torch.zeros(len(boxes),dtype=torch.int64)}, row['id']


def fixed_report(pred, truth, classes, threshold):
    counts = match_counts(pred, truth, threshold)
    by_class, aps = {}, []
    for c in range(1,len(classes)):
        p = [[x for x in row if x['class_id']==c] for row in pred]
        t = [[x for x in row if x['class_id']==c] for row in truth]
        m = match_counts(p,t,threshold)
        m['instances'] = sum(map(len,t))
        if not m['instances']:
            m['recall'] = None
        ap = average_precision(pred,truth,c,.5)
        m['ap50'] = ap
        if ap is not None:
            aps.append(ap)
        by_class[classes[c]] = m
    recalls = [m['recall'] for m in by_class.values() if m['recall'] is not None]
    return {**counts, 'images': len(truth), 'threshold': threshold, 'by_class': by_class,
            'macro_recall': float(np.mean(recalls)) if recalls else None,
            'map50': float(np.mean(aps)) if aps else None}


def choose_threshold(pred, truth, classes, max_fp=1.0):
    # Selection split only. Do not call this on held-out test predictions.
    scores = []
    for threshold in np.linspace(.01,.99,99):
        counts = match_counts(pred,truth,float(threshold))
        if counts['fp_per_image'] > max_fp:
            continue
        perclass = []
        for c in range(1,len(classes)):
            p = [[x for x in row if x['class_id']==c] for row in pred]
            t = [[x for x in row if x['class_id']==c] for row in truth]
            if sum(map(len,t)):
                perclass.append(match_counts(p,t,float(threshold))['recall'])
        score = float(np.mean(perclass)) if perclass else 0.0
        scores.append((score,-counts['fp_per_image'],float(threshold)))
    if not scores:
        return 1.0,0.0
    score, _, threshold = max(scores)
    return threshold, score


def calibration(pred, truth, rows, classes):
    scores, correct, support = [], [], {classes[c]:sum(x['class_id']==c for r in truth for x in r) for c in range(1,len(classes))}
    for candidates, targets in zip(pred,truth):
        used = set()
        for candidate in sorted(candidates,key=lambda x:-x['score']):
            choices = [(iou(candidate['box'],x['box']),j) for j,x in enumerate(targets)
                       if j not in used and candidate['class_id']==x['class_id']]
            hit = bool(choices and max(choices)[0]>=.5)
            if hit:
                used.add(max(choices)[1])
            scores.append(candidate['score']); correct.append(int(hit))
    enough = len(scores)>=30 and sum(correct)>=5 and len(correct)-sum(correct)>=5 and len({r['group_id'] for r in rows})>=5
    fitted = None
    if enough:
        reg = IsotonicRegression(out_of_bounds='clip').fit(scores,correct)
        fitted = {'x':reg.X_thresholds_.tolist(),'y':reg.y_thresholds_.tolist(), 'method':'isotonic_heldout_calibration'}
    return fitted, {'status':'validation_fitted_small_sample' if enough else 'insufficient',
        'predicted_boxes':len(scores),'matched_boxes':sum(correct),'groups':len({r['group_id'] for r in rows}),
        'instances_by_class':support,
        'class_status':{c:'validation_fitted_small_sample' if enough and n>=5 else 'insufficient' for c,n in support.items()},
        'warning':'Isotonic scores estimate box precision on a small validation sample; not a deployment guarantee.'}


def train_one(manifest_path, branch, strategy, epochs=12, batch_size=2, resume=None):
    torch.set_num_threads(8); cv2.setNumThreads(1)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    manifest = load_manifest(manifest_path)
    dataset = 'sodd' if branch=='facilities' else 'uiis10k'
    classes = SODD_CLASSES if branch=='facilities' else ROBOT_CLASSES
    train = subset(manifest,dataset,'train','training')
    if branch=='robot':
        positives = [r for r in train if r['boxes']]
        negatives = sorted((r for r in train if not r['boxes']),key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())
        train = positives + negatives[:min(3*len(positives),1000)]
    selection = subset(manifest,dataset,'selection','selection')
    cal_rows = subset(manifest,dataset,'calibration','calibration')
    if not train or not selection:
        raise ValueError('Not enough independent train/selection data')
    folder = d_path(ROOT/'vision_v040/models')
    folder.mkdir(parents=True,exist_ok=True)
    name = branch+'_'+strategy
    output = folder/(name+'.pt')
    history_path = folder/(name+'.history.json')
    if history_path.exists() and json.loads(history_path.read_text())['status']=='complete':
        print(name+' already complete; preserving run',flush=True); return
    if output.exists():
        raise RuntimeError('Interrupted model must use a new run name, not silently overwrite its history')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = ROOT/'pretrained/downloads/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth'
    model = build_model(len(classes),None if resume else weights).to(device)
    resumed = None
    if resume:
        resumed=torch.load(resume,map_location='cpu',weights_only=False)
        if resumed['classes']!=classes:
            raise ValueError('Resume class mapping differs')
        model.load_state_dict(resumed['state_dict'])
    inference_rpn_threshold=model.rpn.score_thresh
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=1e-4)
    train_set = VisionDataset(train,classes,True,strategy)
    history, best, stale, numerical_skips = [], -1.0, 0, []
    metadata = {'status':'running','branch':branch,'strategy':strategy,'seed':SEED,'initialization_sha256':digest(weights),
                'training_code_sha256':digest(Path(__file__)),
                'manifest_sha256':digest(manifest_path),'train_images':len(train),'train_positive_images':sum(bool(r['boxes']) for r in train),
                'selection_images':len(selection),'calibration_images':len(cal_rows),'max_epochs':epochs,
                'patience':3,'lr':.0002,'history':history}
    metadata['rpn_score_threshold_policy']={'training':0.0 if branch=='robot' else inference_rpn_threshold,
        'validation_and_inference':inference_rpn_threshold,'reason':'Negative-only robot batches need nonempty RoI proposals'}
    if resumed:
        metadata.update(resume_sha256=digest(resume),resume_source=str(resume),optimizer_state='AdamW reset after numerical failure')
        record={**resumed['validation'],'epoch':0,'source_epoch':resumed['validation']['epoch'],'phase':'resume_baseline'}
        history.append(record); best=record['primary_metric']
        resumed['experiment'].update({k:v for k,v in metadata.items() if k!='history'})
        torch.save(resumed,output)
    for epoch in range(epochs):
        model.train(); losses=[]; start=time.perf_counter()
        model.rpn.score_thresh=0.0 if branch=='robot' else inference_rpn_threshold
        loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,collate_fn=collate,num_workers=0)
        for step,(images,targets,image_ids) in enumerate(loader,1):
            optimizer.zero_grad(set_to_none=True)
            try:
                components = model([im.to(device) for im in images],[{k:v.to(device) for k,v in t.items()} for t in targets])
                loss = sum(components.values())
                if not torch.isfinite(loss):
                    numerical_skips.append({'epoch':epoch+1,'step':step,'images':list(image_ids),'stage':'loss',
                        'components':{k:str(float(v.detach())) for k,v in components.items()}})
                    save_json(folder/(name+'.numerical_events.json'),numerical_skips)
                    if len(numerical_skips)>2:
                        raise RuntimeError('Numerical instability exceeded two skipped batches; abort')
                    print(json.dumps({'model':name,'numerical_batch_skipped':numerical_skips[-1]}),flush=True)
                    continue
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                if len(images)==1:
                    raise
                optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache(); batch_size=1
                loss_value=0.0
                for image,target in zip(images,targets):
                    one_loss=sum(model([image.to(device)],[{k:v.to(device) for k,v in target.items()}]).values())/len(images)
                    one_loss.backward(); loss_value+=float(one_loss.detach())
                loss=torch.tensor(loss_value)
            gradient_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            if not torch.isfinite(gradient_norm):
                numerical_skips.append({'epoch':epoch+1,'step':step,'images':list(image_ids),'stage':'gradient','norm':str(float(gradient_norm))})
                save_json(folder/(name+'.numerical_events.json'),numerical_skips)
                optimizer.zero_grad(set_to_none=True)
                if len(numerical_skips)>2:
                    raise RuntimeError('Numerical instability exceeded two skipped batches; abort')
                print(json.dumps({'model':name,'numerical_batch_skipped':numerical_skips[-1]}),flush=True)
                continue
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if step%50==0:
                print(json.dumps({'model':name,'epoch':epoch+1,'step':step,'total_steps':len(loader),'loss':float(np.mean(losses[-50:]))}),flush=True)
        model.rpn.score_thresh=inference_rpn_threshold
        pred,truth,latency=evaluate_model(model,VisionDataset(selection,classes),device,batch_size)
        threshold,score=choose_threshold(pred,truth,classes)
        report=fixed_report(pred,truth,classes,threshold)
        record={'epoch':epoch+1,'loss':float(np.mean(losses)),'seconds':time.perf_counter()-start,
                'selection':report,'primary_metric':score,'batch_size':batch_size}
        history.append(record)
        print(json.dumps({'model':name,**record}),flush=True)
        if score>best+1e-8:
            best,stale=score,0
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'classes':classes,'dataset':'sodd' if branch=='facilities' else 'uiis10k_robots',
                'threshold':threshold,'validation':record,'quality_config':asdict(UNDERWATER_QUALITY_CONFIG),
                'experiment':{k:v for k,v in metadata.items() if k!='history'},'training_samples':len(train)},output)
        else:
            stale+=1
        save_json(history_path,metadata)
        if stale>=3:
            break
    payload=torch.load(output,map_location='cpu',weights_only=False)
    model.load_state_dict(payload['state_dict']); model.eval()
    latest=load_manifest(manifest_path)
    if digest(manifest_path)!=metadata['manifest_sha256']:
        if (subset(latest,dataset,'train','training')!=subset(manifest,dataset,'train','training')
                or subset(latest,dataset,'selection','selection')!=selection):
            raise ValueError('Training/selection data changed during training')
        metadata['pre_validation_manifest_sha256']=metadata['manifest_sha256']
        metadata['manifest_sha256']=digest(manifest_path)
        cal_rows=subset(latest,dataset,'calibration','calibration')
        metadata['calibration_images']=len(cal_rows)
        payload['experiment'].update({k:v for k,v in metadata.items() if k!='history'})
    pred,truth,_=evaluate_model(model,VisionDataset(cal_rows,classes),device,batch_size)
    fitted,status=calibration(pred,truth,cal_rows,classes)
    payload['calibration'],payload['calibration_status']=fitted,status
    torch.save(payload,output)
    metadata.update(status='complete',checkpoint_sha256=digest(output),calibration=status,best_primary_metric=best,numerical_skips=numerical_skips)
    save_json(history_path,metadata)
    save_json(folder/(name+'.calibration.json'),status)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,default=ROOT/'vision_v040/manifest.json')
    p.add_argument('--branch',choices=['facilities','robot','all'],default='all')
    p.add_argument('--strategy',choices=['control','improved','both'],default='both')
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--resume',type=Path,help='Explicit recovery checkpoint; AdamW state is reset and lineage recorded')
    a=p.parse_args()
    if not 1<=a.epochs<=12:
        p.error('Maximum 12 epochs')
    for branch in ['facilities','robot'] if a.branch=='all' else [a.branch]:
        for strategy in (['control','improved'] if a.strategy=='both' and branch=='facilities' else ['improved'] if branch=='robot' else [a.strategy]):
            train_one(a.manifest,branch,strategy,a.epochs,a.batch_size,a.resume)


if __name__=='__main__':
    main()
