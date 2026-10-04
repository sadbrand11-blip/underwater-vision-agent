"""Reproduce empty RoI negative-batch loss using real labelled negatives, without weight updates."""
import json
import torch
from pathlib import Path
from detector import build_model
from optical_agent.experiments.train_vision_detectors import VisionDataset
from optical_agent.vision_data import ROOT, load_manifest, save_json, digest

def main():
    torch.set_num_threads(8)
    folder=ROOT/'vision_v040'; checkpoint=folder/'models/aborted_empty_roi_robot_improved.pt'
    data=load_manifest(folder/'manifest.json')
    rows=[r for r in data['records'] if r['id'] in ['uiis10k:train_2685.jpg','uiis10k:train_3147.jpg']]
    assert len(rows)==2 and all(not r['boxes'] and r['split']=='train' for r in rows)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    device=torch.device('cuda'); model=build_model(2).to(device); model.load_state_dict(payload['state_dict']); model.train()
    items=[VisionDataset(rows,payload['classes'])[i] for i in range(2)]
    images=[x[0].to(device) for x in items]; targets=[{k:v.to(device) for k,v in x[1].items()} for x in items]
    results={}
    with torch.no_grad():
        for threshold in [.05,1.0,0.0]:
            model.rpn.score_thresh=threshold
            transformed,ts=model.transform(images,targets); features=model.backbone(transformed.tensors)
            proposals,_=model.rpn(transformed,features,ts)
            losses=model(images,targets)
            results[str(threshold)]={'proposals_per_image':[len(p) for p in proposals],
                                    'losses':{k:str(float(v)) for k,v in losses.items()},
                                    'all_finite':all(bool(torch.isfinite(v)) for v in losses.values())}
    report={'images':[r['id'] for r in rows],'checkpoint_sha256':digest(checkpoint),'results':results,
        'scope':'Real negative images, validation-selected earlier checkpoint; threshold1.0 is a controlled empty-proposal stress reproduction, not replay of exact failed training weights.',
        'training_only_fix':'Robot RPN score threshold 0 during training, restore .05 for validation/inference; no optimizer step in diagnostic.'}
    save_json(folder/'numerical_diagnosis.json',report); print(json.dumps(report),flush=True)

if __name__=='__main__':main()
