"""Pre-test revalidation after quarantining an invalid COCO calibration negative."""
import json
from pathlib import Path
import torch
from optical_agent.vision_data import ROOT, load_manifest, digest, subset, save_json
from optical_agent.experiments.train_vision_detectors import VisionDataset, calibration, evaluate_model, build_model


def main():
    torch.set_num_threads(8)
    folder=ROOT/'vision_v040'; path=folder/'manifest.json'; manifest=load_manifest(path)
    old_path=folder/'manifest.before_allclass_validation.json'
    previous=json.loads(old_path.read_text())
    if (folder/'freeze.json').exists():
        raise ValueError('Calibration cannot change after test freezing')
    changes=[]
    for name in ['facilities_control','facilities_improved','robot_improved']:
        model_path=folder/'models'/(name+'.pt'); history_path=model_path.with_suffix('.history.json')
        history=json.loads(history_path.read_text())
        if history['status']!='complete':
            raise ValueError('Wait for complete training')
        if history['manifest_sha256']==digest(path):
            continue
        if history['manifest_sha256']!=digest(old_path):
            raise ValueError('Unknown training manifest')
        dataset='sodd' if name.startswith('facilities') else 'uiis10k'
        for split in ['train','selection']:
            if subset(previous,dataset,split)!=subset(manifest,dataset,split):
                raise ValueError('Training/selection changed; a fresh training run is required')
        before=digest(model_path); payload=torch.load(model_path,map_location='cpu',weights_only=False)
        # Only the calibration partition may differ. Weights and epoch selection are preserved.
        old_cal,new_cal=subset(previous,dataset,'calibration'),subset(manifest,dataset,'calibration')
        if old_cal!=new_cal:
            removed=[r for r in old_cal if r not in new_cal]
            if any(r['boxes'] for r in removed):
                raise ValueError('Unexpected removal of a calibration positive')
            model=build_model(len(payload['classes'])); model.load_state_dict(payload['state_dict'])
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device).eval()
            pred,truth,_=evaluate_model(model,VisionDataset(new_cal,payload['classes']),device)
            fitted,status=calibration(pred,truth,new_cal,payload['classes'])
            payload['calibration'],payload['calibration_status']=fitted,status
            history['calibration']=status
            save_json(model_path.with_suffix('.calibration.json'),status)
            del model
            if device.type=='cuda':torch.cuda.empty_cache()
        history.update(pre_validation_manifest_sha256=history['manifest_sha256'],manifest_sha256=digest(path),
            calibration_images=len(new_cal),pre_test_correction='One invalid nonrobot calibration negative excluded; identical train/selection rows and weights retained.')
        if old_cal!=new_cal:
            payload['experiment'].update({k:v for k,v in history.items() if k not in {'history','checkpoint_sha256'}})
            torch.save(payload,model_path)
        history['checkpoint_sha256']=digest(model_path); save_json(history_path,history)
        changes.append({'model':name,'before_sha256':before,'after_sha256':digest(model_path),
                        'training_selection_unchanged':True,'old_calibration_images':len(old_cal),'calibration_images':len(new_cal)})
    save_json(folder/'pre_test_data_correction.json',{'status':'completed_before_test','changes':changes,
        'source_manifest_sha256':digest(old_path),'final_manifest_sha256':digest(path)})
    print(json.dumps(changes),flush=True)


if __name__=='__main__':
    main()
