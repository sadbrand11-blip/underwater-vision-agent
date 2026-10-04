"""Independent facility/robot models with explicit availability and calibration."""
import copy
from dataclasses import asdict
import hashlib
import json
from detector import TorchDetector
from quality import ExposureProfile, UNDERWATER_QUALITY_CONFIG
from optical_agent.vision_data import ROOT, SODD_CLASSES

GLOBAL_CLASSES = [*SODD_CLASSES, 'underwater_robot']

class ModelUnavailableError(RuntimeError):
    reason_code = 'model_unavailable'

class DetectorRouter:
    dataset = 'sodd'
    classes = GLOBAL_CLASSES

    def __init__(self, facilities=None, robot=None, profile=None):
        self.branches = {'facilities': facilities, 'robot': robot}
        self.threshold = min((d.threshold for d in self.branches.values() if d is not None),default=.35)
        self.exposure_profile = profile or ExposureProfile()
        self.quality_config = asdict(UNDERWATER_QUALITY_CONFIG)
        self.calibration = {'branch_status':{k:getattr(d,'calibration_status',None) for k,d in self.branches.items()}}
        self.model_sha256 = hashlib.sha256(json.dumps(self.metadata(),sort_keys=True).encode()).hexdigest()
        self.model_path = 'independent_facility_and_robot_branches'

    def branch_names(self, target_classes=None):
        targets=set(target_classes or self.classes[1:])
        if targets-set(self.classes[1:]):
            raise ValueError('Unsupported target classes')
        return tuple(name for name,supported in [('facilities',set(SODD_CLASSES[1:])),('robot',{'underwater_robot'})] if targets & supported)

    def cache_key(self, target_classes=None):
        return tuple((name,getattr(self.branches[name],'model_sha256','missing')) for name in self.branch_names(target_classes))

    def metadata(self):
        return {'profile':asdict(self.exposure_profile),'branches':{name:{'available':d is not None,
            'model_sha256':getattr(d,'model_sha256',None),'model_path':getattr(d,'model_path',None),
            'calibration':getattr(d,'calibration_status',None),'experimental':True} for name,d in self.branches.items()}}

    def predict(self, rgb, threshold=None, target_classes=None):
        names=self.branch_names(target_classes)
        missing=[name for name in names if self.branches[name] is None]
        if missing:
            raise ModelUnavailableError('检测模型不可用：'+','.join(missing)+'；不能解释为没有目标')
        outputs=[]
        for name in names:
            detector=self.branches[name]
            for box in detector.predict(rgb,threshold):
                box=copy.deepcopy(box)
                box['class_id']=GLOBAL_CLASSES.index(box['class_name'])
                box['model_branch']=name
                box['model_sha256']=getattr(detector,'model_sha256','fixture')
                box['operating_threshold']=detector.threshold
                status=getattr(detector,'calibration_status',None)
                class_status=(status or {}).get('class_status',{}).get(box['class_name'],'insufficient')
                box['calibration_status']=class_status
                if class_status=='insufficient' or not getattr(detector,'calibration',None):
                    box['estimated_box_precision']=0.0
                outputs.append(box)
        return outputs

def load_experimental(mode='experiment_candidate'):
    if mode not in {'experiment_control','experiment_candidate'}:
        raise ValueError('Unknown experimental vision mode')
    facility=ROOT/'vision_v040/models'/('facilities_control.pt' if mode=='experiment_control' else 'facilities_improved.pt')
    robot=ROOT/'vision_v040/models/robot_improved.pt'
    profile=ExposureProfile()
    selected=ROOT/'vision_v040/exposure/selection.json'
    if mode=='experiment_candidate' and selected.exists():
        profile=ExposureProfile(**json.loads(selected.read_text())['selected'])
    return DetectorRouter(TorchDetector(facility) if facility.exists() else None,TorchDetector(robot) if robot.exists() else None,profile)
