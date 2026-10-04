import copy
from dataclasses import asdict
import json
from types import SimpleNamespace
import numpy as np
import pytest

from optical_agent.experiments.download_vision_data import validate_members
from quality import ExposureProfile, UNDERWATER_QUALITY_CONFIG, assess, correct_exposure
from agent import OpticalAgent, _high_confidence
from optical_agent.vision_data import assign_groups, audit_manifest, subset, validate_boxes, SODD_CLASSES
from optical_agent.vision_router import DetectorRouter, ModelUnavailableError, GLOBAL_CLASSES
from optical_agent.tools import ToolContext
from optical_agent.adaptive_tools import AdaptiveState, execute
from optical_agent.adaptive_goals import Goal, parse_goal
from optical_agent.experiments.train_vision_detectors import fixed_report, calibration
from optical_agent.experiments.evaluate_vision_detectors import new_fp, detector_gate, verify_dataset_sources


def row(name,dataset='uieb',pixel=None,phash=0,split=None,group=None):
    r={'id':name,'dataset':dataset,'pixel_sha256':pixel or name,'phash':phash,'width':64,'height':64,'boxes':[]}
    if split:r['split']=split
    if group:r['group_id']=group
    return r


@pytest.mark.parametrize('member',['../x','/x','C:/x','x/../../y','x\\..\\y'])
def test_archive_cannot_escape(member):
    with pytest.raises(ValueError):validate_members([member])


def test_reference_and_global_duplicates_share_split():
    records=[row('a'),row('b','uiis10k',phash=0),row('c',phash=(1<<64)-1)]
    records[0]['reference_fingerprints']=[{'pixel_sha256':'reference','phash':3}]
    assign_groups(records)
    assert records[0]['group_id']==records[1]['group_id']
    assert records[0]['split']==records[1]['split']
    assert audit_manifest({'records':records})


def test_challenge_near_duplicate_forces_whole_group_test():
    records=[row('a'),row('c','uieb_challenge',phash=1)]
    assign_groups(records)
    assert {r['split'] for r in records}=={'test'}


def test_forged_group_cannot_hide_duplicate_leak():
    records=[row('a',pixel='same',split='train',group='a'),row('b',pixel='same',split='test',group='b')]
    with pytest.raises(ValueError,match='Duplicate pixels'):audit_manifest({'records':records})


def test_pair_dimensions_and_test_selection_forbidden():
    r=row('a',split='train',group='a'); r.update(reference='ref',reference_width=63,reference_height=64)
    with pytest.raises(ValueError,match='dimensions'):audit_manifest({'records':[r]})
    with pytest.raises(ValueError,match='Held-out'):subset({'records':[]},'uieb','test','selection')


@pytest.mark.parametrize('box',[[1,1,0,2],[0,0,65,64],[0,0,float('nan'),1]])
def test_invalid_boxes_rejected(box):
    with pytest.raises(ValueError):validate_boxes([{'box':box,'class_id':1}],64,64,2)


class Branch:
    threshold=.4
    model_sha256='version1'
    model_path='fixture'
    calibration=None
    calibration_status={'status':'insufficient','class_status':{'underwater_robot':'insufficient','pipe':'insufficient'}}
    def __init__(self,robot=False):self.robot,self.calls=robot,0
    def predict(self,image,threshold=None):
        self.calls+=1
        return [{'box':[8,8,40,40],'score':.9,'class_id':1 if self.robot else 6,
                 'class_name':'underwater_robot' if self.robot else 'pipe'}]


def test_branch_routing_stable_class_mapping_and_insufficient_calibration():
    facilities,robot=Branch(),Branch(True); router=DetectorRouter(facilities,robot)
    image=np.full((64,64,3),100,np.uint8)
    p=router.predict(image,target_classes=['pipe'])
    assert facilities.calls==1 and robot.calls==0 and p[0]['class_id']==6
    p=router.predict(image,target_classes=['underwater_robot'])
    assert robot.calls==1 and p[0]['class_id']==7
    assert not _high_confidence(p,.35)


def test_missing_model_is_error_not_empty_detections():
    router=DetectorRouter(Branch(),None)
    with pytest.raises(ModelUnavailableError):router.predict(np.zeros((64,64,3),np.uint8),target_classes=['underwater_robot'])
    assert router.predict(np.zeros((64,64,3),np.uint8),target_classes=['pipe'])


def test_adaptive_branch_cache_and_completion_preconditions():
    facilities,robot=Branch(),Branch(True); router=DetectorRouter(facilities,robot)
    state=AdaptiveState(ToolContext(router,np.full((64,64,3),100,np.uint8)))
    state.begin(Goal('detection','original',('pipe',),'never',False))
    assert execute(state,'detect_objects',{'image_id':'original'})['ok']
    assert execute(state,'detect_objects',{'image_id':'original'})['ok']
    assert facilities.calls==1 and robot.calls==0
    state.begin(Goal('analysis','original',('underwater_robot',),'never',True))
    execute(state,'assess_image_quality',{'image_id':'original'})
    assert not execute(state,'assess_reliability',{'image_id':'original'})['ok']
    assert execute(state,'detect_objects',{'image_id':'original'})['data']['data']['detections'][0]['class_id']==7
    assert execute(state,'assess_reliability',{'image_id':'original'})['ok']
    assert facilities.calls==1 and robot.calls==1
    assert len(state.detection_cache)==2


def test_goal_robot_alias_only_available_on_extended_session():
    obj={'task_type':'detection','image_reference':'original','target_classes':['ROV'],
         'correction_policy':'never','require_reliability':False,'question':''}
    state=SimpleNamespace(detector=SimpleNamespace(classes=GLOBAL_CLASSES),images={'original':None},last_candidate=None)
    assert parse_goal(json.dumps(obj),state,'original').target_classes==('underwater_robot',)
    state.detector.classes=SODD_CLASSES
    assert parse_goal(json.dumps(obj),state,'original').disposition=='unsupported'


@pytest.mark.parametrize('change',['version','replace','threshold','calibration'])
def test_branch_mutation_cannot_mix_old_and_new_cache(change):
    router=DetectorRouter(Branch(),Branch(True))
    state=AdaptiveState(ToolContext(router,np.full((64,64,3),100,np.uint8)))
    state.begin(Goal('detection','original',('pipe',),'never',False))
    assert execute(state,'detect_objects',{'image_id':'original'})['ok']
    if change=='replace':
        router.branches['facilities']=Branch()
    elif change=='version':
        router.branches['facilities'].model_sha256='version2'
    elif change=='threshold':
        router.branches['facilities'].threshold=.7
    else:
        router.branches['facilities'].calibration={'new':'calibration'}
    result=execute(state,'detect_objects',{'image_id':'original'})
    assert result['ok'] is False and '重新创建会话' in result['error']
    assert len(state.detection_cache)==1


def test_offline_joint_targets_are_not_silently_dropped():
    from optical_agent.adaptive_fixture import scripted_goal
    parsed=scripted_goal('识别水下机器人、管道和二维码')
    assert set(parsed['target_classes'])=={'underwater_robot','pipe','pipe_type2','qr_codes'}
    parsed=scripted_goal('识别螺旋桨、管道、红色鳍片、渔网、二维码和auv')
    assert set(parsed['target_classes'])==set(GLOBAL_CLASSES[1:])


def test_delivery_multiframe_consensus_cannot_ignore_valid_conflict():
    from optical_agent.vision_delivery import DeliveryOpticalAgent
    class FrameBranch(Branch):
        calibration={'x':[0,1],'y':[.8,.8]}
        def predict(self,image,threshold=None):
            if image[0,0,0]==110:return []
            boxes=super().predict(image,threshold)
            boxes[0]['estimated_box_precision']=.8
            return boxes
    detector=FrameBranch()
    a=np.full((64,64,3),100,np.uint8);b=np.full((64,64,3),110,np.uint8)
    report,views=DeliveryOpticalAgent(detector).inspect([a,b])
    assert report['status']=='unreliable'
    assert report['multi_frame']['consistent_frames']==0
    assert report['accepted_detections']==[] and report['correction']['accepted'] is False
    assert np.array_equal(views['detections_selected'],views['selected'])
    report,_=DeliveryOpticalAgent(detector).inspect([a,a.copy()])
    assert report['status']=='reliable' and report['multi_frame']['consistent_frames']==2
    report,_=DeliveryOpticalAgent(detector).inspect([a,np.zeros_like(a)])
    assert report['multi_frame']['evidence_scope']=='no_verified_cross_exposure_consensus'


def test_experimental_legacy_combination_stops_before_any_inference(monkeypatch):
    import app as web
    context=ToolContext(DetectorRouter(Branch(),None),np.full((64,64,3),100,np.uint8))
    monkeypatch.setattr(web._sessions,'get',lambda sid:SimpleNamespace(context=context))
    response=web.app.test_client().post('/api/chat',json={'session_id':'fixture','message':'只识别管道',
                                                       'mode':'scripted','agent_mode':'legacy'})
    assert response.status_code==400 and '动态目标与规划' in response.json['error']
    assert context.detector.branches['facilities'].calls==0


@pytest.mark.parametrize('value',[0,255])
def test_blank_images_preserved_and_rejected(value):
    rgb=np.full((64,64,3),value,np.uint8); before=rgb.copy()
    q,_=assess(rgb,UNDERWATER_QUALITY_CONFIG); corrected,parameters=correct_exposure(rgb,q,ExposureProfile('candidate',.4,.8,1.45,.15))
    assert q['quality_pass'] is False and parameters['applied'] is False
    assert np.array_equal(rgb,before) and np.array_equal(rgb,corrected)
    assert OpticalAgent(DetectorRouter(Branch(),Branch(True))).inspect([rgb])[0]['status']=='quality_failure'


def test_profile_change_invalidates_cache_and_candidates_from_original():
    router=DetectorRouter(Branch(),Branch(True)); state=AdaptiveState(ToolContext(router,np.full((64,64,3),30,np.uint8)))
    state.begin(Goal('correction','original',('pipe',),'always',False))
    execute(state,'assess_image_quality',{'image_id':'original'})
    assert execute(state,'generate_exposure_candidate',{'image_id':'original','method':'gamma_only'})['ok']
    assert not execute(state,'generate_exposure_candidate',{'image_id':'candidate_gamma','method':'local_bounded'})['ok']
    state.exposure_profile=ExposureProfile('other',.4,.8,1.45,.15)
    assert not execute(state,'assess_image_quality',{'image_id':'original'})['ok']


def test_empty_groundtruth_is_not_perfect_recall_and_calibration_insufficient():
    result=fixed_report([[]],[[]],['background','underwater_robot'],.4)
    assert result['macro_recall'] is None and result['map50'] is None
    fitted,status=calibration([[]],[[]],[{'group_id':'a'}],['background','underwater_robot'])
    assert fitted is None and status['status']=='insufficient'


def test_new_false_positives_use_one_to_one_matching():
    p={'class_id':1,'box':[0,0,10,10],'score':.9}
    assert new_fp([p],[p,p],[])==1
    assert new_fp([],[],[])==0


def test_macro_gain_cannot_hide_single_class_regression():
    control={'macro_recall':.5,'fp_per_image':.1,'by_class':{'a':{'recall':.8},'b':{'recall':.2}}}
    candidate={'macro_recall':.6,'fp_per_image':.1,'by_class':{'a':{'recall':.7},'b':{'recall':.5}}}
    assert not all(detector_gate(control,candidate).values())


def test_pre_test_freeze_detects_modified_image_or_annotation(monkeypatch):
    monkeypatch.setattr('optical_agent.experiments.evaluate_vision_detectors.digest',lambda p:'changed')
    with pytest.raises(ValueError,match='source changed'):
        verify_dataset_sources({'records':[{'path':'image','file_sha256':'original'}], 'annotation_hashes':{}})
    with pytest.raises(ValueError,match='COCO annotation'):
        verify_dataset_sources({'records':[], 'annotation_hashes':{'annotation':'original'}})


def test_negative_batch_empty_proposals_regression():
    """Actual TorchVision loss: empty negative proposals fail, retaining train proposals is finite."""
    import torch
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign
    backbone=torch.nn.Sequential(torch.nn.Conv2d(3,8,3,padding=1,stride=4),torch.nn.ReLU())
    backbone.out_channels=8
    model=FasterRCNN(backbone,num_classes=2,min_size=32,max_size=32,
        rpn_anchor_generator=AnchorGenerator(((8,16),),((1.0,),)),
        box_roi_pool=MultiScaleRoIAlign(['0'],2,2),rpn_score_thresh=1.0,
        rpn_pre_nms_top_n_train=20,rpn_post_nms_top_n_train=10)
    model.train()
    image=torch.full((3,32,32),.4)
    target={'boxes':torch.empty((0,4)), 'labels':torch.empty((0,),dtype=torch.int64)}
    with torch.no_grad():
        bad=model([image],[target])
        assert not torch.isfinite(bad['loss_classifier'])
        model.rpn.score_thresh=0.0
        good=model([image],[target])
        assert all(torch.isfinite(loss) for loss in good.values())
