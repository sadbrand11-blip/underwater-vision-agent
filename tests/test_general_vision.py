"""Ordinary-photo inputs, official ontology and conservative new-profile gates."""
from dataclasses import asdict
from io import BytesIO
import json
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from agent import OpticalAgent, compare_evidence
from optical_agent.general_catalog import (COCO_CLASSES, COCO_ID_TO_NAME,
    COCO_NAME_TO_ID, GENERAL_ALIASES, ONTOLOGY_VERSION, PROFILE_ID, supported_classes)
from optical_agent.general_vision import GeneralDetector, MODEL_SHA256, ModelUnavailableError
from optical_agent.image_input import read_image_bytes
from quality import GENERAL_QUALITY_CONFIG, QualityConfig, assess, correct_exposure


def encoded(image, fmt='PNG', **kwargs):
    output = BytesIO()
    image.save(output, format=fmt, **kwargs)
    return output.getvalue()


def picture():
    rng = np.random.default_rng(17)
    return rng.integers(70, 175, (64, 96, 3), dtype=np.uint8)


class GeneralFixture:
    threshold = .35
    profile_id = PROFILE_ID
    dataset = 'coco2017'
    classes = ('background', *COCO_CLASSES)
    ontology_version = ONTOLOGY_VERSION
    quality_config = asdict(GENERAL_QUALITY_CONFIG)
    calibration = None
    calibration_status = {'status': 'unavailable', 'class_status': {}}

    def predict(self, rgb, threshold=None):
        # Intentionally lacks estimated_box_precision: the old default of 1.0
        # must not silently turn this high-scoring general box into accepted GT.
        return [{'box': [4, 8, 30, 36], 'score': .99, 'class_id': 1, 'class_name': 'person'}]


def test_catalog_has_80_classes_and_original_91_slot_ids():
    assert len(COCO_CLASSES) == len(set(COCO_CLASSES)) == 80
    assert COCO_NAME_TO_ID['person'] == 1
    assert COCO_NAME_TO_ID['stop sign'] == 13
    assert COCO_NAME_TO_ID['toothbrush'] == 90
    assert 12 not in COCO_ID_TO_NAME
    assert GENERAL_ALIASES['行人'] == ('person',)
    assert supported_classes(GeneralFixture()) == COCO_CLASSES
    assert 'pipe' not in COCO_CLASSES and 'qr_codes' not in COCO_CLASSES


def test_lazy_construction_and_metadata_do_not_import_torch(tmp_path):
    script = """import sys
from optical_agent.general_vision import GeneralDetector
d=GeneralDetector(checkpoint='missing-coco.pth',calibration_path='missing-calibration.json')
assert d.metadata()['category_count']==80
assert not d.model_available
assert 'torch' not in sys.modules and 'torchvision' not in sys.modules
"""
    checked = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr


def test_missing_model_does_not_prevent_real_quality_but_detection_is_error(tmp_path):
    detector = GeneralDetector(tmp_path / 'absent.pth', tmp_path / 'absent.json')
    quality, _ = assess(picture(), QualityConfig(**detector.quality_config))
    assert quality['quality_pass'] is True
    assert detector._model is None
    with pytest.raises(ModelUnavailableError, match='不能解释为没有目标') as caught:
        detector.predict(picture())
    assert caught.value.reason_code == 'model_unavailable'


def test_corrupt_checkpoint_rejected_before_torch_load(tmp_path):
    checkpoint = tmp_path / 'bad.pth'
    checkpoint.write_bytes(b'not official weights')
    detector = GeneralDetector(checkpoint, tmp_path / 'absent.json')
    with pytest.raises(ModelUnavailableError, match='SHA256'):
        detector.predict(picture())
    assert detector._model is None and not detector.model_available
    assert assess(picture(), QualityConfig(**detector.quality_config))[0]['quality_pass']


@pytest.mark.parametrize('value', [0, 255])
def test_blank_endpoints_stop_and_correction_does_not_claim_restore(value):
    image = np.full((64, 96, 3), value, np.uint8)
    before, _ = assess(image, GENERAL_QUALITY_CONFIG)
    assert before['quality_pass'] is False
    corrected, params = correct_exposure(image, before)
    assert not params['applied'] and np.array_equal(corrected, image)
    report, _ = OpticalAgent(GeneralFixture()).inspect([image])
    assert report['status'] == 'quality_failure'
    assert report['accepted_detections'] == []


@pytest.mark.parametrize('background', [0, 255])
def test_black_background_or_white_wall_with_visible_structure_is_not_global_failure(background):
    image = np.full((64, 96, 3), background, np.uint8)
    image[16:48, 32:64] = picture()[16:48, 32:64]
    quality, _ = assess(image, GENERAL_QUALITY_CONFIG)
    assert quality['quality_pass'] is True
    assert quality['suspected_clipping_tile_fraction'] > .5
    assert '不能区分' in quality['risk_interpretation']
    assert 'not_camera_exposure' in quality['measurement_scope']


def test_general_gate_never_accepts_missing_precision_calibration():
    image = picture()
    original = image.copy()
    report, views = OpticalAgent(GeneralFixture()).inspect([image])
    assert report['status'] == 'unreliable'
    assert report['accepted_detections'] == []
    assert len(report['tentative_detections']) == 1
    assert report['target_region_quality']['original'][0]['class_name'] == 'person'
    assert report['reliability_layers']['detection_stability'] == 'single_image_only'
    assert report['correction']['accuracy_improvement_verified'] is False
    assert np.array_equal(image, original)
    assert views['detections_selected'].shape == image.shape


def test_tiny_visible_structure_on_white_background_is_risk_not_global_failure():
    image = np.full((256, 256, 3), 255, np.uint8)
    image[120:124, 120:124] = 80
    quality, _ = assess(image, GENERAL_QUALITY_CONFIG)
    assert quality['global']['bright_fraction'] > .995
    assert quality['quality_pass'] is True


def test_general_more_boxes_and_scores_do_not_establish_accuracy():
    image = picture()
    before, heat = assess(image, GENERAL_QUALITY_CONFIG)
    raw = GeneralFixture().predict(image)
    extra = {'box': [50, 8, 70, 36], 'score': .999, 'class_id': 18, 'class_name': 'dog'}
    report, _ = compare_evidence(GeneralFixture(), image, image.copy(), before, before,
        {'applied': True, 'method': 'bounded_gamma'}, raw, raw + [extra], heat, heat)
    assert report['selected_image'] == 'original'
    assert report['status'] == 'unreliable'
    assert report['reliability_layers']['detection_stability'] == 'conflicting'
    assert report['correction']['accuracy_improvement_verified'] is False


def calibration_file(tmp_path, **changes):
    value = {'schema_version': 1, 'profile_id': PROFILE_ID, 'model_sha256': MODEL_SHA256,
        'ontology_version': ONTOLOGY_VERSION,
        'preprocessing_version': 'jpeg_png_8bit_exif_rgb_v1', 'calibration_scope': 'coco_originals',
        'classes': {'person': {'status': 'validated', 'threshold': .7,
                              'calibration': {'x': [0., 1.], 'y': [.1, .95]}},
                    'dog': {'status': 'insufficient', 'threshold': .4}}}
    value.update(changes)
    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


def test_calibration_is_profile_model_and_ontology_bound(tmp_path):
    for key, invalid in [('profile_id', 'underwater'), ('model_sha256', 'wrong'),
                         ('ontology_version', 'wrong'), ('schema_version', 0),
                         ('preprocessing_version', 'different_decoder'), ('calibration_scope', 'enhanced_unknown')]:
        path = calibration_file(tmp_path, **{key: invalid})
        detector = GeneralDetector(tmp_path / 'missing.pth', path)
        assert detector.calibration is None
        assert '不匹配' in detector.calibration_status['reason']


def test_class_calibration_insufficient_remains_candidate(tmp_path):
    detector = GeneralDetector(tmp_path / 'missing.pth', calibration_file(tmp_path))
    assert detector.calibration_status['class_status']['person'] == 'validated'
    assert detector.calibration_status['class_status']['dog'] == 'insufficient'
    assert detector.calibration_status['class_status']['toothbrush'] == 'insufficient'
    assert detector._class_calibration['person']['threshold'] == .7


@pytest.mark.parametrize('content', ['[]','{"classes":[]}','null'])
def test_malformed_calibration_does_not_block_quality(tmp_path, content):
    path=tmp_path/'calibration.json';path.write_text(content,encoding='utf8')
    policy=tmp_path/'exposure_policy.json';policy.write_text('[]',encoding='utf8')
    detector=GeneralDetector(tmp_path/'missing.pth',path)
    assert detector.calibration is None
    assert assess(picture(),QualityConfig(**detector.quality_config))[0]['quality_pass']


def test_quality_failure_retracts_even_calibrated_candidate_acceptance():
    detector=GeneralFixture()
    detector.calibration={'fixture_curve':True}
    detector.calibration_status={'status':'available','class_status':{'person':'validated'}}
    detector.predict=lambda image,threshold=None:[{'box':[0,0,32,32],'score':.99,'class_id':1,
        'class_name':'person','calibration_status':'validated','estimated_box_precision':.95,'operating_threshold':.7}]
    report,_=OpticalAgent(detector).inspect([np.zeros((64,96,3),dtype=np.uint8)])
    assert report['status']=='quality_failure'
    assert report['accepted_detections']==[]
    assert len(report['tentative_detections'])==1


def test_invalid_calibration_curves_cannot_validate_a_class(tmp_path):
    path = calibration_file(tmp_path, classes={'person': {'status': 'validated',
        'calibration': {'x': [1., 0.], 'y': [1., 0.]}}})
    detector = GeneralDetector(tmp_path / 'missing.pth', path)
    assert detector.calibration is None
    assert detector.calibration_status['class_status']['person'] == 'insufficient'


def test_validated_class_requires_real_matching_box_calibration():
    detector = GeneralFixture()
    detector.calibration = {'fixture_curve': True}
    detector.calibration_status = {'status': 'available', 'class_status': {'person': 'validated'}}
    box = detector.predict(picture())[0]
    box.update(calibration_status='validated', estimated_box_precision=.9, operating_threshold=.7)
    image = picture()
    quality, heat = assess(image, GENERAL_QUALITY_CONFIG)
    report, _ = compare_evidence(detector, image, image, quality, quality,
        {'applied': False, 'method': 'identity'}, [box], [box], heat, heat)
    assert report['status'] == 'reliable'
    assert report['accepted_detections'] == [box]
    assert 'single_image' in report['evidence_scope']
    box.pop('calibration_status')
    report, _ = compare_evidence(detector, image, image, quality, quality,
        {'applied': False, 'method': 'identity'}, [box], [box], heat, heat)
    assert report['status'] == 'unreliable'


def test_image_decoder_rgb_exif_and_source_unchanged():
    image = Image.fromarray(picture())
    exif = Image.Exif()
    exif[274] = 6
    payload = encoded(image, 'JPEG', exif=exif)
    unchanged = bytes(payload)
    decoded = read_image_bytes(payload)
    assert decoded.shape == (96, 64, 3)
    assert decoded.dtype == np.uint8 and decoded.flags.owndata
    assert payload == unchanged
    rgba = Image.new('RGBA', (4, 5), (255, 0, 0, 0))
    assert np.all(read_image_bytes(encoded(rgba)) == 255)


@pytest.mark.parametrize('payload', [b'', b'bad image', b'\x00' * (32 * 1024 * 1024 + 1)],
                         ids=['empty', 'corrupt', 'oversize'])
def test_invalid_or_oversize_bytes_rejected(payload):
    with pytest.raises(ValueError):
        read_image_bytes(payload)


def test_unsupported_depth_and_format_rejected():
    with pytest.raises(ValueError, match='8位'):
        read_image_bytes(encoded(Image.fromarray(np.zeros((8, 8), np.uint16))))
    with pytest.raises(ValueError, match='8位'):
        read_image_bytes(encoded(Image.new('RGB', (8, 8)), 'BMP'))


def test_oversize_pixel_count_checked_before_decode(monkeypatch):
    import optical_agent.image_input as module
    monkeypatch.setattr(module, 'MAX_PIXELS', 15)
    with pytest.raises(ValueError, match='1600万'):
        module.read_image_bytes(encoded(Image.new('RGB', (4, 4))))


def test_predict_rejects_unsupported_targets_without_loading_model(tmp_path):
    detector = GeneralDetector(tmp_path / 'missing.pth', tmp_path / 'missing.json')
    with pytest.raises(ValueError, match='不支持'):
        detector.predict(picture(), target_classes=['pipe'])
    with pytest.raises(ValueError, match='RGB'):
        detector.predict(np.zeros((8, 8), np.uint8))
    with pytest.raises(ValueError, match='阈值'):
        detector.predict(picture(), threshold=float('nan'))


def test_predict_keeps_official_ids_and_per_class_operating_threshold(tmp_path):
    torch = pytest.importorskip('torch')
    from types import SimpleNamespace
    path = calibration_file(tmp_path, classes={'person': {'status': 'validated',
        'threshold': .1, 'calibration': {'x': [0., 1.], 'y': [.8, .95]}}})
    detector = GeneralDetector(tmp_path / 'unused.pth', path, device='cpu')
    detector._weights = SimpleNamespace(transforms=lambda: lambda image: torch.zeros(3, 64, 96))
    detector._model = lambda tensors: [{'boxes': torch.tensor([[0., 0., 10., 10.]] * 4),
        'scores': torch.tensor([.2, .99, .9, .6]), 'labels': torch.tensor([1, 12, 90, 18])}]
    boxes = detector.predict(picture())
    assert [box['class_name'] for box in boxes] == ['person', 'toothbrush', 'dog']
    assert [box['class_id'] for box in boxes] == [1, 90, 18]
    assert boxes[0]['calibration_status'] == 'validated'
    assert boxes[0]['operating_threshold'] == .1
    assert boxes[1]['calibration_status'] == 'insufficient'
    assert boxes[1]['estimated_box_precision'] == 0
    override = detector.predict(picture(), threshold=.8, target_classes=['person', 'toothbrush'])
    assert [box['class_name'] for box in override] == ['toothbrush']
