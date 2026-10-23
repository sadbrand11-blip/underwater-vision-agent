"""Independent hand-checked metrics, leakage and calibration guard tests."""
import copy
import pytest

from optical_agent.general_eval import (audit_split, calibrate_class_thresholds,
    group_and_split, iou, match_detections, operating_metrics, validate_archive_names)
from optical_agent.experiments.evaluate_general_vision import diagnostic_subset, perturb


def box(cid=1, score=None, coords=(0, 0, 10, 10), crowd=False):
    value = {'class_id': cid, 'box': list(coords)}
    if score is not None: value['score'] = score
    else: value['iscrowd'] = crowd
    return value


def test_one_to_one_duplicate_predictions_are_false_positives():
    m = match_detections([box(score=.9), box(score=.8)], [box()])
    assert (m['tp'], m['fp'], m['fn'], m['gt']) == (1, 1, 0, 1)
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1/3)


def test_wrong_category_and_crowd_are_handled_separately():
    assert match_detections([box(2, .9)], [box()])['fp'] == 1
    m = match_detections([box(score=.9)], [box(coords=(-10, -10, 20, 20), crowd=True)])
    assert (m['tp'], m['fp'], m['gt']) == (0, 0, 0)
    assert m['labels'][0]['ignored']


def test_metrics_keep_empty_images_in_fp_denominator():
    rows = [{'predictions': [box(score=.9), box(score=.8)], 'ground_truth': [box()]},
            {'predictions': [], 'ground_truth': [box()]},
            {'predictions': [], 'ground_truth': []}]
    report = operating_metrics(rows)
    assert report['images'] == 3
    assert report['micro_recall'] == .5
    assert report['false_positives_per_image'] == pytest.approx(1/3)


def records():
    return [{'id': str(i), 'dataset': 'coco', 'pixel_sha256': str(i),
             'phash': (1 << i)-1, 'scene_id': None} for i in range(15)]


def test_scene_reference_exact_and_phash_are_grouped_before_split():
    rows = records()
    rows[0]['scene_id'] = rows[1]['scene_id'] = 'same_scene'
    rows[2]['pixel_sha256'] = rows[3]['pixel_sha256']
    rows[4]['reference_fingerprints'] = [{'pixel_sha256': rows[5]['pixel_sha256'], 'phash': rows[5]['phash']}]
    group_and_split(rows, phash_radius=0)
    audit_split(rows)
    for a, b in [(0, 1), (2, 3), (4, 5)]:
        assert rows[a]['group_id'] == rows[b]['group_id']
        assert rows[a]['split'] == rows[b]['split']
    other = records(); group_and_split(other, phash_radius=0)
    other2 = records(); group_and_split(other2, phash_radius=0)
    assert other == other2
    broken = copy.deepcopy(rows); broken[1]['split'] = 'bad'
    with pytest.raises(ValueError, match='leakage'): audit_split(broken)


@pytest.mark.parametrize('name', ['../outside', '/absolute', 'D:/file', 'a\\..\\escape', 'a:stream', 'a/./../x'])
def test_archive_traversal_rejected(name):
    with pytest.raises(ValueError): validate_archive_names([name])


def test_no_test_data_in_calibration_and_sparse_classes_insufficient():
    with pytest.raises(ValueError, match='calibration'):
        calibrate_class_thresholds([{'split': 'test'}], [], 'sha', 'ontology')
    result = calibrate_class_thresholds([{'split': 'calibration', 'ground_truth': [box()],
        'predictions': [box(score=.8)]}], [{'id': 1, 'name': 'person'}], 'sha', 'ontology')
    assert result['class_status']['person'] == 'insufficient'
    assert result['class_thresholds']['person'] == .35


def test_calibration_has_disjoint_score_bins_and_sample_counts():
    rows = [{'split': 'calibration', 'ground_truth': [box()], 'predictions': [box(score=.5+i/100)]}
            for i in range(40)]
    result = calibrate_class_thresholds(rows, [{'id': 1, 'name': 'person'}], 'sha', 'ontology', manifest_sha256='f'*64)
    assert result['class_status']['person'] == 'validated'
    assert result['classes']['person']['ground_truth_instances'] == 40
    assert result['classes']['person']['calibration']['y'] == [1., 1.]
    assert result['preprocessing_version'] == 'jpeg_png_8bit_exif_rgb_v1'
    assert result['calibration_scope'] == 'coco_originals'
    assert result['manifest_sha256'] == 'f'*64 and result['split'] == 'calibration'


def test_insufficient_single_calibration_bin_cannot_change_working_threshold():
    # Thirty perfect predictions pass the precision/count gate, but do not
    # supply two score bins. Live runtime and evaluation must both retain .35.
    rows = [{'split': 'calibration', 'ground_truth': [box()], 'predictions': [box(score=.8)]}
            for _ in range(30)]
    result = calibrate_class_thresholds(rows, [{'id': 1, 'name': 'person'}], 'sha', 'ontology')
    assert result['class_status']['person'] == 'insufficient'
    assert result['classes']['person']['threshold'] == .35
    assert result['class_thresholds']['person'] == .35


def test_diagnostic_subset_is_fixed_without_predictions_or_labels():
    rows = [{'id': str(i)} for i in range(100)]
    assert diagnostic_subset(rows, 30) == diagnostic_subset(list(reversed(rows)), 30)
    assert len(diagnostic_subset(rows, 30)) == 30


def test_synthetic_perturbation_preserves_original_and_shape():
    import numpy as np
    rgb = np.full((12, 13, 3), 100, dtype='uint8'); original = rgb.copy()
    assert np.all(perturb(rgb, 'dark_06') == 60)
    assert np.all(perturb(rgb, 'bright_14') == 140)
    mixed = perturb(rgb, 'mixed_06_14')
    assert np.all(mixed[:, 0] == 60) and np.all(mixed[:, -1] == 140)
    assert np.array_equal(rgb, original)


def test_official_coco_eval_perfect_single_box_is_one(tmp_path):
    pytest.importorskip('pycocotools')
    import json
    from optical_agent.experiments.evaluate_general_vision import coco_metrics
    source = tmp_path/'annotations.json'
    source.write_text(json.dumps({'info': {}, 'images': [{'id': 1, 'width': 100, 'height': 100}],
        'categories': [{'id': 1, 'name': 'person', 'supercategory': 'person'}],
        'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 10, 10],
                         'area': 100, 'iscrowd': 0}]}), encoding='utf8')
    metrics = coco_metrics([{'image_id': 1, 'predictions': [box(score=.9)]}], {'source': str(source)})
    assert metrics['AP50'] == pytest.approx(1)
    assert metrics['AP_50_95'] == pytest.approx(1)
    assert metrics['implementation'] == 'pycocotools.COCOeval'


def test_aggregation_bridge_rejects_changed_model_input_or_scoring(tmp_path, monkeypatch):
    from optical_agent import paths
    from optical_agent.experiments import evaluate_general_vision as evaluation
    from optical_agent.general_eval import sha256
    root = tmp_path/'run'; source = tmp_path/'source'
    monkeypatch.setattr(evaluation, 'ROOT', root)
    monkeypatch.setattr(paths, 'PROJECT_ROOT', source)
    functions = {'optical_agent/general_eval.py': ['read_rgb', 'sha256', 'approved_path', 'atomic_json',
        'iou', 'match_detections', 'operating_metrics'],
        'optical_agent/experiments/evaluate_general_vision.py': ['perturb', 'diagnostic_subset', 'process_one',
        'runtime_signature', 'coco_metrics', 'summarize_rows', 'select_policy']}
    previous = {'model_sha256': 'model', 'code': {}}
    for rel in ['quality.py', 'optical_agent/general_vision.py', 'optical_agent/general_catalog.py',
                'optical_agent/image_input.py', *functions]:
        text = '\n'.join(f'def {name}():\n    return 1\n' for name in functions.get(rel, [])) or 'VALUE = 1\n'
        if rel == 'optical_agent/general_eval.py': text += '\ndef calibrate_class_thresholds():\n    return 1\n'
        a = source/rel; b = root/'source_runtime_v2'/rel
        a.parent.mkdir(parents=True, exist_ok=True); b.parent.mkdir(parents=True, exist_ok=True)
        a.write_text(text, encoding='utf8'); b.write_text(text, encoding='utf8')
        previous['code'][rel] = sha256(a)
    current = copy.deepcopy(previous)
    repaired = source/'optical_agent/general_eval.py'
    repaired.write_text(repaired.read_text('utf8').replace('def calibrate_class_thresholds():\n    return 1',
        'def calibrate_class_thresholds():\n    return 2'), encoding='utf8')
    current['code']['optical_agent/general_eval.py'] = sha256(repaired)
    proof = evaluation.aggregation_reuse_proof(previous, current)
    assert proof['new_inference_requests'] == 0 and not proof['raw_prediction_files_retagged']
    bad = copy.deepcopy(current); bad['model_sha256'] = 'different'
    with pytest.raises(ValueError, match='parameters changed'): evaluation.aggregation_reuse_proof(previous, bad)
    bad = copy.deepcopy(current); bad['code']['quality.py'] = 'different'
    with pytest.raises(ValueError, match='visual implementation'): evaluation.aggregation_reuse_proof(previous, bad)
    repaired.write_text(repaired.read_text('utf8').replace('def read_rgb():\n    return 1', 'def read_rgb():\n    return 2'), encoding='utf8')
    with pytest.raises(ValueError, match='Prediction or scoring'): evaluation.aggregation_reuse_proof(previous, current)
