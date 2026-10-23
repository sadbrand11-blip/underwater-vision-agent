"""Frozen COCO baseline, calibration-only thresholds and held-out diagnostics.

Uses torchvision's unchanged official COCO detector and pycocotools COCOeval.
No training or paid model API is involved. Diagnostic perturbations are synthetic.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from optical_agent.general_eval import (ROOT, approved_path, atomic_json, audit_split,
    calibrate_class_thresholds, operating_metrics, read_rgb, sha256, utc_now)

VARIANTS = ('original', 'dark_06', 'bright_14', 'mixed_06_14')
METHODS = ('original', 'gamma_only', 'local_bounded')


def perturb(rgb, variant):
    import numpy as np
    if variant == 'original': return rgb
    if variant == 'dark_06': scale = .6
    elif variant == 'bright_14': scale = 1.4
    elif variant == 'mixed_06_14':
        scale = np.linspace(.6, 1.4, rgb.shape[1])[None, :, None]
    else: raise ValueError('Unknown fixed perturbation')
    return np.clip(np.rint(rgb.astype('float32') * scale), 0, 255).astype('uint8')


def diagnostic_subset(records, count):
    return sorted(records, key=lambda r: hashlib.sha256(('diagnostic_v1:'+r['id']).encode()).hexdigest())[:count]


def runtime_signature(detector, manifest_path, diagnostic_count):
    from quality import GENERAL_QUALITY_CONFIG
    from optical_agent.paths import PROJECT_ROOT
    return {'version': 'general_v060_eval1', 'manifest_sha256': sha256(manifest_path),
            'model_sha256': detector.model_sha256, 'profile': asdict(GENERAL_QUALITY_CONFIG),
            'exposure_profile': asdict(detector.exposure_profile), 'raw_prediction_threshold': .05,
            'operating_threshold': .35, 'diagnostic_images_per_split': diagnostic_count,
            'variants': list(VARIANTS), 'candidate_methods': list(METHODS),
            'code': {p: sha256(PROJECT_ROOT/p) for p in ('quality.py', 'optical_agent/general_vision.py',
                'optical_agent/general_catalog.py', 'optical_agent/image_input.py',
                'optical_agent/general_eval.py', 'optical_agent/experiments/evaluate_general_vision.py')},
            'selection_rule': 'On development diagnostics: maximize macro recall under FP/image increase <= .05; require >.005 gain, else retain original.'}


def process_one(record, detector, variant, method, signature):
    import numpy as np
    from quality import assess, correct_exposure, GENERAL_QUALITY_CONFIG
    path = ROOT/'predictions'/variant/method/(str(record['image_id'])+'.json')
    if path.exists():
        cached = json.loads(path.read_text('utf8'))
        if cached['signature'] != signature: raise ValueError('Prediction cache version changed; use a new experiment directory')
        return cached
    started = time.perf_counter()
    if sha256(record['path']) != record['file_sha256']: raise ValueError('Source image changed')
    source = perturb(read_rgb(record['path']), variant)
    quality, _ = assess(source, GENERAL_QUALITY_CONFIG)
    if method == 'original':
        image = source; parameters = {'applied': False, 'method': 'original'}
    else:
        image, parameters = correct_exposure(source, quality, profile=detector.exposure_profile, method=method)
    unchanged = np.array_equal(source, image)
    after = quality if unchanged else assess(image, GENERAL_QUALITY_CONFIG)[0]
    # Reuse the original forward pass only when candidate bytes are unchanged.
    original_cache = ROOT/'predictions'/variant/'original'/(str(record['image_id'])+'.json')
    if method != 'original' and unchanged and original_cache.exists():
        original = json.loads(original_cache.read_text('utf8'))
        if original['signature'] != signature: raise ValueError('Original prediction signature changed')
        predictions = original['predictions']; inference_ms = 0.; reused = True
    else:
        inference_start = time.perf_counter()
        predictions = detector.predict(image, threshold=.05)
        inference_ms = (time.perf_counter()-inference_start)*1000; reused = False
    row = {'id': record['id'], 'image_id': record['image_id'], 'split': record['split'],
           'group_id': record['group_id'], 'variant': variant, 'method': method,
           'synthetic': variant != 'original', 'signature': signature,
           'ground_truth': record['ground_truth'], 'predictions': predictions,
           'quality_before': {k: v for k, v in quality.items() if k != 'tiles'},
           'quality_after': {k: v for k, v in after.items() if k != 'tiles'},
           'parameters': parameters, 'inference_ms': inference_ms, 'inference_reused': reused,
           'elapsed_ms': (time.perf_counter()-started)*1000}
    atomic_json(path, row); return row


def predict_split(records, split, detector, signature, diagnostic_count):
    selected = [r for r in records if r['dataset'] == 'coco_val2017' and r['split'] == split]
    originals = []
    for i, record in enumerate(selected):
        originals.append(process_one(record, detector, 'original', 'original', signature))
        if (i+1) % 50 == 0:
            atomic_json(ROOT/'progress.json', {'split': split, 'stage': 'original', 'completed': i+1,
                'planned': len(selected), 'at': utc_now()})
            print(f'{split} original {i+1}/{len(selected)}', flush=True)
    diagnostics = {}
    for variant in VARIANTS:
        for method in METHODS:
            rows = [process_one(r, detector, variant, method, signature) for r in diagnostic_subset(selected, diagnostic_count)]
            diagnostics[variant+':'+method] = rows
    return originals, diagnostics


def coco_metrics(rows, manifest):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import contextlib
    import io
    original = json.loads(Path(manifest['source']).read_text('utf8'))
    image_ids = {r['image_id'] for r in rows}
    ground = COCO()
    ground.dataset = {'info': original.get('info', {}), 'licenses': original.get('licenses', []),
        'images': [i for i in original['images'] if i['id'] in image_ids],
        'annotations': [a for a in original['annotations'] if a['image_id'] in image_ids],
        'categories': original['categories']}
    predictions = []
    for row in rows:
        for p in row['predictions']:
            x0, y0, x1, y1 = p['box']
            predictions.append({'image_id': row['image_id'], 'category_id': p['class_id'],
                                'bbox': [x0, y0, x1-x0, y1-y0], 'score': p['score']})
    console = io.StringIO()
    with contextlib.redirect_stdout(console):
        ground.createIndex()
        if predictions: detection = ground.loadRes(predictions)
        else:
            detection = COCO(); detection.dataset = {**ground.dataset, 'annotations': []}; detection.createIndex()
        evaluation = COCOeval(ground, detection, 'bbox'); evaluation.params.imgIds = sorted(image_ids)
        evaluation.evaluate(); evaluation.accumulate(); evaluation.summarize()
    return {'implementation': 'pycocotools.COCOeval', 'images': len(image_ids),
        'AP_50_95': float(evaluation.stats[0]), 'AP50': float(evaluation.stats[1]),
        'AP75': float(evaluation.stats[2]), 'AP_small': float(evaluation.stats[3]),
        'AP_medium': float(evaluation.stats[4]), 'AP_large': float(evaluation.stats[5]),
        'AR100': float(evaluation.stats[8]), 'console': console.getvalue(),
        'prediction_score_floor': .05, 'max_detections_per_image': 100}


def summarize_rows(rows, thresholds=None):
    import numpy as np
    result = operating_metrics(rows, thresholds)
    result['quality_failure_rate'] = sum(r['quality_after']['quality_pass'] is False for r in rows)/len(rows) if rows else None
    result['latency_ms'] = {'mean': float(np.mean([r['elapsed_ms'] for r in rows])),
                            'p95': float(np.percentile([r['elapsed_ms'] for r in rows], 95))} if rows else None
    result['actual_inferences'] = sum(not r['inference_reused'] for r in rows)
    return result


def select_policy(diagnostics):
    scores = {key: summarize_rows(rows) for key, rows in diagnostics.items()}
    eligible = []
    for method in METHODS[1:]:
        gains = []
        okay = True
        for variant in VARIANTS:
            before = scores[variant+':original']; after = scores[variant+':'+method]
            gains.append((after['macro_recall'] or 0)-(before['macro_recall'] or 0))
            if after['false_positives_per_image'] > before['false_positives_per_image']+.05: okay = False
        if okay and sum(gains)/len(gains) > .005: eligible.append((sum(gains)/len(gains), method))
    selected = max(eligible)[1] if eligible else 'original'
    return {'selected_method': selected, 'development_scores': scores,
            'reason': 'Development diagnostics satisfy fixed recall/false-positive gate.' if eligible else 'No correction method passed development gate; original remains default.',
            'not_accuracy_claim': 'Candidate count or score growth is not accuracy improvement.'}


def aggregation_reuse_proof(previous, current):
    """Only permit a documented aggregation repair, never retag old predictions."""
    import ast
    from optical_agent.paths import PROJECT_ROOT
    if {k:v for k,v in previous.items() if k != 'code'} != {k:v for k,v in current.items() if k != 'code'}:
        raise ValueError('Model, input, split or declared experiment parameters changed')
    untouched = ('quality.py', 'optical_agent/general_vision.py', 'optical_agent/general_catalog.py',
                 'optical_agent/image_input.py')
    if any(previous['code'][p] != current['code'][p] for p in untouched):
        raise ValueError('Actual visual implementation changed; cannot reuse raw inference')
    function_checks = {
        'optical_agent/general_eval.py': ('read_rgb', 'sha256', 'approved_path', 'atomic_json',
                                         'iou', 'match_detections', 'operating_metrics'),
        'optical_agent/experiments/evaluate_general_vision.py': ('perturb', 'diagnostic_subset',
            'process_one', 'runtime_signature', 'coco_metrics', 'summarize_rows', 'select_policy')}
    allowed = {'optical_agent/general_eval.py': {'calibrate_class_thresholds'},
        'optical_agent/experiments/evaluate_general_vision.py':
            {'aggregation_reuse_proof', 'reaggregate_cached', 'evaluate_stage', 'main'}}
    results = {}
    def hashes(path, names):
        tree = ast.parse(path.read_text('utf8'))
        nodes = {n.name:n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        return {name: hashlib.sha256(ast.dump(nodes[name], include_attributes=False).encode()).hexdigest() for name in names}
    for rel, names in function_checks.items():
        old = ROOT/'source_runtime_v2'/rel
        if not old.exists() or sha256(old) != previous['code'][rel]:
            raise ValueError('Original exact source snapshot unavailable: '+rel)
        before = hashes(old, names); after = hashes(PROJECT_ROOT/rel, names)
        if before != after: raise ValueError('Prediction or scoring function changed: '+rel)
        def remaining(path):
            tree = ast.parse(path.read_text('utf8'))
            tree.body = [node for node in tree.body if not
                         (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in allowed[rel])]
            return ast.dump(tree, include_attributes=False)
        if remaining(old) != remaining(PROJECT_ROOT/rel):
            raise ValueError('Undeclared module-level implementation changed: '+rel)
        results[rel] = {'functions': before, 'unchanged': True, 'module_structure_except_declared_repairs_unchanged': True}
    return {'unchanged_visual_source_hashes': {p: current['code'][p] for p in untouched},
        'function_ast_checks': results, 'allowed_change': 'Insufficient score-bin calibration thresholds revert to .35; orchestration adds explicit aggregation-only provenance.',
        'raw_prediction_files_retagged': False, 'new_inference_requests': 0}


def reaggregate_cached(manifest, previous, current, detector, ontology_version):
    """Re-score unchanged real inference and preserve the previous artifacts."""
    import shutil
    proof = aggregation_reuse_proof(previous, current)
    identity_path = ROOT/'aggregation_identity.json'
    if identity_path.exists():
        existing = json.loads(identity_path.read_text('utf8'))
        if existing['current_signature'] != current or existing['previous_signature'] != previous:
            raise ValueError('Existing aggregation bridge belongs to another implementation')
        if (existing['calibration_sha256'] != sha256(ROOT/'calibration.json')
                or existing['exposure_policy_sha256'] != sha256(ROOT/'exposure_policy.json')):
            raise ValueError('Reaggregated artifacts changed')
        return existing
    archive = ROOT/'before_aggregation_boundary_fix'; archive.mkdir(exist_ok=True)
    for name in ('evaluation_identity.json', 'development_summary.json', 'calibration_summary.json',
                 'calibration.json', 'exposure_policy.json'):
        source = ROOT/name; target = archive/name
        if not source.exists(): raise ValueError('Incomplete pre-fix pass: '+name)
        if target.exists():
            if sha256(target) != sha256(source): raise ValueError('Pre-fix archive already differs')
        else: shutil.copy2(source, target)
    count = previous['diagnostic_images_per_split']
    def load(records, variant, method):
        rows = []
        for record in records:
            path = ROOT/'predictions'/variant/method/(str(record['image_id'])+'.json')
            row = json.loads(path.read_text('utf8'))
            if row['signature'] != previous or row['split'] != record['split'] or row['id'] != record['id']:
                raise ValueError('Cached inference identity/image/split mismatch')
            rows.append(row)
        return rows
    results = {}; originals_by_split = {}; diagnostics_by_split = {}
    for split in ('development', 'calibration'):
        records = [r for r in manifest['records'] if r['dataset'] == 'coco_val2017' and r['split'] == split]
        originals = load(records, 'original', 'original'); originals_by_split[split] = originals
        subset = diagnostic_subset(records, count)
        diagnostics = {variant+':'+method: load(subset, variant, method) for variant in VARIANTS for method in METHODS}
        diagnostics_by_split[split] = diagnostics
        results[split] = {'version': '0.6.0', 'split': split, 'planned_original_images': len(originals),
            'original': summarize_rows(originals), 'coco_AP': coco_metrics(originals, manifest),
            'diagnostics': {key: summarize_rows(rows) for key, rows in diagnostics.items()},
            'diagnostic_images_per_split': count, 'synthetic_variants': list(VARIANTS[1:]),
            'sice_evaluation_status': manifest['sice_status'], 'paid_api_requests': 0,
            'scope': 'Official COCO val2017 project split; no model training. Synthetic variants are not real camera exposure changes.'}
    calibration = calibrate_class_thresholds(originals_by_split['calibration'], manifest['categories'],
        detector.model_sha256, ontology_version, manifest_sha256=current['manifest_sha256'])
    policy = select_policy(diagnostics_by_split['development'])
    policy.update(schema_version=1, profile_id='general_rgb_v1', model_sha256=detector.model_sha256,
                  ontology_version=ontology_version, manifest_sha256=current['manifest_sha256'])
    old_calibration = json.loads((archive/'calibration.json').read_text('utf8'))
    old_policy = json.loads((archive/'exposure_policy.json').read_text('utf8'))
    if calibration != old_calibration or policy != old_policy:
        raise ValueError('Aggregation repair changed this batch calibration/policy; requires explicit new experiment decision')
    def numerical(summary):
        summary = json.loads(json.dumps(summary)); summary['coco_AP'].pop('console', None)
        return summary
    scores_equal = {}
    for split, summary in results.items():
        old = json.loads((archive/(split+'_summary.json')).read_text('utf8'))
        scores_equal[split] = numerical(summary) == numerical(old)
        if not scores_equal[split]: raise ValueError('Reaggregated metrics changed: '+split)
    for split, summary in results.items(): atomic_json(ROOT/(split+'_summary.json'), summary)
    atomic_json(ROOT/'calibration.json', calibration); atomic_json(ROOT/'exposure_policy.json', policy)
    identity = {'schema_version': 1, 'at': utc_now(), 'previous_signature': previous,
        'current_signature': current, 'proof': proof, 'numeric_scores_equal': scores_equal,
        'calibration_values_equal': True, 'exposure_policy_values_equal': True,
        'insufficient_classes': sum(s != 'validated' for s in calibration['class_status'].values()),
        'all_insufficient_thresholds_default': all(calibration['class_thresholds'][n] == .35 for n,s in calibration['class_status'].items() if s != 'validated'),
        'calibration_sha256': sha256(ROOT/'calibration.json'), 'exposure_policy_sha256': sha256(ROOT/'exposure_policy.json'),
        'archive': 'before_aggregation_boundary_fix', 'scope': 'Explicit aggregation-only repair; no test data used, no original inference recomputed or relabeled.'}
    atomic_json(identity_path, identity)
    return identity


def evaluate_stage(stage, diagnostic_count=30, device=None):
    import torch
    torch.set_num_threads(4)
    from optical_agent.general_vision import GeneralDetector
    from optical_agent.general_catalog import ONTOLOGY_VERSION
    from optical_agent.paths import PROJECT_ROOT, code_files
    manifest_path = ROOT/'manifest.json'; manifest = json.loads(manifest_path.read_text('utf8'))
    audit_split(manifest['records'])
    detector = GeneralDetector(device=device, calibration_path=ROOT/'__no_calibration_for_raw_eval__.json')
    signature = runtime_signature(detector, manifest_path, diagnostic_count)
    signature_path = ROOT/'evaluation_identity.json'
    previous = json.loads(signature_path.read_text('utf8')) if signature_path.exists() else None
    if stage == 'reaggregate':
        if previous is None: raise ValueError('Raw inference identity is missing')
        return reaggregate_cached(manifest, previous, signature, detector, ONTOLOGY_VERSION)
    if previous is not None and previous != signature:
        bridge_path = ROOT/'aggregation_identity.json'
        if stage not in {'freeze', 'test'} or not bridge_path.exists():
            raise ValueError('Frozen identity changed; use explicit aggregation-only proof or a new experiment')
        bridge = json.loads(bridge_path.read_text('utf8'))
        if bridge['previous_signature'] != previous or bridge['current_signature'] != signature:
            raise ValueError('Aggregation bridge does not match actual current code')
        aggregation_reuse_proof(previous, signature)
    elif previous is None:
        atomic_json(signature_path, signature)
    if stage in {'development', 'calibration', 'test'}:
        freeze_path = ROOT/'freeze.json'
        if stage == 'test':
            if not freeze_path.exists(): raise ValueError('Freeze calibration and policy before held-out test')
            freeze = json.loads(freeze_path.read_text('utf8'))
            if (freeze['signature'] != signature or freeze['calibration_sha256'] != sha256(ROOT/'calibration.json')
                    or freeze['exposure_policy_sha256'] != sha256(ROOT/'exposure_policy.json')):
                raise ValueError('Frozen configuration/calibration changed')
            if freeze['recursive_code_sha256'] != {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in code_files()}:
                raise ValueError('Frozen recursive implementation changed')
        originals, diagnostics = predict_split(manifest['records'], stage, detector, signature, diagnostic_count)
        thresholds = None
        if stage == 'calibration':
            calibration = calibrate_class_thresholds(originals, manifest['categories'], detector.model_sha256,
                ONTOLOGY_VERSION, manifest_sha256=signature['manifest_sha256'])
            atomic_json(ROOT/'calibration.json', calibration)
        if stage == 'test':
            calibration = json.loads((ROOT/'calibration.json').read_text('utf8'))
            thresholds = {str(c['id']): calibration['class_thresholds'][c['name']] for c in manifest['categories']}
        summary = {'version': '0.6.0', 'split': stage, 'planned_original_images': len(originals),
            'original': summarize_rows(originals, thresholds), 'coco_AP': coco_metrics(originals, manifest),
            'diagnostics': {k: summarize_rows(v, thresholds) for k, v in diagnostics.items()},
            'diagnostic_images_per_split': diagnostic_count, 'synthetic_variants': list(VARIANTS[1:]),
            'sice_evaluation_status': manifest['sice_status'], 'paid_api_requests': 0,
            'scope': 'Official COCO val2017 project split; no model training. Synthetic variants are not real camera exposure changes.'}
        if stage == 'development':
            policy = select_policy(diagnostics)
            policy.update(schema_version=1, profile_id='general_rgb_v1', model_sha256=detector.model_sha256,
                          ontology_version=ONTOLOGY_VERSION, manifest_sha256=signature['manifest_sha256'])
            atomic_json(ROOT/'exposure_policy.json', policy)
        atomic_json(ROOT/(stage+'_summary.json'), summary)
        print(json.dumps({'split': stage, 'original_images': len(originals), 'AP50': summary['coco_AP']['AP50'],
                          'micro_recall': summary['original']['micro_recall']}), flush=True)
        return summary
    if stage == 'freeze':
        for name in ('development_summary.json', 'calibration_summary.json', 'exposure_policy.json', 'calibration.json'):
            if not (ROOT/name).exists(): raise ValueError('Missing pre-test artifact: '+name)
        freeze = {'signature': signature, 'created_at': utc_now(),
            'calibration_sha256': sha256(ROOT/'calibration.json'),
            'exposure_policy_sha256': sha256(ROOT/'exposure_policy.json'),
            'artifacts': {name: sha256(ROOT/name) for name in ('development_summary.json', 'calibration_summary.json')},
            'recursive_code_sha256': {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in code_files()},
            'raw_inference_identity_sha256': sha256(signature_path),
            'aggregation_identity_sha256': sha256(ROOT/'aggregation_identity.json') if (ROOT/'aggregation_identity.json').exists() else None,
            'test_rule': 'One frozen run, resume only exact cached identity; do not choose best repeated test score.'}
        path = ROOT/'freeze.json'
        if path.exists():
            old = json.loads(path.read_text('utf8'))
            if {k:v for k,v in old.items() if k != 'created_at'} != {k:v for k,v in freeze.items() if k != 'created_at'}:
                raise ValueError('Existing freeze cannot be changed')
        else: atomic_json(path, freeze)
        return freeze
    raise ValueError('Unknown stage')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['development', 'calibration', 'reaggregate', 'freeze', 'test', 'all'], default='development')
    parser.add_argument('--diagnostic-count', type=int, default=30)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    if args.diagnostic_count < 1: parser.error('diagnostic-count must be positive and frozen before running')
    stages = ['development', 'calibration', 'freeze', 'test'] if args.stage == 'all' else [args.stage]
    for stage in stages: evaluate_stage(stage, args.diagnostic_count, args.device)


if __name__ == '__main__': main()
