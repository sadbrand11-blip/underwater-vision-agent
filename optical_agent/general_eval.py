"""Independent natural-image data grouping and detection metrics.

No labels or detector output are sent to an LLM. All paths are local and the
download/evaluation entrypoints keep their artifacts in the approved data root.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re

ROOT = Path(r'D:\CodexData\optical_agent\general_v060')
SEED = 20261023
SPLITS = {'development': .60, 'calibration': .20, 'test': .20}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    tmp.replace(path)


def approved_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('General-image artifacts must stay inside ' + str(ROOT))
    return path


def validate_archive_names(names):
    for name in names:
        normalized = name.replace('\\', '/')
        parts = PurePosixPath(normalized).parts
        if (not normalized or normalized.startswith('/') or re.match(r'^[A-Za-z]:', normalized)
                or '..' in parts or '\x00' in normalized
                or any(':' in p or p.rstrip(' .') in {'', '..'} for p in parts)):
            raise ValueError('Unsafe archive member: ' + name)


def read_rgb(path):
    from PIL import Image, ImageOps
    import numpy as np
    with Image.open(path) as image:
        if image.mode in {'I', 'F', 'I;16', 'I;16B', 'I;16L'}:
            raise ValueError('Only 8-bit natural images are supported')
        rgb = np.asarray(ImageOps.exif_transpose(image).convert('RGB')).copy()
    return rgb


def fingerprint(path):
    import cv2
    import numpy as np
    rgb = read_rgb(path)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    block = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype('float32')
    coeff = cv2.dct(block)[:8, :8].flatten()
    bits = coeff > np.median(coeff[1:])
    return {'width': rgb.shape[1], 'height': rgb.shape[0], 'file_sha256': sha256(path),
            'pixel_sha256': hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest(),
            'phash': sum(int(b) << i for i, b in enumerate(bits))}


class _Union:
    def __init__(self, n): self.p = list(range(n))
    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]; i = self.p[i]
        return i
    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        self.p[max(a, b)] = min(a, b)


def group_and_split(records, seed=SEED, phash_radius=4):
    """Union entire scenes, exact pixels and near duplicates before allocation.

    pHash is a conservative heuristic, not proof of true scene independence.
    The small benchmark uses a BK tree rather than a quadratic image scan.
    """
    from optical_agent.vision_data import HashTree
    union = _Union(len(records)); exact = {}; scenes = {}; tree = HashTree(); links = Counter()
    for i, row in enumerate(records):
        scene = row.get('scene_id')
        if scene:
            if scene in scenes: union.join(i, scenes[scene]); links['scene'] += 1
            scenes.setdefault(scene, i)
        for fp in [row, *row.get('reference_fingerprints', [])]:
            pixel = fp['pixel_sha256']
            if pixel in exact: union.join(i, exact[pixel]); links['exact'] += 1
            exact.setdefault(pixel, i)
            for j in tree.query_add(int(fp['phash']), i, radius=phash_radius):
                # Extreme solid images share pHash despite different content;
                # grouping extra images is conservative and documented.
                union.join(i, j); links['perceptual'] += 1
    buckets = defaultdict(list)
    for i, row in enumerate(records): buckets[union.find(i)].append(row)
    order = list(buckets)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda k: -len(buckets[k]))
    counts = {s: Counter() for s in SPLITS}; totals = Counter()
    for row in records:
        totals[row['dataset']] += 1
    for key in order:
        rows = buckets[key]; feature = Counter(r['dataset'] for r in rows)
        def cost(split):
            return sum(((counts[split][k] + v - totals[k] * SPLITS[split]) ** 2
                        - (counts[split][k] - totals[k] * SPLITS[split]) ** 2) / max(totals[k], 1)
                       for k, v in feature.items())
        split = min(SPLITS, key=cost)
        counts[split].update(feature)
        group_id = hashlib.sha256('|'.join(sorted(r['id'] for r in rows)).encode()).hexdigest()[:20]
        for row in rows: row.update(group_id=group_id, split=split)
    return {'seed': seed, 'ratios': SPLITS, 'phash_radius': phash_radius, 'links': dict(links),
            'groups': len(buckets), 'largest_group': max(map(len, buckets.values()), default=0),
            'cross_dataset_groups': sum(len({r['dataset'] for r in rows}) > 1 for rows in buckets.values()),
            'counts': {s: dict(c) for s, c in counts.items()},
            'limitation': 'Exact/pHash groups cannot guarantee independent real-world scenes.'}


def audit_split(records):
    ids = set(); groups = {}; pixels = {}; scenes = {}
    for row in records:
        if row['id'] in ids: raise ValueError('Duplicate record id')
        ids.add(row['id'])
        for mapping, key in [(groups, row['group_id']), (pixels, row['pixel_sha256']),
                             (scenes, row.get('scene_id'))]:
            if key is None: continue
            if key in mapping and mapping[key] != row['split']: raise ValueError('Group leakage')
            mapping[key] = row['split']
        for fp in row.get('reference_fingerprints', []):
            key = fp['pixel_sha256']
            if key in pixels and pixels[key] != row['split']: raise ValueError('Reference leakage')
            pixels[key] = row['split']


def iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., right - left) * max(0., bottom - top)
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-12)


def match_detections(predictions, ground_truth, threshold=.35):
    """Score-ordered one-to-one matching at IoU .5, with COCO crowd ignore.

    Crowd-overlapping predictions are ignored using intersection/prediction-area,
    not counted as false positives. Official AP is separately pycocotools COCOeval.
    """
    used = set(); labels = []
    valid = [g for g in ground_truth if not g.get('iscrowd', False)]
    crowd = [g for g in ground_truth if g.get('iscrowd', False)]
    for p in sorted(predictions, key=lambda p: -p['score']):
        if p['score'] < threshold: continue
        candidates = [(iou(p['box'], g['box']), j) for j, g in enumerate(valid)
                      if j not in used and p['class_id'] == g['class_id']]
        overlap, j = max(candidates, default=(0., -1))
        if overlap >= .5:
            used.add(j); labels.append({'prediction': p, 'tp': 1, 'ignored': False}); continue
        ignored = False
        for g in crowd:
            if p['class_id'] != g['class_id']: continue
            a, b = p['box'], g['box']
            intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
            area = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
            if intersection / max(area, 1e-12) >= .5: ignored = True; break
        labels.append({'prediction': p, 'tp': 0, 'ignored': ignored})
    tp = sum(x['tp'] for x in labels)
    fp = sum(not x['tp'] and not x['ignored'] for x in labels)
    return {'tp': tp, 'fp': fp, 'fn': len(valid)-tp, 'gt': len(valid), 'labels': labels}


def operating_metrics(rows, thresholds=None, default_threshold=.35):
    counts = defaultdict(Counter)
    for row in rows:
        cats = {p['class_id'] for p in row['predictions']} | {g['class_id'] for g in row['ground_truth']}
        for cat in cats:
            predictions = [p for p in row['predictions'] if p['class_id'] == cat]
            truth = [g for g in row['ground_truth'] if g['class_id'] == cat]
            threshold = (thresholds or {}).get(str(cat), default_threshold)
            m = match_detections(predictions, truth, threshold)
            counts[cat].update({k: m[k] for k in ('tp', 'fp', 'fn', 'gt')})
    by_class = {}
    for cat, c in sorted(counts.items()):
        by_class[str(cat)] = {**dict(c), 'recall': c['tp']/c['gt'] if c['gt'] else None,
                             'precision': c['tp']/(c['tp']+c['fp']) if c['tp']+c['fp'] else None}
    totals = Counter()
    for c in counts.values(): totals.update(c)
    recalls = [v['recall'] for v in by_class.values() if v['recall'] is not None]
    return {'images': len(rows), **dict(totals), 'micro_recall': totals['tp']/totals['gt'] if totals['gt'] else None,
            'macro_recall': sum(recalls)/len(recalls) if recalls else None,
            'false_positives_per_image': totals['fp']/len(rows) if rows else None, 'classes': by_class}


def calibrate_class_thresholds(rows, categories, model_sha256, ontology_version,
                               manifest_sha256=None, preprocessing_version='jpeg_png_8bit_exif_rgb_v1'):
    """Calibration-only threshold selection; sparse classes remain insufficient."""
    if any(row.get('split') != 'calibration' for row in rows):
        raise ValueError('Thresholds may only use calibration data')
    import numpy as np
    classes = {}; class_status = {}; class_thresholds = {}
    for category in categories:
        cid, name = category['id'], category['name']
        gt_count = sum(sum(g['class_id'] == cid and not g.get('iscrowd') for g in r['ground_truth']) for r in rows)
        image_count = sum(any(g['class_id'] == cid and not g.get('iscrowd') for g in r['ground_truth']) for r in rows)
        candidates = []
        for threshold in [round(.1 + .05*i, 2) for i in range(17)]:
            matches = [match_detections([p for p in r['predictions'] if p['class_id'] == cid],
                                       [g for g in r['ground_truth'] if g['class_id'] == cid], threshold) for r in rows]
            tp = sum(m['tp'] for m in matches); fp = sum(m['fp'] for m in matches)
            if tp+fp >= 30 and tp/(tp+fp) >= .8 and gt_count >= 30 and image_count >= 10:
                candidates.append((tp/max(gt_count, 1), -threshold, threshold, tp, fp))
        selected = max(candidates) if candidates else None
        threshold = selected[2] if selected else .35
        # Calibration curve uses disjoint score bins and raw actual box outcomes.
        labels = []
        for row in rows:
            matched = match_detections([p for p in row['predictions'] if p['class_id'] == cid],
                                      [g for g in row['ground_truth'] if g['class_id'] == cid], .05)
            labels.extend((v['prediction']['score'], v['tp']) for v in matched['labels'] if not v['ignored'])
        labels.sort()
        bins = [b.tolist() for b in np.array_split(np.asarray(labels), min(5, max(1, len(labels)//20))) if len(b)] if labels else []
        x = [float(np.mean([p[0] for p in b])) for b in bins]
        y = [float(np.mean([p[1] for p in b])) for b in bins]
        status = 'validated' if selected and len(x) >= 2 else 'insufficient'
        if status == 'insufficient':
            threshold = .35
        classes[name] = {'status': status, 'threshold': threshold, 'ground_truth_instances': gt_count,
                         'positive_images': image_count, 'prediction_samples': len(labels),
                         'calibration': {'method': 'disjoint_score_bins_iou50', 'x': x, 'y': y},
                         'selection_precision': selected[3]/(selected[3]+selected[4]) if selected else None,
                         'selection_recall': selected[0] if selected else None}
        class_status[name] = status; class_thresholds[name] = threshold
    return {'schema_version': 1, 'profile_id': 'general_rgb_v1', 'model_sha256': model_sha256,
            'preprocessing_version': preprocessing_version, 'calibration_scope': 'coco_originals',
            'manifest_sha256': manifest_sha256, 'split': 'calibration', 'image_count': len(rows),
            'ontology_version': ontology_version, 'status': 'available' if any(s == 'validated' for s in class_status.values()) else 'insufficient',
            'class_status': class_status, 'class_thresholds': class_thresholds, 'classes': classes,
            'calibration': {'method': 'per_class_disjoint_score_bins_iou50', 'x': [], 'y': []},
            'selection_rule': {'min_gt_instances': 30, 'min_positive_images': 10, 'min_predictions': 30,
                               'min_precision': .8, 'threshold_grid': [.1, .9, .05], 'objective': 'max_recall_then_lower_threshold'},
            'scope': 'COCO calibration project split only; not a universal probability or field guarantee.'}


def utc_now(): return datetime.now(timezone.utc).isoformat()
