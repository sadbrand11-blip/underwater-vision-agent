"""Validated, globally grouped experimental data. No model output supplies labels."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random

import cv2
import numpy as np

SEED = 20261001
ROOT = Path(r'D:\CodexData\optical_agent')
SODD_CLASSES = ['background', 'propeller', 'pipe_type2', 'red_fin', 'net', 'qr_codes', 'pipe']
ROBOT_CLASSES = ['background', 'underwater_robot']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    temp.replace(path)


def d_path(path):
    path = Path(path).resolve()
    if path.drive.upper() != 'D:' or not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Data and training artifacts must stay in D:/CodexData/optical_agent')
    return path


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or min(image.shape[:2]) < 8:
        raise ValueError('Image cannot be decoded or is smaller than 8 pixels: ' + str(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def fingerprints(path):
    rgb = read_rgb(path)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8].flatten()
    bits = dct > np.median(dct[1:])
    phash = sum(int(x) << i for i, x in enumerate(bits))
    return {'width': rgb.shape[1], 'height': rgb.shape[0], 'file_sha256': digest(path),
            'pixel_sha256': hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest(),
            'phash': phash}


def validate_boxes(boxes, width, height, class_count):
    for box in boxes:
        coords = box['box']
        if len(coords) != 4 or not all(np.isfinite(coords)):
            raise ValueError('Nonfinite/invalid box')
        x0, y0, x1, y1 = coords
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError('Box outside image or empty')
        if not 1 <= box['class_id'] < class_count:
            raise ValueError('Invalid class id')


class Groups:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        a, b = self.find(i), self.find(j)
        self.parent[max(a, b)] = min(a, b)


class HashTree:
    """BK tree for Hamming near duplicates, avoiding a quadratic all-pairs scan."""
    def __init__(self):
        self.root = None

    def query_add(self, key, index, radius=6):
        hits = []
        stack = [self.root] if self.root else []
        while stack:
            node = stack.pop()
            distance = (key ^ node[0]).bit_count()
            if distance <= radius:
                hits.append(node[1])
            stack.extend(child for edge, child in node[2].items() if distance-radius <= edge <= distance+radius)
        if self.root is None:
            self.root = [key, index, {}]
        else:
            node = self.root
            while True:
                distance = (key ^ node[0]).bit_count()
                if distance == 0:
                    break
                if distance not in node[2]:
                    node[2][distance] = [key, index, {}]
                    break
                node = node[2][distance]
        return hits


def assign_groups(records, seed=SEED):
    """One allocation shared by all datasets and reference variants."""
    union, exact, tree = Groups(len(records)), {}, HashTree()
    links = Counter()
    for i, record in enumerate(records):
        for fp in [record, *record.get('reference_fingerprints', [])]:
            key = fp['pixel_sha256']
            if key in exact:
                union.union(i, exact[key]); links['exact'] += 1
            else:
                exact[key] = i
            for j in tree.query_add(fp['phash'], i):
                union.union(i, j); links['perceptual'] += 1
    buckets = defaultdict(list)
    for i, record in enumerate(records):
        buckets[union.find(i)].append(record)
    features, totals = {}, Counter()
    for key, rows in buckets.items():
        feature = Counter()
        for row in rows:
            feature['dataset:' + row['dataset']] += 1
            for box in row.get('boxes', []):
                feature[row['dataset'] + ':class:' + str(box['class_id'])] += 1
        features[key] = feature
        totals.update(feature)
    proportions = {'train': .70, 'selection': .075, 'calibration': .075, 'test': .15}
    counts = {s: Counter() for s in proportions}
    order = list(buckets)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda key: -len(buckets[key]))
    for key in order:
        rows, feature = buckets[key], features[key]
        if any(r['dataset'] == 'uieb_challenge' for r in rows):
            split = 'test'
        else:
            def cost(split):
                return sum(((counts[split][f]+v-totals[f]*proportions[split])**2
                            -(counts[split][f]-totals[f]*proportions[split])**2)/max(totals[f], 1)
                           for f, v in feature.items())
            split = min(proportions, key=cost)
        counts[split].update(feature)
        gid = 'group_' + hashlib.sha256('|'.join(sorted(r['id'] for r in rows)).encode()).hexdigest()[:16]
        for row in rows:
            row['group_id'], row['split'] = gid, split
    return {'links': dict(links), 'groups': len(buckets),
            'cross_dataset_groups': sum(len({r['dataset'] for r in rows}) > 1 for rows in buckets.values()),
            'largest_group': max(map(len, buckets.values()), default=0)}


def audit_manifest(manifest):
    groups, ids, pixels, tree, rows = {}, set(), {}, HashTree(), manifest['records']
    for r in manifest['records']:
        if r['id'] in ids:
            raise ValueError('Duplicate record id')
        ids.add(r['id'])
        if r['group_id'] in groups and groups[r['group_id']] != r['split']:
            raise ValueError('Group leakage across splits')
        groups[r['group_id']] = r['split']
        for fp in [r, *r.get('reference_fingerprints', [])]:
            pixel = fp['pixel_sha256']
            if pixel in pixels and pixels[pixel] != r['split']:
                raise ValueError('Duplicate pixels leak across splits')
            pixels[pixel] = r['split']
            for j in tree.query_add(fp['phash'], len(ids)-1):
                if rows[j]['split'] != r['split']:
                    raise ValueError('Perceptual near duplicate leaks across splits')
        if r['dataset'] == 'uieb_challenge' and r['split'] != 'test':
            raise ValueError('Challenge data cannot select models')
        if r.get('reference') and (r['width'], r['height']) != (r['reference_width'], r['reference_height']):
            raise ValueError('Reference dimensions differ')
        validate_boxes(r.get('boxes', []), r['width'], r['height'], 7 if r['dataset'] == 'sodd' else 2)
    return True


def prepare(root=ROOT):
    root = d_path(root)
    records, quarantined = [], []
    raw = root / 'uieb/data/raw-890'
    refs = root / 'uieb/data/reference-890'
    challenge = root / 'uieb/data/challenging-60'
    def images(folder):
        return sorted(p for p in folder.rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    refmap = {p.name: p for p in images(refs)}
    raw_names = {p.name for p in images(raw)}
    quarantined += [{'file':str(p),'reason':'orphan_reference'} for n,p in refmap.items() if n not in raw_names]
    for p in images(raw):
        if p.name not in refmap:
            quarantined.append({'file': str(p), 'reason': 'missing_reference'})
        else:
            records.append({'id': 'uieb:' + p.name, 'dataset': 'uieb', 'path': str(p), 'reference': str(refmap[p.name])})
    for p in images(challenge):
        records.append({'id': 'challenge:' + p.name, 'dataset': 'uieb_challenge', 'path': str(p)})
    sodd = root / 'sodd/SODD/data'
    for p in images(sodd):
        if not p.stem.endswith('_original'):
            continue
        label = p.parent.parent / 'labels' / (p.stem + '.txt')
        if not label.exists():
            quarantined.append({'file': str(p), 'reason': 'missing_label'})
        else:
            records.append({'id': 'sodd:' + p.name, 'dataset': 'sodd', 'path': str(p), 'label': str(label)})
    annotation_paths = sorted((root/'uiis10k/data').rglob('multiclass_*.json'))
    if not annotation_paths:
        raise ValueError('No official UIIS10K multiclass COCO annotation files found')
    uiis_categories = None
    all_uiis = {p.name: p for p in images(root/'uiis10k/data')}
    for annotation in annotation_paths:
        data = json.loads(annotation.read_text(encoding='utf8'))
        uiis_categories = data['categories']
        robots = {c['id'] for c in data['categories'] if any(x in c['name'].lower() for x in ('robot', 'rov', 'auv'))}
        if len(robots) != 1:
            raise ValueError('Robot category must be verified explicitly: ' + repr(data['categories']))
        boxes = defaultdict(list)
        for a in data['annotations']:
            if a['category_id'] in robots:
                x, y, w, h = a['bbox']
                boxes[a['image_id']].append({'box': [x, y, x+w, y+h], 'class_id': 1})
        for image in data['images']:
            p = all_uiis.get(Path(image['file_name']).name)
            if p is None:
                quarantined.append({'file': image['file_name'], 'reason': 'missing_coco_image'})
                continue
            records.append({'id': 'uiis10k:' + p.name, 'dataset': 'uiis10k', 'path': str(p),
                            'boxes': boxes[image['id']], 'annotation': str(annotation),
                            'declared_width': image['width'], 'declared_height': image['height']})
    def inspect(row):
        row = dict(row)
        try:
            row.update(fingerprints(row['path']))
            if 'reference' in row:
                fp = fingerprints(row['reference'])
                row.update(reference_width=fp['width'], reference_height=fp['height'], reference_sha256=fp['file_sha256'])
                row['reference_fingerprints'] = [fp]
                if (fp['width'], fp['height']) != (row['width'], row['height']):
                    raise ValueError('pair_dimension_mismatch')
            if row['dataset'] == 'sodd':
                row['boxes'] = []
                for line in Path(row['label']).read_text().splitlines():
                    c, x, y, w, h = map(float, line.split())
                    row['boxes'].append({'class_id': int(c)+1, 'box': [(x-w/2)*row['width'], (y-h/2)*row['height'],
                                          (x+w/2)*row['width'], (y+h/2)*row['height']]})
                row['label_sha256'] = digest(row['label'])
            if 'declared_width' in row and (row['width'], row['height']) != (row['declared_width'], row['declared_height']):
                raise ValueError('coco_dimension_mismatch')
            validate_boxes(row.get('boxes', []), row['width'], row['height'], 7 if row['dataset'] == 'sodd' else 2)
            return row, None
        except (ValueError, OSError) as exc:
            return None, {'file': row['path'], 'reason': str(exc)}
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=6) as pool:
        checked = list(pool.map(inspect, records))
    records = [row for row, error in checked if row is not None]
    quarantined += [error for row, error in checked if error is not None]
    grouping = assign_groups(records)
    manifest = {'schema': 1, 'seed': SEED, 'split_policy': 'project_experimental_global_groups_70_7.5_7.5_15',
                'scene_independence': 'unknown; no verified sequence ids; exact pixels and pHash Hamming<=6 only',
                'grouping': grouping, 'records': records, 'quarantined': quarantined,
                'uiis_categories': uiis_categories, 'annotation_hashes': {str(p): digest(p) for p in annotation_paths}}
    audit_manifest(manifest)
    manifest['summary'] = {ds: {s: {'images': sum(r['dataset']==ds and r['split']==s for r in records),
        'groups': len({r['group_id'] for r in records if r['dataset']==ds and r['split']==s}),
        'instances': dict(Counter(str(b['class_id']) for r in records if r['dataset']==ds and r['split']==s for b in r.get('boxes', [])))}
        for s in ['train', 'selection', 'calibration', 'test']} for ds in sorted({r['dataset'] for r in records})}
    output = root/'vision_v040/manifest.json'
    if output.exists():
        old = json.loads(output.read_text(encoding='utf8'))
        if old != manifest:
            raise ValueError('Frozen manifest changed; use a new experiment directory')
    save_json(output, manifest)
    print(json.dumps({'manifest': str(output), 'sha256': digest(output), 'grouping': grouping,
                      'summary': manifest['summary'], 'quarantined': len(quarantined)}, ensure_ascii=False), flush=True)
    return manifest


def audit_source_annotations(root=ROOT):
    results=[]
    for path in sorted((root/'uiis10k/data').rglob('multiclass_*.json')):
        data=json.loads(path.read_text(encoding='utf8'))
        dimensions={i['id']:(i['width'],i['height']) for i in data['images']}
        errors=[]
        for a in data['annotations']:
            try:
                x,y,w,h=a['bbox']
                validate_boxes([{'box':[x,y,x+w,y+h],'class_id':a['category_id']}],*dimensions[a['image_id']],11)
            except (ValueError,KeyError) as exc:
                errors.append({'id':a['id'],'image_id':a['image_id'],'reason':str(exc)})
        results.append({'path':str(path),'sha256':digest(path),'annotations':len(data['annotations']),'errors':errors})
    save_json(root/'vision_v040/source_annotation_audit.json',results)
    print(json.dumps({'source_annotation_validation':[(r['annotations'],len(r['errors'])) for r in results]}),flush=True)
    return results


def quarantine_invalid_annotations():
    """Pre-test correction keeps every unaffected group/split stable; original audit stays recoverable."""
    audits=audit_source_annotations()
    bad={}
    for report in audits:
        data=json.loads(Path(report['path']).read_text(encoding='utf8'))
        names={i['id']:i['file_name'] for i in data['images']}
        for error in report['errors']:
            bad['uiis10k:'+Path(names[error['image_id']]).name]=error['reason']
    path=ROOT/'vision_v040/manifest.json'
    manifest=load_manifest(path)
    removed=[r for r in manifest['records'] if r['id'] in bad]
    if not removed:
        return
    if (ROOT/'vision_v040/freeze.json').exists():
        raise ValueError('Cannot change data after held-out test freezing')
    backup=path.with_name('manifest.before_allclass_validation.json')
    if not backup.exists():
        save_json(backup,manifest)
    manifest['records']=[r for r in manifest['records'] if r['id'] not in bad]
    manifest['quarantined'] += [{'file':r['path'],'reason':bad[r['id']],'action':'excluded; original file retained'} for r in removed]
    manifest['data_validation_fix']={'previous_manifest_sha256':digest(backup),'excluded_ids':sorted(bad),
        'unaffected_splits':'preserved','scope':'before any held-out metric; SODD/UIEB training and selection rows unchanged'}
    for ds,splits in manifest['summary'].items():
        for split in splits:
            rows=[r for r in manifest['records'] if r['dataset']==ds and r['split']==split]
            splits[split]={'images':len(rows),'groups':len({r['group_id'] for r in rows}),
                'instances':dict(Counter(str(b['class_id']) for r in rows for b in r.get('boxes',[])))}
    manifest['grouping']['groups']=len({r['group_id'] for r in manifest['records']})
    audit_manifest(manifest); save_json(path,manifest)
    print(json.dumps({'quarantined':list(bad),'manifest_sha256':digest(path)}),flush=True)


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding='utf8'))
    audit_manifest(manifest)
    return manifest


def subset(manifest, dataset, split, purpose='evaluation'):
    if purpose in {'selection', 'calibration', 'training'} and split == 'test':
        raise ValueError('Held-out test data is forbidden during model/config selection')
    return [r for r in manifest['records'] if r['dataset'] == dataset and r['split'] == split]
