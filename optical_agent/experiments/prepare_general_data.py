"""Download official COCO/SICE, check integrity and freeze group-level splits."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import zipfile

from optical_agent.general_eval import (ROOT, SEED, approved_path, atomic_json, audit_split,
    fingerprint, group_and_split, sha256, utc_now, validate_archive_names)

COCO_URLS = {
    'val2017.zip': 'http://images.cocodataset.org/zips/val2017.zip',
    'annotations_trainval2017.zip': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
}
SICE_SOURCES = [
    'https://drive.usercontent.google.com/download?id=1HiLtYiyT9R7dR9DRTLRlUUrAicC4zzWN&export=download&confirm=t',
    'https://drive.usercontent.google.com/download?id=16VoHNPAZ5Js19zspjFOsKiGRrfkDgHoN&export=download&confirm=t',
    'https://pan.baidu.com/s/1kXotehL',
    'https://pan.baidu.com/s/1x1Dq9xef1dBTXXHcMjPAyA',
    'https://pan.baidu.com/s/1zZR5xU92q7UwcCJq-_9xmQ',
]


def download(url, target, attempts=4):
    import requests
    target = approved_path(target); target.parent.mkdir(parents=True, exist_ok=True)
    record = target.with_suffix(target.suffix + '.source.json')
    if target.exists() and record.exists():
        old = json.loads(record.read_text('utf8'))
        if old['url'] == url and old['sha256'] == sha256(target): return old
        raise ValueError('Existing download identity changed')
    session = requests.Session(); session.trust_env = False
    part = target.with_suffix(target.suffix + '.part'); history = []
    for attempt in range(attempts):
        offset = part.stat().st_size if part.exists() else 0
        try:
            with session.get(url, headers={'Range': f'bytes={offset}-'} if offset else {},
                             stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                if 'text/html' in response.headers.get('Content-Type', ''):
                    text = response.content[:60000].decode('utf8', errors='replace')
                    html = target.with_suffix('.access.html'); html.write_text(text, encoding='utf8')
                    reason = 'Google Drive download quota exceeded' if 'Quota exceeded' in text else 'HTML access/confirmation page'
                    raise RuntimeError(f'HTTP {response.status_code}: official endpoint returned {reason}, not an archive')
                mode = 'ab' if offset and response.status_code == 206 else 'wb'
                expected = int(response.headers.get('Content-Length', 0))
                received = 0; before = offset if mode == 'ab' else 0
                with part.open(mode) as stream:
                    for block in response.iter_content(2 * 1024 * 1024):
                        if block: stream.write(block); received += len(block)
                if expected and expected != received: raise IOError('Incomplete response body')
                history.append({'attempt': attempt+1, 'http_status': response.status_code,
                                'resumed_bytes': before, 'received_bytes': received})
            part.replace(target)
            result = {'url': url, 'resolved_url': response.url, 'size': target.stat().st_size,
                      'sha256': sha256(target), 'checksum_scope': 'local_integrity_not_publisher_digest',
                      'at': utc_now(), 'attempts': history}
            atomic_json(record, result); return result
        except Exception as exc:
            history.append({'attempt': attempt+1, 'error_type': type(exc).__name__, 'error': str(exc)[:500]})
            atomic_json(target.with_suffix('.failure.json'), {'url': url, 'at': utc_now(), 'attempts': history})
            if isinstance(exc, RuntimeError) or attempt+1 == attempts: raise
            time.sleep(min(2**attempt, 8))


def extract_zip(archive, destination, only_val_annotations=False):
    destination = approved_path(destination); destination.mkdir(parents=True, exist_ok=True)
    digest = sha256(archive); marker = destination/(Path(archive).name + '.extracted.json')
    if marker.exists() and json.loads(marker.read_text('utf8'))['sha256'] == digest: return
    with zipfile.ZipFile(archive) as z:
        validate_archive_names(z.namelist())
        if any((i.external_attr >> 16) & 0o170000 == 0o120000 for i in z.infolist()): raise ValueError('Archive symlink')
        bad = z.testzip()
        if bad: raise ValueError('ZIP CRC failed: ' + bad)
        names = ['annotations/instances_val2017.json'] if only_val_annotations else z.namelist()
        for name in names: z.extract(name, destination)
    atomic_json(marker, {'sha256': digest, 'crc_checked': True, 'at': utc_now(), 'extracted_members': len(names)})


def download_coco():
    def one(item):
        name, url = item; path = ROOT/'coco/downloads'/name
        # The documented COCO host is an S3 bucket. Its path-style HTTPS URL
        # avoids the legacy host's certificate mismatch and supports ranges.
        delivery_url = url.replace('http://images.cocodataset.org/', 'https://s3.amazonaws.com/images.cocodataset.org/')
        if name.startswith('annotations'):
            return download_val_annotations(delivery_url, published_url=url)
        result = parallel_coco_download(delivery_url, path, published_url=url, workers=48)
        print(name + ' downloaded, checking CRC', flush=True)
        extract_zip(path, ROOT/'coco', only_val_annotations=name.startswith('annotations'))
        return result
    with ThreadPoolExecutor(max_workers=2) as pool: results = list(pool.map(one, COCO_URLS.items()))
    atomic_json(ROOT/'coco/sources.json', {'source_page': 'https://cocodataset.org/#download', 'files': results})


def download_val_annotations(url, published_url):
    """Read only the official ZIP's val instances member via verified ranges.

    ZIP central-directory metadata and zlib CRC validate the selected member;
    this intentionally does not claim a checksum of the undownloaded archive.
    """
    import io
    import requests
    target = ROOT/'coco/annotations/instances_val2017.json'
    marker = target.with_suffix('.source.json')
    if target.exists() and marker.exists():
        old = json.loads(marker.read_text('utf8'))
        if old['sha256'] != sha256(target): raise ValueError('Annotation hash changed')
        return old
    session = requests.Session(); session.trust_env = False
    initial = session.get(url, headers={'Range': 'bytes=0-0'}, timeout=(30, 90))
    initial.raise_for_status()
    if initial.status_code != 206: raise ValueError('Annotation range unsupported')
    total = int(initial.headers['Content-Range'].split('/')[-1])
    class RemoteZip(io.RawIOBase):
        def __init__(self): self.position = 0; self.bytes_read = 0
        def readable(self): return True
        def seekable(self): return True
        def tell(self): return self.position
        def seek(self, offset, whence=0):
            self.position = offset if whence == 0 else self.position+offset if whence == 1 else total+offset
            if self.position < 0: raise ValueError('Negative remote ZIP seek')
            return self.position
        def read(self, size=-1):
            size = total-self.position if size < 0 else min(size, total-self.position)
            if size <= 0: return b''
            if size > 32*1024*1024: raise ValueError('Unexpectedly large ZIP metadata read')
            start, end = self.position, self.position+size-1
            if size > 1024*1024:
                parts = [(i, min(end, i+1024*1024-1)) for i in range(start, end+1, 1024*1024)]
                def fetch(bounds):
                    a, b = bounds
                    for attempt in range(4):
                        try:
                            connection = requests.Session(); connection.trust_env = False
                            result = connection.get(url, headers={'Range': f'bytes={a}-{b}'}, timeout=(30, 90))
                            result.raise_for_status()
                            if (result.status_code != 206 or result.headers.get('Content-Range') != f'bytes {a}-{b}/{total}'
                                    or len(result.content) != b-a+1): raise ValueError('Wrong annotation member range')
                            return result.content
                        except Exception:
                            if attempt == 3: raise
                with ThreadPoolExecutor(max_workers=12) as pool: content = b''.join(pool.map(fetch, parts))
                self.position += size; self.bytes_read += size
                return content
            for attempt in range(4):
                try:
                    result = session.get(url, headers={'Range': f'bytes={start}-{end}'}, timeout=(30, 90))
                    result.raise_for_status()
                    if result.status_code != 206 or result.headers.get('Content-Range') != f'bytes {start}-{end}/{total}':
                        raise ValueError('Wrong annotation ZIP range')
                    if len(result.content) != size: raise ValueError('Incomplete annotation range')
                    self.position += size; self.bytes_read += size
                    return result.content
                except Exception:
                    if attempt == 3: raise
            raise AssertionError('unreachable')
    remote = RemoteZip()
    with zipfile.ZipFile(remote) as z:
        validate_archive_names(z.namelist())
        member = z.getinfo('annotations/instances_val2017.json')
        data = z.read(member)  # ZipFile checks this member's decompressed CRC.
    parsed = json.loads(data)
    if len(parsed.get('images', [])) != 5000: raise ValueError('Unexpected COCO val image count')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix('.json.part'); partial.write_bytes(data); partial.replace(target)
    record = {'url': published_url, 'delivery_url': url, 'archive_size': total,
        'member': member.filename, 'member_crc32': member.CRC, 'member_crc_checked': True,
        'size': len(data), 'sha256': sha256(target), 'downloaded_range_bytes': remote.bytes_read,
        'checksum_scope': 'selected_member_local_integrity_not_whole_archive', 'at': utc_now()}
    atomic_json(marker, record)
    print('Downloaded verified val2017 annotation member only', flush=True)
    return record


def parallel_coco_download(url, target, published_url, chunk_bytes=4*1024*1024, workers=12):
    """Resume verified byte ranges from the same official COCO S3 bucket."""
    import requests
    target = approved_path(target); target.parent.mkdir(parents=True, exist_ok=True)
    marker = target.with_suffix(target.suffix+'.source.json')
    if target.exists() and marker.exists():
        old = json.loads(marker.read_text('utf8'))
        if old['sha256'] != sha256(target): raise ValueError('Completed COCO file checksum changed')
        return old
    session = requests.Session(); session.trust_env = False
    with session.get(url, headers={'Range': 'bytes=0-0'}, stream=True, timeout=(30, 45)) as response:
        response.raise_for_status()
        content_range = response.headers.get('Content-Range', '')
        if response.status_code != 206 or not content_range.startswith('bytes 0-0/'):
            raise RuntimeError('Official server did not honor byte ranges')
        total = int(content_range.split('/')[-1])
    folder = target.with_suffix('.ranges'); folder.mkdir(parents=True, exist_ok=True)
    def one(index):
        start = index*chunk_bytes; end = min(total, start+chunk_bytes)-1
        part = folder/f'{index:05d}.part'; needed = end-start+1
        if part.exists() and part.stat().st_size == needed: return index
        for attempt in range(5):
            offset = part.stat().st_size if part.exists() else 0
            if offset > needed: raise ValueError('Range file is larger than expected')
            try:
                connection = requests.Session(); connection.trust_env = False
                with connection.get(url, headers={'Range': f'bytes={start+offset}-{end}'}, stream=True,
                                    timeout=(30, 90)) as result:
                    result.raise_for_status()
                    expected = f'bytes {start+offset}-{end}/{total}'
                    if result.status_code != 206 or result.headers.get('Content-Range') != expected:
                        raise ValueError('Wrong HTTP byte range')
                    with part.open('ab' if offset else 'wb') as stream:
                        for block in result.iter_content(256*1024):
                            if block: stream.write(block)
                if part.stat().st_size != needed: raise IOError('Incomplete byte range')
                return index
            except Exception as exc:
                atomic_json(folder/f'{index:05d}.failure.json', {'url': url, 'range_start': start,
                    'range_end': end, 'attempt': attempt+1, 'at': utc_now(),
                    'error_type': type(exc).__name__, 'reason': str(exc)[:500]})
                if attempt == 4: raise
                time.sleep(min(2**attempt, 8))
    count = (total+chunk_bytes-1)//chunk_bytes
    completed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index in pool.map(one, range(count)):
            completed.append(index)
            atomic_json(folder/'progress.json', {'completed_ranges': len(completed), 'planned_ranges': count,
                'total_bytes': total, 'at': utc_now()})
            if len(completed) % 10 == 0: print(f'{target.name}: {len(completed)}/{count} ranges', flush=True)
    assembled = target.with_suffix(target.suffix+'.assembled')
    with assembled.open('wb') as output:
        for index in range(count):
            with (folder/f'{index:05d}.part').open('rb') as stream:
                for block in iter(lambda: stream.read(4*1024*1024), b''): output.write(block)
    if assembled.stat().st_size != total: raise ValueError('Assembled size mismatch')
    assembled.replace(target)
    record = {'url': published_url, 'delivery_url': url, 'official_bucket': 'images.cocodataset.org',
        'size': total, 'sha256': sha256(target), 'range_count': count, 'range_content_verified': True,
        'checksum_scope': 'local_integrity_not_publisher_digest', 'at': utc_now()}
    atomic_json(marker, record); return record


def attempt_sice():
    attempts = []
    for i, url in enumerate(SICE_SOURCES):
        try:
            if 'pan.baidu.com' in url:
                import requests
                session = requests.Session(); session.trust_env = False
                response = session.get(url, timeout=(20, 30))
                attempts.append({'url': url, 'http_status': response.status_code, 'status': 'browser_or_account_required',
                                 'reason': 'Author link is an HTML sharing page; no authenticated download client configured.'})
                (ROOT/'sice').mkdir(parents=True, exist_ok=True)
                (ROOT/'sice'/f'baidu_{i}.html').write_text(response.text[:60000], encoding='utf8')
                continue
            target = ROOT/'sice/downloads'/f'part{i+1}.zip'
            result = download(url, target, attempts=1)
            extract_zip(target, ROOT/'sice/extracted')
            attempts.append({'url': url, 'status': 'downloaded', **result})
        except Exception as exc:
            attempts.append({'url': url, 'status': 'unavailable', 'error_type': type(exc).__name__, 'reason': str(exc)[:500]})
    status = 'downloaded' if any(a['status'] == 'downloaded' for a in attempts) else 'not_available'
    atomic_json(ROOT/'sice/acquisition.json', {'source_page': 'https://github.com/csjcai/SICE', 'at': utc_now(),
        'status': status, 'attempts': attempts, 'evaluation_status': 'not_run' if status != 'downloaded' else 'pending_validation',
        'reference_limitation': 'Selected enhanced references are not unique camera exposure ground truth.'})


def build_manifest():
    import cv2
    cv2.setNumThreads(1)
    annotation = ROOT/'coco/annotations/instances_val2017.json'
    payload = json.loads(annotation.read_text('utf8'))
    by_image = {}
    for a in payload['annotations']:
        x, y, w, h = a['bbox']
        if w <= 0 or h <= 0: raise ValueError('Invalid COCO annotation box')
        by_image.setdefault(a['image_id'], []).append({'box': [x, y, x+w, y+h], 'class_id': a['category_id'],
                                                      'iscrowd': bool(a['iscrowd']), 'annotation_id': a['id']})
    records = []; quarantined = []
    def inspect(image):
        path = ROOT/'coco/val2017'/image['file_name']
        try:
            fp = fingerprint(path)
            if (fp['width'], fp['height']) != (image['width'], image['height']): raise ValueError('COCO dimensions differ from decoded image')
            for b in by_image.get(image['id'], []):
                x0, y0, x1, y1 = b['box']
                if not (0 <= x0 < x1 <= image['width']+.1 and 0 <= y0 < y1 <= image['height']+.1):
                    raise ValueError('COCO box outside image')
            return {'id': f"coco:{image['id']:012d}", 'image_id': image['id'], 'dataset': 'coco_val2017',
                    'path': str(path), 'scene_id': None, 'ground_truth': by_image.get(image['id'], []), **fp}
        except Exception as exc:
            return {'quarantine': image['id'], 'reason': str(exc)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for row in pool.map(inspect, payload['images']):
            if 'quarantine' in row: quarantined.append(row)
            else: records.append(row)
    # User-acquired SICE can be imported from sice/input/<scene>/images + reference.
    sice_root = ROOT/'sice/input'
    for scene in sorted(sice_root.iterdir()) if sice_root.exists() else []:
        if not scene.is_dir(): continue
        images = sorted((scene/'images').glob('*')) if (scene/'images').exists() else []
        refs = sorted(scene.glob('reference.*'))
        for path in images:
            if path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}: continue
            try:
                row = {'id': f'sice:{scene.name}:{path.name}', 'dataset': 'sice', 'scene_id': 'sice:'+scene.name,
                       'path': str(path), 'ground_truth': [], **fingerprint(path)}
                if refs:
                    row.update(reference=str(refs[0]), reference_fingerprints=[fingerprint(refs[0])])
                records.append(row)
            except Exception as exc: quarantined.append({'quarantine': str(path), 'reason': str(exc)})
    grouping = group_and_split(records); audit_split(records)
    manifest = {'schema_version': 1, 'version': 'general_v060_split1', 'seed': SEED,
        'created_at': utc_now(), 'annotation_sha256': sha256(annotation), 'source': str(annotation),
        'categories': payload['categories'], 'grouping': grouping, 'records': records,
        'quarantine': quarantined, 'coco_scope': 'Public val2017 project split, not guaranteed unseen training data.',
        'sice_status': 'available' if any(r['dataset'] == 'sice' for r in records) else 'not_run_source_unavailable'}
    path = ROOT/'manifest.json'
    if path.exists():
        old = json.loads(path.read_text('utf8'))
        # The original freeze is immutable; a different corpus requires new version.
        old_rows = [{k: v for k, v in r.items() if k not in {'path', 'reference'}} for r in old['records']]
        new_rows = [{k: v for k, v in r.items() if k not in {'path', 'reference'}} for r in records]
        if old_rows != new_rows: raise ValueError('Frozen corpus differs; use a new version directory')
        return old
    atomic_json(path, manifest)
    atomic_json(ROOT/'manifest.sha256.json', {'sha256': sha256(path), 'records': len(records), 'grouping': grouping})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--sice', action='store_true', help='Try only author-listed official download endpoints')
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if not any(vars(args).values()): parser.print_help(); return
    if args.download:
        if args.sice:
            with ThreadPoolExecutor(max_workers=2) as pool:
                a = pool.submit(download_coco); b = pool.submit(attempt_sice); a.result(); b.result()
        else: download_coco()
    elif args.sice: attempt_sice()
    if args.prepare:
        manifest = build_manifest(); print(json.dumps({'records': len(manifest['records']), 'grouping': manifest['grouping']}), flush=True)


if __name__ == '__main__': main()
