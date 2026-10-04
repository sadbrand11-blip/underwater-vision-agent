"""Official-only resumable downloads and guarded extraction, exclusively on D:."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
import zipfile
import requests

ROOT = Path(r'D:\CodexData\optical_agent')
PASSWORDS = {'raw-890.rar': '1234567', 'reference-890.rar': '8901234', 'challenging-60.rar': '5678901'}
FILES = {
    'raw-890.rar': ('https://drive.usercontent.google.com/download?id=12W_kkblc2Vryb9zHQ6BfGQ_NKUfXYk13&export=download&confirm=t', 'uieb', None),
    'reference-890.rar': ('https://drive.usercontent.google.com/download?id=1cA-8CzajnVEL4feBRKdBxjEe6hwql6Z7&export=download&confirm=t', 'uieb', None),
    'challenging-60.rar': ('https://drive.usercontent.google.com/download?id=1Ew_r83nXzVk0hlkfuomWqsAIxuq6kaN4&export=download&confirm=t', 'uieb', None),
    'UIIS10K.zip': ('https://huggingface.co/datasets/LiamLian0727/UIIS10K/resolve/main/UIIS10K.zip', 'uiis10k', '14a86e9e7df715fbba2b60b2a5aef18488da0f124ef197b09a570eae981bb19c'),
    'fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth': ('https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth', 'pretrained', '907ea3f9'),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def validate_members(names):
    for name in names:
        normalized = name.replace('\\', '/')
        if (not normalized or normalized.startswith('/') or re.match(r'^[A-Za-z]:', normalized)
                or '..' in PurePosixPath(normalized).parts or '\x00' in normalized
                or any(':' in part or part.rstrip(' .') in {'', '..'} for part in PurePosixPath(normalized).parts)):
            raise ValueError('Archive member leaves extraction directory: ' + name)


def ensure_unrar():
    """Portable official tool only, no system install and all downloads remain on D:."""
    folder=ROOT/'tools'; executable=folder/'UnRAR.exe'
    if executable.exists():
        return executable
    folder.mkdir(parents=True,exist_ok=True)
    url='https://www.rarlab.com/rar/unrarw64.exe'
    session=requests.Session(); session.trust_env=False
    r=session.get(url,timeout=(30,90)); r.raise_for_status()
    package=folder/'unrarw64.exe'; package.write_bytes(r.content)
    listing=subprocess.run(['tar','-tf',str(package)],check=True,capture_output=True,text=True).stdout
    validate_members(listing.splitlines())
    subprocess.run(['tar','-xf',str(package),'-C',str(folder)],check=True)
    (folder/'unrar_source.json').write_text(json.dumps({'source':url,'size':package.stat().st_size,
        'sha256':sha(package),'checksum_scope':'local_integrity_only'},indent=2),encoding='utf8')
    if not executable.exists():
        raise RuntimeError('Official UnRAR package did not contain UnRAR.exe')
    return executable


def extract(archive, destination):
    if not destination.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Extraction destination leaves the approved D: data root')
    marker = destination / (archive.name + '.extracted.json')
    digest = sha(archive)
    if marker.exists() and json.loads(marker.read_text())['sha256'] == digest:
        return
    destination.mkdir(parents=True, exist_ok=True)
    if archive.suffix == '.zip':
        with zipfile.ZipFile(archive) as z:
            validate_members(z.namelist())
            if any((i.external_attr >> 16) & 0o170000 == 0o120000 for i in z.infolist()):
                raise ValueError('Symlink in archive')
            bad = z.testzip()
            if bad:
                raise ValueError('ZIP CRC failed: ' + bad)
            z.extractall(destination)
    else:
        listing = subprocess.run(['tar', '-tf', str(archive)], check=True, capture_output=True, text=True, encoding='utf8').stdout
        validate_members(listing.splitlines())
        verbose = subprocess.run(['tar', '-tvf', str(archive)], check=True, capture_output=True, text=True, encoding='utf8').stdout
        if any(line.startswith(('l', 'h')) for line in verbose.splitlines()):
            raise ValueError('Archive links are prohibited')
        unrar = ensure_unrar()
        password = '-p' + PASSWORDS[archive.name]
        # Verify the decrypted archive CRC before writing any image.
        subprocess.run([str(unrar), 't', '-idq', password, str(archive)], check=True)
        subprocess.run([str(unrar), 'x', '-idq', '-o+', password, str(archive), str(destination) + '/'], check=True)
    marker.write_text(json.dumps({'sha256': digest, 'crc_checked_by': 'zipfile' if archive.suffix=='.zip' else 'RARLAB_UnRAR_password_CRC', 'at': datetime.now(timezone.utc).isoformat()}), encoding='utf8')


def download(name, root):
    url, dataset, expected = FILES[name]
    folder = root / dataset
    archive = folder / 'downloads' / name
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(archive.suffix + '.part')
    session = requests.Session()
    session.trust_env = False
    if not archive.exists():
        for attempt in range(4):
            offset = partial.stat().st_size if partial.exists() else 0
            try:
                with session.get(url, stream=True, headers={'Range': f'bytes={offset}-'} if offset else {}, timeout=(30, 90)) as r:
                    if r.status_code == 416 and offset and r.headers.get('content-range') == f'bytes */{offset}':
                        partial.replace(archive)
                        break
                    r.raise_for_status()
                    if 'text/html' in r.headers.get('content-type', ''):
                        raise RuntimeError('Official download needs login/confirmation; no archive received: ' + name)
                    if r.status_code == 206:
                        if not r.headers.get('content-range', '').startswith(f'bytes {offset}-'):
                            raise ValueError('Unexpected resume range')
                    elif offset:
                        offset = 0
                    total = offset + int(r.headers.get('content-length', 0))
                    last, done = time.monotonic(), offset
                    with partial.open('ab' if offset else 'wb') as f:
                        for b in r.iter_content(1024 * 1024):
                            f.write(b)
                            done += len(b)
                            if time.monotonic() - last > 20:
                                print(json.dumps({'download': name, 'bytes': done, 'total': total}), flush=True)
                                last = time.monotonic()
                    if total and done != total:
                        raise ValueError('Incomplete transfer')
                    if partial.stat().st_size < 1000:
                        raise ValueError('Invalid tiny download')
                    partial.replace(archive)
                break
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2)
    digest = sha(archive)
    if expected and not digest.startswith(expected):
        raise ValueError('Source checksum mismatch: ' + name)
    info = {'file': name, 'source': url, 'size': archive.stat().st_size, 'sha256': digest,
            'source_checksum': expected, 'checksum_scope': 'publisher' if expected else 'local_integrity_only',
            'time': datetime.now(timezone.utc).isoformat()}
    if archive.suffix in {'.rar', '.zip'}:
        extract(archive, folder / 'data')
    (archive.parent / (name + '.manifest.json')).write_text(json.dumps(info, indent=2), encoding='utf8')
    print(json.dumps({'ready': name, 'size': info['size'], 'sha256': digest}), flush=True)
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--files', nargs='+', choices=list(FILES), default=list(FILES))
    a = p.parse_args()
    if a.root.resolve().drive.upper() != 'D:' or not a.root.resolve().is_relative_to(ROOT.resolve()):
        p.error('Downloads, extraction and cache must remain on D:')
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda name: download(name, a.root), a.files))


if __name__ == '__main__':
    main()
