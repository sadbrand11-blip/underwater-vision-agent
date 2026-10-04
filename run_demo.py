"""No-key exposure demo; explicitly opt into the published CPU detector."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import requests

ROOT = Path(__file__).resolve().parent


def download_detector(destination):
    manifest = json.loads((ROOT / 'models/release.json').read_text(encoding='utf8'))
    destination = Path(destination)
    def checksum(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for part in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(part)
        return h.hexdigest()
    if destination.is_file() and checksum(destination) == manifest['sha256']:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix('.part')
    try:
        with requests.get(manifest['url'], stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with partial.open('wb') as stream:
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
        if partial.stat().st_size != manifest['size_bytes'] or checksum(partial) != manifest['sha256']:
            raise RuntimeError('Detector download failed size/SHA256 verification; it will not be loaded.')
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-detector', action='store_true', help='Download verified SODD weights; requires optional torch/torchvision.')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    from optical_agent.public_config import MODEL_PATH
    if args.with_detector:
        try:
            import torch
            import torchvision
        except ImportError:
            parser.error('Install requirements-detector.txt first. Exposure mode needs no Torch.')
        download_detector(MODEL_PATH)
    import app as web
    if args.with_detector:
        detector = web.get_agent().detector
        if getattr(detector, 'model_available', True) is False:
            raise RuntimeError('The detector is unavailable; real mode cannot start.')
    print(f'Open http://127.0.0.1:{args.port}/showcase\nOffline scripted scheduler; actual local visual tools. Cloud is opt-in.')
    web.app.run(host='127.0.0.1', port=args.port, debug=False)


if __name__ == '__main__':
    main()
