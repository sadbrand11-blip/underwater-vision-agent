"""Download pinned official assets into the approved D-drive RAG directory."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import requests

ROOT = Path(r'D:\CodexData\optical_agent\rag')
MODEL_ID = 'BAAI/bge-small-zh-v1.5'
REVISION = '7999e1d3359715c523056ef9478215996d62a620'
PAPERS = {
    'uieb': 'https://li-chongyi.github.io/proj_benchmark.html',
    'sea_thru': 'https://openaccess.thecvf.com/content_CVPR_2019/papers/Akkaynak_Sea-Thru_A_Method_for_Removing_Water_From_Underwater_Images_CVPR_2019_paper.pdf',
    'machine_iqa': 'https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Image_Quality_Assessment_From_Human_to_Machine_Preference_CVPR_2025_paper.pdf',
    'reed': 'https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_From_Abyssal_Darkness_to_Blinding_Glare_A_Benchmark_on_Extreme_ICCV_2025_paper.pdf',
    'rhcnet': 'https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_RHCNet_Residual-Guided_Hierarchical_Calibration_Network_for_Robust_Underwater_Object_Detection_CVPR_2026_paper.pdf',
    'enhancement_augmentation': 'https://openaccess.thecvf.com/content/WACV2026W/WVAQ/papers/Saleem_Enhancement_as_Augmentation_Improving_Detection_in_Highly_Degraded_Underwater_Images_WACVW_2026_paper.pdf',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(session, url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    record_path = target.with_suffix(target.suffix + '.download.json')
    if target.exists() and record_path.exists():
        previous = json.loads(record_path.read_text(encoding='utf8'))
        if previous.get('url') == url and previous.get('sha256') == digest(target):
            return previous
        raise ValueError(f'Asset changed: {target}')
    part = target.with_suffix(target.suffix + '.part')
    for attempt in range(3):
        offset = part.stat().st_size if part.exists() else 0
        headers = {'Range': f'bytes={offset}-'} if offset else {}
        try:
            with session.get(url, stream=True, headers=headers, timeout=(20, 90)) as response:
                if response.status_code == 416 and offset:
                    response.close()
                    with part.open('wb'):
                        pass
                    continue
                response.raise_for_status()
                if offset and response.status_code != 206:
                    offset = 0
                total = response.headers.get('Content-Range', '').split('/')[-1]
                expected = int(total) if total.isdigit() else int(response.headers.get('Content-Length', 0)) + offset
                with part.open('ab' if offset else 'wb') as output:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            output.write(block)
                if expected and part.stat().st_size != expected:
                    raise ValueError('Incomplete download')
                if target.suffix == '.pdf' and part.read_bytes()[:5] != b'%PDF-':
                    raise ValueError('Not a PDF')
                part.replace(target)
                record = {'url': url, 'bytes': target.stat().st_size, 'sha256': digest(target),
                          'http_status': response.status_code, 'downloaded_at': '2026-10-03',
                          'revision': REVISION if '/huggingface.co/' in url else None}
                record_path.write_text(json.dumps(record, indent=2), encoding='utf8')
                return record
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', choices=['all', 'model', 'papers'], default='all')
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    session.headers['User-Agent'] = 'Mozilla/5.0 (research provenance download)'
    session.headers['Accept-Encoding'] = 'identity'
    results = {}
    if args.assets in {'all', 'model'}:
        filenames = ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.txt',
                     'special_tokens_map.json', 'model.safetensors', 'README.md',
                     '1_Pooling/config.json', 'modules.json', 'sentence_bert_config.json']
        for name in filenames:
            url = f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}'
            results['model:' + name] = download(session, url, ROOT / 'models/bge-small-zh-v1.5' / name)
            print('Downloaded model asset', name, results['model:' + name]['bytes'], flush=True)
    if args.assets in {'all', 'papers'}:
        for name, url in PAPERS.items():
            target = ROOT / 'papers' / (name + ('.pdf' if url.endswith('.pdf') else '.html'))
            try:
                results[name] = download(session, url, target)
                print('Downloaded official source', name, flush=True)
            except Exception as exc:
                results[name] = {'url': url, 'status': 'unavailable', 'error_type': type(exc).__name__,
                                 'http_status': getattr(getattr(exc, 'response', None), 'status_code', None)}
                print('Official full text unavailable', name, results[name], flush=True)
    path = ROOT / ('assets_' + args.assets + '.json')
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
