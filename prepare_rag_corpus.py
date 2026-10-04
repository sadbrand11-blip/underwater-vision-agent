"""Byte-reproducible corpus reconstruction, independent of Git checkout newlines."""
import json
from pathlib import Path
import shutil
import subprocess

from optical_agent.rag import KnowledgeRetriever
from optical_agent.rag_engine import DATA_ROOT, build_chunks, file_hash

REPO=Path(__file__).parent


def render(raw, spec):
    text=raw.decode('utf8').replace('\r\n','\n')
    if 'line_endings' in spec:
        lines=text.splitlines(keepends=True)
        if len(lines)!=len(spec['line_endings']):raise ValueError('Source line count changed')
        text=''.join(line.removesuffix('\n')+ending for line,ending in zip(lines,spec['line_endings']))
    elif spec['newline']=='CRLF':text=text.replace('\n','\r\n')
    data=text.encode('utf8')
    import hashlib
    if hashlib.sha256(data).hexdigest()!=spec['sha256']:
        raise ValueError('Source differs from release corpus; stop rather than silently re-index')
    return data


def snapshots():
    layout=json.loads((REPO/'knowledge/corpus_layout.json').read_text(encoding='utf8'))
    if (DATA_ROOT/'c0_source').exists() or (DATA_ROOT/'c1_source').exists():
        raise ValueError('Frozen snapshots already exist; cannot overwrite')
    pending={}
    for path,spec in layout['c0'].items():
        raw=subprocess.run(['git','show',layout['base_commit']+':'+path],cwd=REPO,
                           capture_output=True,check=True).stdout
        pending[path]=render(raw,spec)
    # Validate everything before writing the immutable D-drive snapshots.
    cards={path:render((REPO/path).read_bytes(),spec) for path,spec in layout['cards'].items()}
    manifest=render((REPO/'knowledge/sources_v2.json').read_bytes(),layout['manifest'])
    old=DATA_ROOT/'c0_source'
    for path,data in pending.items():
        target=old/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    (DATA_ROOT/'c0_chunks.json').write_text(json.dumps(KnowledgeRetriever(old).chunks,ensure_ascii=False,indent=2),encoding='utf8')
    new=DATA_ROOT/'c1_source';shutil.copytree(old,new)
    (new/'knowledge/sources_v2.json').write_bytes(manifest)
    for path,data in cards.items():
        target=new/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    (DATA_ROOT/'c1_chunks.json').write_text(json.dumps(build_chunks(new),ensure_ascii=False,indent=2),encoding='utf8')
    print('Reconstructed release corpus with verified bytes')


if __name__=='__main__':snapshots()
