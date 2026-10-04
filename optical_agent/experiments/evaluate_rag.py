"""Reproducible local RAG experiment; no LLM or paid API calls."""
from optical_agent.paths import PROJECT_ROOT
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

from optical_agent.rag_engine import DATA_ROOT, LocalEncoder, RagRetriever, build_chunks, fingerprint
from optical_agent.rag_eval import CORPORA, freeze_files, load_corpus, run_split, validate_labels

REPO=PROJECT_ROOT
BASE_COMMIT='8d2d59212d011eb025e7a7dabe0a6e1c46e8508f'


def snapshots():
    """Reconstruct C0 from the pinned project commit, never from new reports."""
    from prepare_rag_corpus import snapshots as reconstruct
    return reconstruct()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['snapshot','prepare','freeze','dev','test'])
    parser.add_argument('--run-id',default='rag_v050_20261003')
    args=parser.parse_args()
    if not args.run_id.replace('_','').replace('-','').isalnum():
        raise ValueError('Unsafe run ID')
    run=DATA_ROOT/'runs'/args.run_id
    if args.stage=='snapshot':
        snapshots()
    elif args.stage=='prepare':
        # Same eight sources, new segmentation/scopes: isolate additions from chunking.
        root=DATA_ROOT/'c0_struct_source'
        if not root.exists():
            shutil.copytree(DATA_ROOT/'c1_source',root)
            manifest=json.loads((root/'knowledge/sources_v2.json').read_text(encoding='utf8'))[:8]
            (root/'knowledge/sources_v2.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
        chunks=build_chunks(root)
        (DATA_ROOT/'c0_struct_chunks.json').write_text(json.dumps(chunks,ensure_ascii=False,indent=2),encoding='utf8')
        encoder=LocalEncoder(device='cuda')
        for corpus in CORPORA:
            r=RagRetriever(load_corpus(corpus),'embedding',encoder=encoder,build=True,corpus_name=corpus)
            print(corpus,len(r.chunks),'index build',round(r.index_build_ms,1),'ms',fingerprint(r.chunks),flush=True)
    elif args.stage=='freeze':
        validate_labels(json.loads((REPO/'eval/rag_questions.json').read_text(encoding='utf8')),DATA_ROOT/'c1_source')
        run.mkdir(parents=True,exist_ok=False)
        frozen={'created_at':datetime.now(timezone.utc).isoformat(),'files':freeze_files(REPO),
                'protocol':{'seed':20261003,'cpu_threads':2,'rrf_constant':60,'rrf_top_each':20,
                            'latency_repeats':5,'holdout_selection':False,'paid_api_requests':0}}
        (run/'freeze.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print('Frozen; no retrieval questions executed')
    else:
        if args.stage=='test' and not (run/'selection.json').is_file():
            raise ValueError('Dev selection must be frozen before holdout')
        run_split(REPO,run,args.stage)


if __name__=='__main__':
    main()
