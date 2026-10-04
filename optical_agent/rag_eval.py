"""Independent retrieval scoring; gold never enters an index or encoder."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from optical_agent.paths import code_files
from time import perf_counter

import numpy as np

from .rag_engine import DATA_ROOT, LocalEncoder, RagRetriever, file_hash, fingerprint

KS = (1, 3, 5, 10)
MODES = ('tfidf', 'embedding', 'hybrid')
CORPORA = ('C0', 'C0_struct', 'C1')


def validate_labels(dataset, source_root):
    questions, evidence = dataset['questions'], dataset['evidence']
    if len(questions) != 100 or len({q['id'] for q in questions}) != 100:
        raise ValueError('Question count / unique IDs invalid')
    counts = Counter((q['split'], q['kind']) for q in questions)
    expected = {(s,k):n for s in ('dev','test') for k,n in
                (('existing',30),('literature',10),('unanswerable',10))}
    if dict(counts) != expected:
        raise ValueError('Split denominators differ from protocol')
    groups, requirements = defaultdict(set), defaultdict(set)
    for q in questions:
        groups[q['fact_group']].add(q['split'])
        if (q['kind'] == 'unanswerable') != (not q['required_groups']):
            raise ValueError('Unanswerable evidence mismatch')
        for group in q['required_groups']:
            requirements[group].add(q['split'])
            if group not in evidence:
                raise ValueError('Missing evidence group')
    if any(len(v)>1 for v in list(groups.values())+list(requirements.values())):
        raise ValueError('Fact or required evidence leaked between splits')
    root = Path(source_root).resolve()
    for group in evidence.values():
        for alt in group['alternatives']:
            path = (root / alt['path']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Evidence source outside corpus')
            text = path.read_text(encoding='utf8')
            if text[alt['start']:alt['end']] != alt['quote']:
                raise ValueError('Gold quote offset invalid')
    return True


def chunk_groups(chunk, evidence):
    sid = chunk.get('source_id', chunk['citation_id'].split(':')[0])
    return sorted(group for group, spec in evidence.items()
                  if any(alt['source_id']==sid and alt['quote'] in chunk['text']
                         for alt in spec['alternatives']))


def score_question(question, ranked, evidence):
    required = set(question['required_groups'])
    seen, recall, full, reciprocal = set(), {}, {}, 0.
    for rank, chunk in enumerate(ranked[:10], 1):
        hits = set(chunk_groups(chunk, evidence)) & required
        if hits and not reciprocal:
            reciprocal = 1. / rank
        seen.update(hits)
        if rank in KS:
            recall[str(rank)] = len(seen)/len(required) if required else None
            full[str(rank)] = bool(required and required <= seen)
    for k in KS:
        recall.setdefault(str(k), len(seen)/len(required) if required else None)
        full.setdefault(str(k), bool(required and required <= seen))
    return {'recall':recall, 'mrr10':reciprocal if required else None,
            'full':full, 'missing_at10':sorted(required-seen),
            'false_return':bool(ranked) if not required else None}


def summarize(records, field='delivered'):
    answerable = [r[field] for r in records if r['kind'] != 'unanswerable']
    negative = [r[field] for r in records if r['kind'] == 'unanswerable']
    return {'answerable_n':len(answerable), 'negative_n':len(negative),
            'recall':{str(k):float(np.mean([s['recall'][str(k)] for s in answerable]))
                       if answerable else None for k in KS},
            'mrr10':float(np.mean([s['mrr10'] for s in answerable])) if answerable else None,
            'full_hit':{str(k):float(np.mean([s['full'][str(k)] for s in answerable]))
                        if answerable else None for k in KS},
            'no_answer_false_return':float(np.mean([s['false_return'] for s in negative])) if negative else None}


def paired_interval(left, right, metric='recall3', seed=20261003, repetitions=2000):
    """Resample fact groups, preserving both phrasings and cross-source units."""
    a,b = {r['id']:r for r in left}, {r['id']:r for r in right}
    if a.keys() != b.keys():
        raise ValueError('Paired comparisons require identical question IDs')
    grouped = defaultdict(list)
    for qid in a:
        if a[qid]['kind']=='unanswerable':
            continue
        def value(r):
            return r['delivered']['recall']['3'] if metric=='recall3' else r['delivered']['mrr10']
        grouped[a[qid]['fact_group']].append(value(b[qid])-value(a[qid]))
    values = list(grouped.values())
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(repetitions):
        sample = [v for i in rng.integers(0,len(values),len(values)) for v in values[i]]
        boot.append(float(np.mean(sample)))
    return {'metric':metric,'delta':float(np.mean([v for g in values for v in g])),
            'ci95':[float(x) for x in np.percentile(boot,[2.5,97.5])],
            'fact_groups':len(values),'resamples':repetitions,'seed':seed}


def load_corpus(name):
    return json.loads((DATA_ROOT/(name.lower()+'_chunks.json')).read_text(encoding='utf8'))


def cases_for(dataset, split, corpus):
    return [q for q in dataset['questions'] if q['split']==split and
            (corpus=='C1' or q['kind']!='literature')]


def evaluate_questions(retriever, cases, dataset):
    records=[]
    for q in cases:
        # C0 retains original unscoped behavior. C0_struct/C1 share metadata filtering.
        retriever.vision_mode = q['vision_mode'] if retriever.corpus_name!='C0' else None
        raw = retriever.rank(q['query'],20,filtered=False)
        delivered = retriever.rank(q['query'],20,filtered=True)
        records.append(dict(q, raw=score_question(q,raw,dataset['evidence']),
            delivered=score_question(q,delivered,dataset['evidence']),
            raw_ranking=raw, delivered_ranking=delivered,
            error_citations=[], failure_class=('no_answer_returned' if not q['required_groups'] and delivered
                else 'missing_evidence' if q['required_groups'] and not score_question(q,delivered,dataset['evidence'])['full']['3']
                else None)))
    return records


def select_thresholds(retriever, cases, dataset):
    """Only dev labels used. Reject-all is available, so feasibility is explicit."""
    lexical = [.05,.1,.15,.2,.25,.3,.4,.5,1.1]
    semantic = [.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1.1]
    candidates = [(a,b) for a in lexical for b in semantic]
    if retriever.mode=='tfidf':
        candidates=[(a,.75) for a in lexical]
    if retriever.mode=='embedding':
        candidates=[(.1,b) for b in semantic]
    if retriever.corpus_name=='C0' and retriever.mode=='tfidf':
        candidates=[(.1,.75)]  # exact historical control, never re-calibrate it.
    best, tried=None,[]
    for a,b in candidates:
        retriever.thresholds={'tfidf':a,'dense':b}
        records=evaluate_questions(retriever,cases,dataset)
        s=summarize(records)
        feasible=s['no_answer_false_return']<=.1
        tried.append({'thresholds':dict(retriever.thresholds),'metrics':s,'feasible':feasible})
        if feasible and (best is None or (s['recall']['3'],s['mrr10'],-a,-b)>best[0]):
            best=((s['recall']['3'],s['mrr10'],-a,-b),dict(retriever.thresholds))
    # Historical control may violate rejection constraint; don't alter that baseline.
    chosen = best[1] if best else {'tfidf':.1,'dense':.75}
    retriever.thresholds=chosen
    return chosen,tried


def latency(retriever, cases, repeats=5):
    retriever.rank(cases[0]['query'])  # warm model, not timed.
    uncached,cached=[],[]
    for _ in range(repeats):
        for q in cases:
            retriever.vision_mode=q['vision_mode'] if retriever.corpus_name!='C0' else None
            retriever.query_cache.clear()  # include encoding, don't time only a dict lookup.
            started=perf_counter(); retriever.rank(q['query']); uncached.append((perf_counter()-started)*1000)
            started=perf_counter(); retriever.rank(q['query']); cached.append((perf_counter()-started)*1000)
    return {'cpu_warm_mean_ms':float(np.mean(uncached)), 'cpu_warm_p95_ms':float(np.percentile(uncached,95)),
            'cached_mean_ms':float(np.mean(cached)), 'cached_p95_ms':float(np.percentile(cached,95)),
            'timed_queries':len(uncached),'repetitions':repeats,'independent_questions':len(cases),
            'includes_query_encoding':True,'excludes_model_load':True}


def freeze_files(repo):
    paths=[repo/'optical_agent/experiments/evaluate_rag.py',repo/'optical_agent/experiments/prepare_rag.py',repo/'optical_agent/experiments/prepare_rag_corpus.py',repo/'optical_agent/experiments/publish_rag.py',
           repo/'knowledge/corpus_layout.json',repo/'optical_agent/experiments/build_rag_questions.py',repo/'optical_agent/rag.py',
           repo/'optical_agent/rag_engine.py',repo/'optical_agent/rag_eval.py',repo/'eval/rag_questions.json']
    paths.extend(code_files(repo))
    for name in CORPORA:
        paths.append(DATA_ROOT/(name.lower()+'_chunks.json'))
        paths.extend(p for p in (DATA_ROOT/(name.lower()+'_source')).rglob('*') if p.is_file())
    paths.extend(p for p in (DATA_ROOT/'models/bge-small-zh-v1.5').rglob('*') if p.is_file())
    paths.extend(p for p in (DATA_ROOT/'papers').rglob('*') if p.is_file())
    paths.extend(DATA_ROOT/p for p in ('assets_all.json','assets_papers.json','model_metadata.json'))
    paths.extend(p for p in (DATA_ROOT/'indexes').rglob('*') if p.name in {'index.json','vectors.npy'})
    return {str(p):file_hash(p) for p in sorted(set(paths))}


def verify_freeze(repo, run, archived=False):
    frozen=json.loads((run/'freeze.json').read_text(encoding='utf8'))
    if archived:
        for name,digest in frozen['files'].items():
            path=Path(name)
            if path.is_relative_to(repo):path=run/'frozen_code'/path.relative_to(repo)
            if file_hash(path)!=digest:raise ValueError('Archived evaluated inputs changed')
    elif frozen['files'] != freeze_files(repo):
        raise ValueError('Frozen source, label, corpus or model changed; stop this run')
    selection=run/'selection.json'
    if selection.exists() and frozen.get('selection_sha256')!=file_hash(selection):
        raise ValueError('Frozen dev selection changed')


def adoption_gate(candidate, reference, latency_ms):
    return {'recall3_gain':candidate['recall']['3']-reference['recall']['3']>=.05-1e-12,
            'mrr_not_lower':candidate['mrr10']>=reference['mrr10']-1e-12,
            'no_answer':candidate['no_answer_false_return']<=.1,
            'cpu_p95':latency_ms<=500}


def run_split(repo, run, split):
    import psutil
    verify_freeze(repo,run)
    dataset=json.loads((repo/'eval/rag_questions.json').read_text(encoding='utf8'))
    validate_labels(dataset,DATA_ROOT/'c1_source')
    output=run/(split+'.json')
    if output.exists():
        raise ValueError('Existing split results cannot be overwritten')
    encoder=LocalEncoder(device='cpu')
    process=psutil.Process()
    results, selections={},{}
    frozen_selection=json.loads((run/'selection.json').read_text(encoding='utf8')) if split=='test' else None
    for corpus in CORPORA:
        cases=cases_for(dataset,split,corpus)
        for mode in MODES:
            key=corpus+'/'+mode
            started=perf_counter()
            retriever=RagRetriever(load_corpus(corpus),mode,encoder=encoder if mode!='tfidf' else None,
                                   corpus_name=corpus)
            load_ms=(perf_counter()-started)*1000
            if split=='dev':
                chosen,grid=select_thresholds(retriever,cases,dataset)
                selections[key]={'thresholds':chosen,'grid':grid}
            else:
                retriever.thresholds=frozen_selection['profiles'][key]['thresholds']
            records=evaluate_questions(retriever,cases,dataset)
            speed=latency(retriever,cases)
            results[key]={'delivered':summarize(records),'raw':summarize(records,'raw'),
                'old_questions':summarize([r for r in records if r['kind']!='literature']),
                'literature':summarize([r for r in records if r['kind']=='literature']),
                'latency':speed,'records':records,'metadata':retriever.metadata(),
                'index_build_ms':retriever.index_build_ms,
                'load_ms':load_ms,
                'process_rss_mb':process.memory_info().rss/1024**2,
                'matrix_bytes':int(retriever.matrix.data.nbytes+retriever.matrix.indices.nbytes+retriever.matrix.indptr.nbytes),
                'vector_bytes':int(retriever.vectors.nbytes) if retriever.vectors is not None else 0}
            (run/(split+'_progress.json')).write_text(json.dumps({'complete':list(results)},indent=2),encoding='utf8')
            print(key, 'R3',round(results[key]['delivered']['recall']['3'],4),
                  'MRR',round(results[key]['delivered']['mrr10'],4),
                  'false',results[key]['delivered']['no_answer_false_return'],flush=True)
    payload={'split':split,'results':results,'http_requests':0,'paid_api_requests':0,
             'freeze_sha256':file_hash(run/'freeze.json')}
    output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    if split=='dev':
        eligible=[m for m in MODES if results['C1/'+m]['delivered']['no_answer_false_return']<=.1]
        candidate=max(eligible,key=lambda m:(results['C1/'+m]['delivered']['recall']['3'],
            results['C1/'+m]['delivered']['mrr10'],-results['C1/'+m]['latency']['cpu_warm_p95_ms']))
        selection={'profiles':selections,'candidate':candidate,'selected_using':'dev_only'}
        (run/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        frozen=json.loads((run/'freeze.json').read_text(encoding='utf8'))
        frozen['selection_sha256']=file_hash(run/'selection.json')
        (run/'freeze.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return payload
