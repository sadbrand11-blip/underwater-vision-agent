"""Local dense/sparse retrieval with immutable corpus identities and explicit scores."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import re
from threading import RLock
from time import perf_counter
import unicodedata

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from optical_agent.public_config import PUBLIC_ROOT
DATA_ROOT = PUBLIC_ROOT / 'rag'
MODEL_REVISION = '7999e1d3359715c523056ef9478215996d62a620'
MODES = {'legacy', 'tfidf', 'embedding', 'hybrid'}
QUERY_INSTRUCTION = '为这个句子生成表示以用于检索相关文章：'


class RagUnavailable(RuntimeError):
    """Unavailable must not be reported as no relevant knowledge."""


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf8')).hexdigest()


def split_block(text, maximum=400, overlap=80):
    """Return contiguous source spans; tables break only between whole rows."""
    if maximum < 1 or not 0 <= overlap < maximum:
        raise ValueError('Invalid chunk limits')
    table = text.lstrip().startswith('|')
    start = 0
    while start < len(text):
        end = min(start + maximum, len(text))
        if end < len(text):
            if table:
                cut = text.rfind('\n', start, end + 1)
                if cut <= start:
                    raise ValueError('Table row exceeds chunk size; preserve row rather than truncate')
                end = cut + 1
            else:
                boundaries = list(re.finditer(r'[。！？.!?；;]\s*|\n', text[start:end]))
                if boundaries and boundaries[-1].end() >= maximum // 2:
                    end = start + boundaries[-1].end()
        yield start, end
        if end == len(text):
            break
        start = end if table else max(start + 1, end - overlap)


def build_chunks(root, manifest_path='knowledge/sources_v2.json'):
    root = Path(root)
    manifest = json.loads((root / manifest_path).read_text(encoding='utf8'))
    chunks, seen = [], set()
    for source in manifest:
        path = (root / source['path']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Knowledge path leaves corpus root')
        text = path.read_text(encoding='utf8')
        heading = ''
        for match in re.finditer(r'\S[\s\S]*?(?=\n\s*\n|\Z)', text):
            block = match.group()
            if re.fullmatch(r'#{1,6}\s+[^\n]+\s*', block):
                heading = block.strip().lstrip('#').strip()
                continue
            for start, end in split_block(block):
                body = block[start:end].strip()
                if not body:
                    continue
                identity = (source['id'], body)
                if identity in seen:
                    continue
                seen.add(identity)
                leading = len(block[start:end]) - len(block[start:end].lstrip())
                offset = match.start() + start + leading
                chunks.append({
                    'citation_id': source['id'] + ':' + fingerprint(body)[:16],
                    'source_id': source['id'], 'title': source['title'], 'heading': heading,
                    'source': source['url'], 'path': source['path'], 'verified_at': source['verified_at'],
                    'source_type': source['source_type'], 'vision_modes': source['vision_modes'],
                    'verification_level': source.get('verification_level', 'project_files'),
                    'source_start': offset, 'source_end': offset + len(body),
                    'source_sha256': file_hash(path), 'text': body})
    if not chunks:
        raise ValueError('Empty knowledge corpus')
    return chunks


class LocalEncoder:
    def __init__(self, model_path=None, device='cpu'):
        self.path = Path(model_path or DATA_ROOT / 'models/bge-small-zh-v1.5')
        weights = self.path / 'model.safetensors'
        if not weights.is_file():
            raise RagUnavailable('本地Embedding模型缺失；请先运行 prepare_rag.py，网页不会自动下载。')
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RagUnavailable(
                'Local Embedding dependencies unavailable; install the optional Torch/Transformers dependencies. '
                'TF-IDF remains available.'
            ) from exc
        self.model_hash = file_hash(weights)
        self.asset_hash = fingerprint({p.name:file_hash(p) for p in sorted(self.path.iterdir())
            if p.is_file() and p.suffix in {'.json','.txt','.safetensors'}})
        self.device = device
        self.lock = RLock()
        # Query inference remains CPU even when corpus preparation used CUDA.
        if device == 'cpu':
            torch.set_num_threads(2)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True, trust_remote_code=False)
            self.model = AutoModel.from_pretrained(self.path, local_files_only=True,
                                                  trust_remote_code=False, use_safetensors=True).to(device).eval()
        except Exception as exc:
            raise RagUnavailable('本地Embedding模型无法加载：' + type(exc).__name__) from exc

    def encode(self, texts, query=False):
        import torch
        texts = [QUERY_INSTRUCTION + text if query else text for text in texts]
        vectors = []
        with self.lock, torch.inference_mode():
            for start in range(0, len(texts), 16):
                batch = texts[start:start + 16]
                if any(len(self.tokenizer.encode(t, add_special_tokens=True)) > 512 for t in batch):
                    raise ValueError('Embedding输入超过512 tokens；未截断证据，请缩短问题或修订分段。')
                tokens = self.tokenizer(batch, padding=True, truncation=False, return_tensors='pt').to(self.device)
                hidden = self.model(**tokens).last_hidden_state[:, 0]
                result = torch.nn.functional.normalize(hidden.float(), p=2, dim=1)
                vectors.append(result.cpu().numpy())
        return np.concatenate(vectors) if vectors else np.empty((0, 512), dtype=np.float32)


class RagRetriever:
    def __init__(self, chunks, mode='tfidf', *, encoder=None, cache_root=None, build=False,
                 thresholds=None, vision_mode=None, corpus_name='C1'):
        if mode not in {'tfidf', 'embedding', 'hybrid'}:
            raise ValueError('Unknown RAG mode')
        if vision_mode is not None and vision_mode not in {'legacy', 'experiment_control', 'experiment_candidate'}:
            raise ValueError('Unknown vision context')
        if len({x['citation_id'] for x in chunks}) != len(chunks):
            raise ValueError('Duplicate citation IDs')
        self.chunks = json.loads(json.dumps(chunks, ensure_ascii=False))
        self.mode, self.vision_mode, self.corpus_name = mode, vision_mode, corpus_name
        self.corpus_hash = fingerprint(self.chunks)
        self.thresholds = dict(thresholds or {'tfidf': .1, 'dense': .75})
        self.min_score = self.thresholds['tfidf']
        self.encoder, self.vectors = encoder, None
        self.query_cache = OrderedDict()
        self.lock = RLock()
        texts = [c['title'] + '\n' + c['text'] for c in self.chunks]
        started = perf_counter()
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(texts)
        if mode in {'embedding', 'hybrid'}:
            self.encoder = encoder or LocalEncoder()
            folder = Path(cache_root or DATA_ROOT / 'indexes') / self.corpus_hash / self.encoder.asset_hash
            array_path, meta_path = folder / 'vectors.npy', folder / 'index.json'
            if array_path.exists() or meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding='utf8'))
                    if meta['corpus_hash'] != self.corpus_hash or meta['asset_hash'] != self.encoder.asset_hash:
                        raise ValueError('Index fingerprint mismatch')
                    if meta['vectors_sha256'] != file_hash(array_path):
                        raise ValueError('Corrupt vector index')
                    self.vectors = np.load(array_path, allow_pickle=False)
                    if self.vectors.shape != (len(chunks), 512) or not np.isfinite(self.vectors).all():
                        raise ValueError('Invalid vector matrix')
                    self.index_build_ms = meta['index_build_ms']
                except Exception as exc:
                    raise RagUnavailable('Embedding索引损坏或版本不符；需显式重建，不能解释为无知识。') from exc
            elif build:
                self.vectors = self.encoder.encode(texts)
                folder.mkdir(parents=True, exist_ok=True)
                np.save(array_path, self.vectors, allow_pickle=False)
                self.index_build_ms = (perf_counter() - started) * 1000
                meta_path.write_text(json.dumps({'corpus_hash': self.corpus_hash,
                    'model_sha256': self.encoder.model_hash, 'model_revision': MODEL_REVISION,
                    'asset_hash': self.encoder.asset_hash,
                    'vectors_sha256': file_hash(array_path), 'count': len(chunks), 'dimension': 512,
                    'index_build_ms': self.index_build_ms, 'build_device': self.encoder.device}, indent=2), encoding='utf8')
            else:
                raise RagUnavailable('Embedding索引尚未准备；请先运行 evaluate_rag.py prepare。')
        else:
            self.index_build_ms = (perf_counter() - started) * 1000

    def metadata(self):
        return {'mode': self.mode, 'corpus': self.corpus_name, 'corpus_hash': self.corpus_hash,
                'model_revision': MODEL_REVISION if self.encoder else None,
                'model_sha256': self.encoder.model_hash if self.encoder else None,
                'asset_hash': self.encoder.asset_hash if self.encoder else None,
                'vision_mode': self.vision_mode, 'thresholds': self.thresholds,
                'score_is_probability': False, 'query_device': self.encoder.device if self.encoder else 'cpu'}

    def scores(self, query):
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError('检索问题须为1—2000字符的文本。')
        normalized = query if self.corpus_name=='C0' else unicodedata.normalize('NFKC', query).strip()
        with self.lock:
            if normalized in self.query_cache:
                self.query_cache.move_to_end(normalized)
                return self.query_cache[normalized]
            sparse = (self.matrix @ self.vectorizer.transform([normalized]).T).toarray().ravel()
            dense = self.vectors @ self.encoder.encode([normalized], query=True)[0] if self.vectors is not None else None
            self.query_cache[normalized] = (sparse, dense)
            if len(self.query_cache) > 128:
                self.query_cache.popitem(last=False)
            return sparse, dense

    def rank(self, query, top_k=3, *, filtered=True):
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError('top_k须为1—20的整数。')
        sparse, dense = self.scores(query)
        eligible = [i for i, c in enumerate(self.chunks)
                    if self.vision_mode is None or self.vision_mode in c.get('vision_modes', ['legacy', 'experiment_control', 'experiment_candidate'])]
        sparse_rank = sorted(eligible, key=lambda i: (-float(sparse[i]), i))
        dense_rank = sorted(eligible, key=lambda i: (-float(dense[i]), i)) if dense is not None else []
        if self.mode == 'hybrid':
            fusion = {}
            for ranking in (sparse_rank[:20], dense_rank[:20]):
                for rank, i in enumerate(ranking, 1):
                    fusion[i] = fusion.get(i, 0.) + 1. / (60 + rank)
            ranked = sorted(fusion, key=lambda i: (-fusion[i], i))
        else:
            fusion = {}
            ranked = sparse_rank if self.mode == 'tfidf' else dense_rank
        def allowed(i):
            if not filtered:
                return True
            lexical = sparse[i] >= self.thresholds['tfidf']
            semantic = dense is not None and dense[i] >= self.thresholds['dense']
            return lexical if self.mode == 'tfidf' else semantic if self.mode == 'embedding' else lexical or semantic
        result = []
        for i in ranked:
            if not allowed(i):
                continue
            score = sparse[i] if self.mode == 'tfidf' else dense[i] if self.mode == 'embedding' else fusion[i]
            result.append(dict(self.chunks[i], score=round(float(score), 6),
                tfidf_score=round(float(sparse[i]), 6),
                embedding_score=round(float(dense[i]), 6) if dense is not None else None,
                score_kind={'tfidf': 'tfidf_cosine', 'embedding': 'embedding_cosine', 'hybrid': 'rrf_rank_fusion'}[self.mode],
                corpus_hash=self.corpus_hash, score_is_probability=False))
            if len(result) == top_k:
                break
        return result

    def search(self, query, top_k=3):
        return self.rank(query, top_k, filtered=True)


def release_config(root):
    path = Path(root) / 'knowledge/rag_release.json'
    return json.loads(path.read_text(encoding='utf8')) if path.exists() else {'mode': 'legacy', 'thresholds': {'tfidf': .1, 'dense': .75}}


def create_retriever(root, mode=None, vision_mode='legacy', *, encoder=None, allow_fallback=False):
    from optical_agent.rag import KnowledgeRetriever
    root = Path(root)
    config = release_config(root)
    selected = mode or config['mode']
    if selected not in MODES:
        raise ValueError('rag_mode须为legacy、tfidf、embedding或hybrid。')
    if selected == 'legacy':
        retriever = KnowledgeRetriever(root)
        retriever.metadata = lambda: {'mode': 'legacy', 'corpus': 'public_live',
            'corpus_hash': fingerprint(retriever.chunks), 'score_is_probability': False,
            'vision_mode': vision_mode, 'fallback_reason': None}
        return retriever
    source_root = root
    chunks = build_chunks(source_root)
    if config.get('corpus_hash') and config['corpus_hash'] != fingerprint(chunks):
        raise RagUnavailable('发布配置与知识库快照不一致；请创建新配置，不复用旧会话。')
    try:
        return RagRetriever(chunks, selected, encoder=encoder, vision_mode=vision_mode,
                            thresholds=config.get('profiles', {}).get(selected,config['thresholds']),
                            corpus_name=config.get('corpus', 'C1'))
    except RagUnavailable as exc:
        if not allow_fallback or mode is not None:
            raise
        retriever = RagRetriever(chunks, 'tfidf', vision_mode=vision_mode,
                                thresholds=config.get('profiles',{}).get('tfidf',config['thresholds']),
                                corpus_name=config.get('corpus', 'C1'))
        original = retriever.metadata
        retriever.metadata = lambda: dict(original(), requested_mode=selected, fallback_reason=str(exc))
        return retriever
