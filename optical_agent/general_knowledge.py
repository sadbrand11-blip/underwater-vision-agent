"""An independent natural-image corpus; never reads the frozen C0/C1 sources."""
from pathlib import Path

from optical_agent.rag_engine import RagRetriever, RagUnavailable, build_chunks, fingerprint

CORPUS_VERSION = 'general-natural-v1'
GENERAL_RAG_ROOT = Path(r'D:\CodexData\optical_agent\rag\general_rgb_v1')


def general_release_config(root=None):
    root = Path(root or Path(__file__).resolve().parents[1])
    config = {'profile_id': 'general_rgb_v1', 'corpus': CORPUS_VERSION,
              'mode': 'tfidf', 'thresholds': {'tfidf': .10, 'dense': .75},
              'acceptance_status': 'not_formally_evaluated',
              'corpus_hash': fingerprint(build_chunks(root, 'knowledge/GENERAL_SOURCES.json'))}
    return dict(config, release_config_hash=fingerprint(config))


class GeneralKnowledgeRetriever(RagRetriever):
    """KnowledgeRetriever.search-compatible local retrieval with explicit versions."""
    def __init__(self, root=None, chunk_size=400, overlap=80, min_score=.10, *,
                 mode='tfidf', encoder=None, build=False, cache_root=None):
        if chunk_size != 400 or overlap != 80:
            raise ValueError('通用语料使用固定400字符、最多80重叠分段；修改须建立新版本')
        root = Path(root or Path(__file__).resolve().parents[1])
        chunks = build_chunks(root, 'knowledge/GENERAL_SOURCES.json')
        if any(c.get('vision_modes') != ['general_rgb_v1'] for c in chunks):
            raise ValueError('通用知识清单含非通用资料，拒绝混用历史水下证据')
        super().__init__(chunks, mode, encoder=encoder, build=build,
                         cache_root=cache_root or GENERAL_RAG_ROOT / 'indexes',
                         thresholds={'tfidf': min_score, 'dense': .75},
                         vision_mode=None, corpus_name=CORPUS_VERSION)
        self.root, self.vision_mode = root, 'general_rgb_v1'
        self.release_config = general_release_config(root)
        if self.release_config['corpus_hash'] != self.corpus_hash:
            raise RagUnavailable('通用知识在创建过程中发生变化，请重新创建会话')

    def metadata(self):
        return dict(super().metadata(), profile_id='general_rgb_v1',
                    corpus_version=CORPUS_VERSION, frozen_historical_scores_applicable=False,
                    acceptance_status='not_formally_evaluated', fallback_reason=None,
                    release_config_hash=self.release_config['release_config_hash'])


def create_general_retriever(root=None, mode=None, vision_mode='general_rgb_v1', *,
                             encoder=None, allow_fallback=False, build=False, cache_root=None):
    if vision_mode != 'general_rgb_v1':
        raise ValueError('通用知识只适用于general_rgb_v1，旧水下资料请使用归档检索器')
    # An explicit legacy selection retains sparse behavior but does not import C0.
    selected = 'tfidf' if mode in {None, 'legacy'} else mode
    if selected not in {'tfidf', 'embedding', 'hybrid'}:
        raise ValueError('rag_mode须为legacy、tfidf、embedding或hybrid')
    try:
        retriever = GeneralKnowledgeRetriever(root, mode=selected, encoder=encoder,
                                             build=build, cache_root=cache_root)
    except (RagUnavailable, ImportError) as exc:
        if not allow_fallback or mode is not None:
            raise RagUnavailable('通用Embedding/Hybrid不可用；需为当前独立语料准备本地模型和索引：' + str(exc)) from exc
        retriever = GeneralKnowledgeRetriever(root)
        original, reason = retriever.metadata, str(exc)
        retriever.metadata = lambda: dict(original(), requested_mode=selected, fallback_reason=reason)
    if mode == 'legacy':
        original = retriever.metadata
        retriever.metadata = lambda: dict(original(), requested_mode='legacy',
            mode_alias_notice='通用配置的legacy选项使用独立TF-IDF资料；未加载旧C0语料')
    return retriever
