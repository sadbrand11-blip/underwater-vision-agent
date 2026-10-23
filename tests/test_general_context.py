"""General profile semantics and namespace isolation; no cloud or real-model claims."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from optical_agent.adaptive_goals import goal_system, parse_goal
from optical_agent.adaptive_prompts import ACTION_SYSTEM, action_system
from optical_agent.general_catalog import COCO_CLASSES, ONTOLOGY_VERSION
from optical_agent.general_knowledge import GeneralKnowledgeRetriever, create_general_retriever
from optical_agent.memory import GeneralMemoryStore, MemoryStore, MemoryTurn, versions_for
from optical_agent.rag_engine import RagUnavailable
from optical_agent.runtime import dispatch_system
from optical_agent.tasks import INTENT_SYSTEM, intent_system, scripted_intent
from optical_agent.state import Session
from optical_agent.tools import ToolContext


class GeneralFixture:
    profile_id = 'general_rgb_v1'
    classes = ('background', *COCO_CLASSES)
    ontology_version = ONTOLOGY_VERSION
    preprocessing_version = 'exif_rgb_uint8_v1'
    threshold = .5
    calibration, calibration_status = None, 'uncalibrated'
    model_sha256 = 'fixture-model-not-a-real-weight'

    def predict(self, image):
        return []


def state():
    return SimpleNamespace(detector=GeneralFixture(), last_candidate=None,
                           images={'original': np.full((32, 32, 3), 100, np.uint8)})


def goal(targets, **changes):
    value = {'task_type': 'detection', 'image_reference': 'original',
             'target_classes': targets, 'correction_policy': 'never',
             'require_reliability': False, 'question': ''}
    value.update(changes)
    return json.dumps(value, ensure_ascii=False)


def test_general_aliases_and_unknowns_cannot_be_silently_replaced():
    parsed = parse_goal(goal(['行人', '猫', 'CAR']), state(), 'original')
    assert parsed.target_classes == ('person', 'car', 'cat')
    for unsupported in ('管道', '二维码', 'pipe', 'qr_codes', '水下机器人', 'fish'):
        parsed = parse_goal(goal(['person', unsupported]), state(), 'original')
        assert parsed.disposition == 'unsupported'
        assert 'COCO 80' in parsed.question


def test_general_all_classes_and_explicit_scope_override_preference():
    prefs = {'enabled': True, 'target_classes': ['dog'], 'display_detail': 'compact'}
    all_goal = parse_goal(goal([], target_scope='all'), state(), 'original', prefs)
    assert all_goal.target_classes == COCO_CLASSES
    implicit = parse_goal(goal([], target_scope='default'), state(), 'original', prefs)
    assert implicit.target_classes == ('dog',)
    explicit = parse_goal(goal(['person'], target_scope='explicit'), state(), 'original', prefs)
    assert explicit.target_classes == ('person',)
    quality = parse_goal(goal([], task_type='quality', target_scope='default'), state(), 'original', prefs)
    assert quality.task_type == 'quality'
    assert quality.target_classes == COCO_CLASSES  # Preference cannot add detection to the tool contract.


def test_general_prompt_selection_is_explicit_and_legacy_is_unchanged():
    current, old = state(), SimpleNamespace(detector=SimpleNamespace())
    assert 'GENERAL_PROFILE=general_rgb_v1' in goal_system(current)
    assert '水下视觉Agent的目标解析器' in goal_system(old)
    assert action_system(old) == ACTION_SYSTEM
    assert '受约束的水下视觉Agent' not in action_system(current)
    assert intent_system(old) == INTENT_SYSTEM
    assert 'COCO' in intent_system(current)
    assert 'SODD 六类' not in intent_system(current)
    assert '仅支持 propeller' not in dispatch_system(current)
    assert scripted_intent('识别人和猫', 'general_rgb_v1')['task_type'] == 'detection'
    assert scripted_intent('只识别管道', 'general_rgb_v1')['task_type'] == 'unsupported'
    assert scripted_intent('只识别管道')['task_type'] == 'detection'


def test_general_memory_only_migrates_display_readonly_and_persists(tmp_path):
    old = MemoryStore(tmp_path / 'legacy.sqlite3')
    old.set_preferences({'enabled': False, 'target_classes': ['管道'], 'display_detail': 'expanded'})
    old.save({'summary': '旧水下分析', 'versions': {'profile_id': 'sodd'}})
    before = hashlib.sha256(old.path.read_bytes()).hexdigest()
    current = GeneralMemoryStore(tmp_path / 'general.sqlite3', legacy_path=old.path)
    assert current.preferences() == {'enabled': True, 'target_classes': [], 'display_detail': 'expanded'}
    assert current.list_history() == []
    assert hashlib.sha256(old.path.read_bytes()).hexdigest() == before
    current.set_preferences({'enabled': True, 'target_classes': ['人', '手机'], 'display_detail': 'compact'})
    assert GeneralMemoryStore(current.path, legacy_path=old.path).preferences()['target_classes'] == ['person', 'cell phone']
    with pytest.raises(ValueError, match='水下'):
        current.set_preferences({'enabled': True, 'target_classes': ['管道'], 'display_detail': 'compact'})
    with pytest.raises(ValueError, match='命名空间'):
        current.save({'summary': '错误版本', 'versions': {'profile_id': 'sodd'}})
    payload = {'summary': '当前普通照片', 'versions': {'profile_id': 'general_rgb_v1'}}
    current.save(payload)
    assert len(GeneralMemoryStore(current.path).list_history()) == 1


def test_general_memory_versions_bind_profile_model_ontology_preprocessing(tmp_path):
    detector = GeneralFixture()
    session = Session(ToolContext(detector, np.full((32, 32, 3), 100, np.uint8)))
    first = versions_for(session, 'adaptive')
    assert first['profile_id'] == 'general_rgb_v1'
    assert first['ontology_version'] == ONTOLOGY_VERSION
    assert first['preprocessing_version'] == 'exif_rgb_uint8_v1'
    assert first['calibration_identity']
    detector.preprocessing_version = 'other-preprocessing'
    assert versions_for(session, 'adaptive') != first
    store = GeneralMemoryStore(tmp_path / 'general.sqlite3', legacy_path=None)
    turn = MemoryTurn(store, session, 'adaptive')
    assert turn.context['namespace'] == 'general_rgb_v1'
    assert turn.context['migration_notice']
    assert turn.context['history_refs'] == []


def test_general_memory_filters_wrong_profile_rows_even_if_manually_inserted(tmp_path):
    store = GeneralMemoryStore(tmp_path / 'general.sqlite3', legacy_path=None)
    MemoryStore.save(store, {'summary': '不能用的历史', 'versions': {'profile_id': 'sodd'}})
    assert store.snapshot()['history'] == []


def test_general_memory_runtime_saves_real_empty_results_and_never_reuses_boxes(tmp_path):
    from optical_agent.adaptive_fixture import AdaptiveScriptedClient
    from optical_agent.adaptive_runtime import AdaptiveRuntime
    store = GeneralMemoryStore(tmp_path / 'general.sqlite3', legacy_path=None)
    store.set_preferences({'enabled': True, 'target_classes': ['person'], 'display_detail': 'compact'})

    def run(message, value=100):
        session = Session(ToolContext(GeneralFixture(), np.full((32, 32, 3), value, np.uint8)))
        result = AdaptiveRuntime(AdaptiveScriptedClient(), memory_store=store).run(session, message)
        assert result['task_status'] == 'completed', result['answer']
        assert result['memory_context']['save_status'] == 'saved'
        return result

    exposure = run('只检查曝光')
    assert exposure['computation_counts']['detection'] == 0
    assert store.list_history()[0]['target_classes'] == []
    detection = run('只识别目标')
    assert detection['goal_contract']['target_classes'] == ['person']
    assert detection['target_count'] == 0
    assert store.list_history()[0]['facts']['selected_candidate_count'] == 0
    again = run('只识别目标', value=110)
    assert again['memory_context']['history_refs']
    assert again['computation_counts']['detection'] == 1
    assert not any(e['evidence_id'].startswith('memory:') for e in again['evidence'])


def test_general_knowledge_uses_only_new_sources_and_reports_new_identity():
    retriever = GeneralKnowledgeRetriever()
    meta = retriever.metadata()
    assert meta['corpus'] == 'general-natural-v1'
    assert meta['mode'] == 'tfidf'
    assert meta['frozen_historical_scores_applicable'] is False
    assert meta['acceptance_status'] == 'not_formally_evaluated'
    assert all(c['source_id'].startswith('general-') for c in retriever.chunks)
    assert all('GENERAL_' in c['path'] for c in retriever.chunks)
    result = retriever.search('为什么目标框变多或分数提高不能证明识别准确率提高？')
    assert result
    assert all({'citation_id', 'source', 'path', 'title', 'text', 'score_kind', 'corpus_hash'} <= set(r) for r in result)
    assert all(r['score_is_probability'] is False for r in result)
    assert retriever.search('zxqv91827') == []


def test_general_knowledge_legacy_alias_never_imports_c0_and_dense_missing_is_explicit():
    current = create_general_retriever(mode='legacy')
    assert current.metadata()['corpus'] == 'general-natural-v1'
    assert current.metadata()['requested_mode'] == 'legacy'
    assert current.metadata()['mode'] == 'tfidf'
    with patch('optical_agent.rag_engine.LocalEncoder', side_effect=RagUnavailable('fixture missing model')):
        assert create_general_retriever().mode == 'tfidf'
        with pytest.raises(RagUnavailable, match='不可用'):
            create_general_retriever(mode='embedding')
        with pytest.raises(RagUnavailable):
            create_general_retriever(mode='hybrid', allow_fallback=True)


def test_general_corpus_identity_stays_pinned_to_its_session_snapshot(tmp_path):
    import shutil
    source = Path(__file__).resolve().parents[1] / 'knowledge'
    target = tmp_path / 'knowledge'
    target.mkdir()
    for path in source.glob('GENERAL_*'):
        shutil.copyfile(path, target / path.name)
    original = GeneralKnowledgeRetriever(tmp_path)
    metadata = original.metadata()
    card = target / 'GENERAL_CAPABILITIES.md'
    card.write_text(card.read_text(encoding='utf8') + '\n\n本行是测试新语料版本。\n', encoding='utf8')
    assert original.metadata() == metadata
    updated = GeneralKnowledgeRetriever(tmp_path)
    assert updated.metadata()['corpus_hash'] != metadata['corpus_hash']
    assert updated.metadata()['release_config_hash'] != metadata['release_config_hash']


class FixtureEncoder:
    model_hash, asset_hash, device = 'fixture-model', 'fixture-assets', 'cpu'

    def encode(self, texts, query=False):
        vectors = np.zeros((len(texts), 512), np.float32)
        vectors[:, 0] = 1.
        return vectors


def test_general_dense_index_isolated_and_corruption_is_not_empty_knowledge(tmp_path):
    encoder = FixtureEncoder()
    with pytest.raises(RagUnavailable, match='准备'):
        GeneralKnowledgeRetriever(mode='embedding', encoder=encoder, cache_root=tmp_path)
    built = GeneralKnowledgeRetriever(mode='embedding', encoder=encoder, cache_root=tmp_path, build=True)
    assert built.search('COCO 检测模型')
    reopened = GeneralKnowledgeRetriever(mode='hybrid', encoder=encoder, cache_root=tmp_path)
    assert reopened.metadata()['corpus_hash'] == built.metadata()['corpus_hash']
    index = next(tmp_path.rglob('index.json'))
    metadata = json.loads(index.read_text(encoding='utf8'))
    metadata['corpus_hash'] = 'old-c1-hash'
    index.write_text(json.dumps(metadata), encoding='utf8')
    with pytest.raises(RagUnavailable, match='损坏'):
        GeneralKnowledgeRetriever(mode='hybrid', encoder=encoder, cache_root=tmp_path)
