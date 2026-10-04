"""Public bootstrap: no weights must mean unavailable, never empty recognition."""
import io
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

import app as web
from optical_agent.public_config import UnavailableDetector
from optical_agent.memory import MemoryStore
from optical_agent.state import SessionStore
from run_demo import download_detector


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, 'MODEL', tmp_path/'missing.pt')
    monkeypatch.setattr(web, '_agent', None)
    monkeypatch.setattr(web, '_sessions', SessionStore())
    monkeypatch.setattr(web, '_memory_store', MemoryStore(tmp_path/'memory.sqlite3'))
    monkeypatch.setattr(web, 'PUBLIC_ROOT', tmp_path)
    return web.app.test_client()


def create(client, variant='dark'):
    response=client.post('/api/sessions', data={'demo':'true','demo_variant':variant,'rag_mode':'tfidf'})
    assert response.status_code==201, response.json
    return response.json['session_id']


def ask(client, sid, message):
    return client.post('/api/chat', json={'session_id':sid,'message':message,'mode':'scripted','agent_mode':'adaptive'}).json


def test_public_exposure_without_detector(client):
    sid=create(client)
    before=web._sessions.get(sid).context.images['original'].copy()
    result=ask(client,sid,'只检查曝光')
    assert result['task_status']=='completed'
    assert result['request_attempts']==0
    assert not any(t.get('tool')=='detect_objects' for t in result['trace'])
    np.testing.assert_array_equal(before,web._sessions.get(sid).context.images['original'])
    computed=web._sessions.get(sid).adaptive.counts['quality']
    repeated=ask(client,sid,'只检查曝光')
    assert repeated['task_status']=='completed'
    assert web._sessions.get(sid).adaptive.counts['quality']==computed


def test_public_correction_and_black_guard(client):
    sid=create(client)
    result=ask(client,sid,'只校正曝光')
    assert result['task_status']=='completed'
    assert any(t.get('tool')=='generate_exposure_candidate' for t in result['trace'])
    black=create(client,'lost')
    result=ask(client,black,'只校正曝光')
    assert result['task_status']=='completed'
    assert not any(t.get('tool')=='generate_exposure_candidate' for t in result['trace'])


def test_detection_unavailable_not_empty(client):
    sid=create(client)
    result=ask(client,sid,'只识别管道')
    assert result['task_status']!='completed'
    failures=[t for t in result['trace'] if t.get('tool')=='detect_objects']
    assert failures and all(t.get('error_category')=='model_unavailable' for t in failures)
    assert not web._sessions.get(sid).adaptive.raw_detections


def test_public_routes_and_corpus(client):
    assert client.get('/showcase').status_code==200
    assert client.get('/agent').status_code==200
    health=client.get('/health').json
    assert not health['model_present']
    assert health['rag_release']['corpus']=='public-sanitized-v2'
    response=client.post('/api/knowledge/search',json={'query':'管道 检测 候选 准确率','modes':['tfidf']})
    assert response.status_code==200
    assert response.json['comparisons']['tfidf']['available']


def test_missing_model_raises():
    with pytest.raises(RuntimeError,match='unavailable'):
        UnavailableDetector().predict(np.zeros((16,16,3),dtype=np.uint8))


def test_optional_embedding_dependencies_are_explicitly_unavailable(tmp_path,monkeypatch):
    import builtins
    from optical_agent.rag_engine import LocalEncoder, RagUnavailable
    (tmp_path/'model.safetensors').write_bytes(b'fixture; never loaded')
    original=builtins.__import__
    for missing in ('torch','transformers'):
        def blocked(name,*args,**kwargs):
            if name==missing:
                raise ImportError('Optional dependency absent')
            return original(name,*args,**kwargs)
        with monkeypatch.context() as scoped:
            scoped.setattr(builtins,'__import__',blocked)
            with pytest.raises(RagUnavailable,match='dependencies unavailable'):
                LocalEncoder(tmp_path)


def test_cached_checkpoint_without_torch_still_allows_quality(client,tmp_path,monkeypatch):
    model=tmp_path/'cached.pt'
    model.write_bytes(b'0'*1_000_001)
    monkeypatch.setattr(web,'MODEL',model)
    original=web.find_spec
    monkeypatch.setattr(web,'find_spec',lambda name:None if name in {'torch','torchvision'} else original(name))
    result=ask(client,create(client),'只检查曝光')
    assert result['task_status']=='completed'
    assert not client.get('/health').json['detector_available']


def test_corrupt_download_never_installed(tmp_path,monkeypatch):
    import run_demo
    class Response:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def raise_for_status(self):pass
        def iter_content(self,*a):yield b'wrong model'
    monkeypatch.setattr(run_demo.requests,'get',lambda *a,**k:Response())
    dest=tmp_path/'model.pt'
    with pytest.raises(RuntimeError,match='verification'):
        download_detector(dest)
    assert not dest.exists() and not dest.with_suffix('.part').exists()
