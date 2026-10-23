"""Current public-facing contract, independent of archived underwater fixtures."""
from dataclasses import asdict
from io import BytesIO

import numpy as np
from PIL import Image
import pytest

import app as web
from optical_agent.general_catalog import COCO_CLASSES, PROFILE_ID, ONTOLOGY_VERSION
from optical_agent.general_vision import GeneralDetector
from optical_agent.general_knowledge import GeneralKnowledgeRetriever
from optical_agent.memory import GeneralMemoryStore
from optical_agent.state import SessionStore
from optical_agent.vision_delivery import DeliveryOpticalAgent
from quality import GENERAL_QUALITY_CONFIG


def png():
    rng = np.random.default_rng(3)
    image = Image.fromarray(rng.integers(65, 185, (64, 96, 3), dtype=np.uint8))
    data = BytesIO(); image.save(data, format='PNG'); data.seek(0)
    return data


@pytest.fixture
def client(monkeypatch, tmp_path):
    detector = GeneralDetector(tmp_path/'missing-model.pth', tmp_path/'missing-calibration.json')
    monkeypatch.setattr(web, '_agent', DeliveryOpticalAgent(detector))
    monkeypatch.setattr(web, '_sessions', SessionStore())
    monkeypatch.setattr(web, '_retriever', GeneralKnowledgeRetriever())
    monkeypatch.setattr(web, '_memory_store', GeneralMemoryStore(tmp_path/'general.sqlite3', legacy_path=tmp_path/'old.sqlite3'))
    monkeypatch.setattr(web.CloudClient, 'complete', lambda *a, **k: pytest.fail('No cloud calls authorized'))
    return web.app.test_client()


def upload(client):
    response = client.post('/api/sessions', data={'image': (png(), 'photo.png')})
    assert response.status_code == 201, response.json
    return response.json


def ask(client, session, message, engine='native'):
    response = client.post('/api/chat', json={'session_id': session['session_id'], 'message': message,
        'mode': 'scripted', 'agent_mode': 'adaptive', 'runtime_engine': engine})
    assert response.status_code == 200, response.json
    return response.json


def test_current_capabilities_are_80_real_classes_and_no_water_default(client):
    data = client.get('/api/capabilities').json
    assert data['profile_id'] == PROFILE_ID
    assert len(data['supported_classes']) == 80
    names = {x['name'] for x in data['supported_classes']}
    assert 'person' in names and 'pipe' not in names and 'N/A' not in names
    page = client.get('/agent').get_data(as_text=True)
    assert '只识别人' in page and 'COCO80' in page
    assert '默认六类模型' not in page


def test_exposure_session_without_weights_completes_without_detection(client):
    session = upload(client)
    assert session['profile_id'] == PROFILE_ID
    result = ask(client, session, '只检查曝光')
    assert result['task_status'] == 'completed'
    assert result['request_attempts'] == 0
    assert result['target_count'] is None
    assert [x['tool'] for x in result['trace'] if x['type'] == 'tool'] == ['assess_image_quality']


def test_missing_detector_is_error_not_zero_targets(client):
    result = ask(client, upload(client), '只识别人')
    assert result['task_status'] != 'completed'
    assert result['target_count'] is None
    failures = [x for x in result['trace'] if x['type'] == 'tool' and not x['ok']]
    assert failures and failures[0]['error_category'] == 'model_unavailable'


@pytest.mark.parametrize('message', ['只识别管道', '识别二维码', '只识别鱼', '只识别蜥蜴', '只识别猫头鹰', '只识别pipe', '只识别人，蜥蜴', '只识别人,蜥蜴', '检测人，然后检测蜥蜴', '分析这张图中的蜥蜴'])
def test_unsupported_classes_do_not_become_all_classes(client, message):
    result = ask(client, upload(client), message)
    assert result['task_status'] == 'unsupported'
    assert result['tool_calls'] == 0


@pytest.mark.parametrize('message,expected', [('只识别热狗',['hot dog']),('只识别泰迪熊',['teddy bear']),('只识别单板滑雪板',['snowboard']),('只识别人和汽车',['person','car'])])
def test_compound_class_aliases_do_not_expand_to_short_substrings(client, message, expected):
    result = ask(client, upload(client), message)
    assert result['goal_contract']['target_classes'] == expected


def test_development_original_policy_completes_dark_conditional_task(client, monkeypatch):
    from test_general_vision import GeneralFixture
    detector = GeneralFixture()
    detector.exposure_policy = {'status':'development_selected','selected_method':'original'}
    monkeypatch.setattr(web, '_agent', DeliveryOpticalAgent(detector))
    image = np.full((64,96,3), 35, dtype=np.uint8)
    stream=BytesIO(); Image.fromarray(image).save(stream,format='PNG');stream.seek(0)
    response=client.post('/api/sessions',data={'image':(stream,'dark.png')})
    result=ask(client,response.json,'检查这张图，必要时校正曝光，然后只识别人，说明结果是否可靠。')
    assert result['task_status']=='completed'
    assert result['selected_image_id']=='original'
    assert result['computation_counts']['correction']==0
    assert result['task_validation']['passed']


def test_old_water_experiment_mode_is_explicitly_archived(client):
    response = client.post('/api/sessions', data={'image': (png(), 'photo.png'), 'vision_mode':'experiment_candidate'})
    assert response.status_code == 400 and '归档' in response.json['error']


def test_general_quality_native_langgraph_parity(client):
    pytest.importorskip('langgraph')
    results = [ask(client, upload(client), '只检查曝光', engine) for engine in ('native','langgraph')]
    for field in ('goal_contract','task_status','vision_status','selected_image_id','tool_calls','request_attempts'):
        assert results[0][field] == results[1][field]
    assert results[1]['graph_trace']


def test_new_memory_preferences_support_person_not_old_pipe(client):
    response = client.put('/api/memory/preferences', json={'enabled':True,'target_classes':['人'],'display_detail':'compact'})
    assert response.status_code == 200 and response.json['preferences']['target_classes'] == ['person']
    assert response.json['namespace'] == PROFILE_ID
    response = client.put('/api/memory/preferences', json={'enabled':True,'target_classes':['管道'],'display_detail':'compact'})
    assert response.status_code == 400
