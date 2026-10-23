"""Local upload and result viewer. Start with `python app.py`."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, Response

from optical_agent.vision_delivery import DeliveryOpticalAgent, DELIVERY_REVISION
from optical_agent.llm import CloudClient, ScriptedClient, load_local_env
from optical_agent.rag import KnowledgeRetriever
from optical_agent.rag_engine import create_retriever, release_config, LocalEncoder, RagUnavailable, DATA_ROOT, build_chunks, fingerprint, file_hash
from optical_agent.rag_delivery import KnowledgeView
from threading import RLock
from time import perf_counter
from optical_agent.runtime import AgentRuntime
from optical_agent.adaptive_runtime import AdaptiveRuntime
from optical_agent.adaptive_fixture import AdaptiveScriptedClient
from optical_agent.state import SessionStore
from optical_agent.memory import MemoryStore, GeneralMemoryStore, MemoryUnavailable
from optical_agent import __version__
from optical_agent.ui_capabilities import probe_capabilities
from importlib.util import find_spec
from optical_agent.general_vision import GeneralDetector, DEFAULT_CHECKPOINT
from optical_agent.general_catalog import COCO_CLASSES, CLASS_NAMES, PROFILE_ID, ONTOLOGY_VERSION
from optical_agent.image_input import read_image_bytes
from optical_agent.general_knowledge import create_general_retriever, general_release_config
from optical_agent.general_demo import demo_image
from optical_agent.public_config import PUBLIC_ROOT


MODEL = DEFAULT_CHECKPOINT
SODD_TEST = Path(r"D:\CodexData\optical_agent\sodd\SODD\data\test\images")
DEMO_SAMPLE = "frame_1031.61_original.jpg"
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
_agent = None
_vision_agents = {}
_retriever = None
_rag_retrievers = {}
_rag_encoder = None
_rag_encoder_signature = None
_rag_lock = RLock()
_sessions = SessionStore()
_memory_store = GeneralMemoryStore()  # Lazy: import and health checks do not open the database.
load_local_env(Path(__file__).parent / '.env')
STATUS_LABELS = {"reliable": "通过内部检查", "unreliable": "识别结论不可靠",
                 "quality_failure": "图像质量失败", "quality_unassessable": "无法评估原始曝光"}


@app.before_request
def require_local_origin():
    if urlsplit(request.host_url).hostname not in {'localhost', '127.0.0.1'}:
        return jsonify({'error': '此服务仅允许本机访问'}), 403
    origin = request.headers.get('Origin')
    if origin is not None and origin != request.host_url.rstrip('/'):
        return jsonify({'error': '此服务仅允许同源网页访问'}), 403


def get_agent():
    global _agent
    if _agent is None:
        _agent = DeliveryOpticalAgent(GeneralDetector(checkpoint=MODEL))
    return _agent


def get_vision_agent(mode):
    if mode in {'legacy', 'general_rgb_v1'}:
        return get_agent()
    raise ValueError('水下实验配置已归档；当前版本请使用 general_rgb_v1 通用照片配置。')


def encode(rgb):
    success, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not success:
        raise ValueError("Image encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")




@app.get("/")
def index():
    return render_template('showcase.html')


@app.get('/showcase')
def showcase():
    return render_template('showcase.html')


@app.get('/analysis')
def analysis_page():
    return render_template('analysis.html', result=None, error=None, status_labels=STATUS_LABELS)


def render_analysis(images, vision_mode='general_rgb_v1'):
    result, views = get_vision_agent(vision_mode).inspect(images)
    return render_template('analysis.html', result=result, images={name: encode(img) for name, img in views.items()},
                           result_json=json.dumps(result, ensure_ascii=False, indent=2),
                           status_labels=STATUS_LABELS, error=None)


@app.get("/demo")
def demo():
    try:
        rgb, _ = demo_image()
        return render_analysis([rgb])
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        return render_template('analysis.html', result=None, status_labels=STATUS_LABELS, error=str(exc)), 400


@app.post("/")
def inspect():
    try:
        files = [item for item in request.files.getlist("images") if item.filename]
        if not files:
            raise ValueError("请选择至少一张图像")
        images = []
        for item in files:
            images.append(read_image_bytes(item.read()))
        return render_analysis(images, request.form.get('vision_mode','general_rgb_v1'))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        return render_template('analysis.html', result=None, status_labels=STATUS_LABELS, error=str(exc)), 400


@app.get("/health")
def health():
    return jsonify({"model_present": MODEL.exists() and MODEL.stat().st_size > 1_000_000,
                    "detector_available": MODEL.exists() and MODEL.stat().st_size > 1_000_000 and find_spec('torch') is not None and find_spec('torchvision') is not None,
                    "model_path": str(MODEL), "app_version": __version__,
                    "runtime_revision": DELIVERY_REVISION,
                    "rag_revision": "v050_rag_delivery_r3",
                    "memory_revision": "v051_memory_r1",
                    "graph_revision": "v052_langgraph_r1",
                    "langgraph_available": find_spec('langgraph') is not None,
                    "rag_release": general_release_config(Path(__file__).parent),
                    "profile_id": PROFILE_ID, "ontology_version": ONTOLOGY_VERSION,
                    "supported_class_count": len(COCO_CLASSES),
                    "cloud_configured": CloudClient().configured})


def _get_archived_retriever(mode=None, vision_mode='legacy'):
    global _retriever, _rag_encoder, _rag_encoder_signature
    if _retriever is not None and mode is None and vision_mode=='legacy':
        return _retriever  # old test/application injection remains supported.
    config=release_config(Path(__file__).parent)
    selected=mode or config['mode']
    corpus_root=DATA_ROOT/'c1_source'
    if selected=='legacy':
        old_root=DATA_ROOT/'c0_source'
        old_root=old_root if old_root.exists() else Path(__file__).parent
        manifest=json.loads((old_root/'knowledge/sources.json').read_text(encoding='utf8'))
        corpus_identity=fingerprint({s['path']:file_hash(old_root/s['path']) for s in manifest})
    else:
        corpus_identity=fingerprint(build_chunks(corpus_root if corpus_root.exists() else Path(__file__).parent))
    model_root=DATA_ROOT/'models/bge-small-zh-v1.5'
    model_signature=tuple((p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(model_root.iterdir()) if p.is_file()) if model_root.exists() else ()
    key=(mode,vision_mode,fingerprint(config),corpus_identity,model_signature)
    with _rag_lock:
        if key not in _rag_retrievers:
            if selected in {'embedding','hybrid'}:
                try:
                    if _rag_encoder is None or _rag_encoder_signature!=model_signature:
                        _rag_encoder=LocalEncoder(device='cpu')
                        _rag_encoder_signature=model_signature
                except RagUnavailable:
                    if mode is not None: raise
                    _rag_encoder=None
            _rag_retrievers[key]=create_retriever(Path(__file__).parent,mode,vision_mode,
                                                 encoder=_rag_encoder,allow_fallback=True)
            _rag_retrievers[key]=KnowledgeView(_rag_retrievers[key],vision_mode)
            if fingerprint(release_config(Path(__file__).parent))!=fingerprint(config):
                del _rag_retrievers[key]
                raise RagUnavailable('创建期间发布配置发生变化，请重试')
            while len(_rag_retrievers)>12:
                del _rag_retrievers[next(iter(_rag_retrievers))]
        return _rag_retrievers[key]


def get_retriever(mode=None, vision_mode='general_rgb_v1'):
    if _retriever is not None and mode is None:
        return _retriever  # explicit test/application injection remains supported.
    if vision_mode == 'legacy':
        return _get_archived_retriever(mode, vision_mode)
    if vision_mode in {'experiment_control', 'experiment_candidate'}:
        raise ValueError('水下实验知识已归档，请使用通用照片知识库。')
    global _rag_encoder, _rag_encoder_signature
    root = Path(__file__).parent
    config = general_release_config(root)
    model_root = DATA_ROOT / 'models/bge-small-zh-v1.5'
    signature = tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(model_root.iterdir()) if p.is_file()) if model_root.exists() else ()
    key = ('general', mode, config['release_config_hash'], signature)
    with _rag_lock:
        if key not in _rag_retrievers:
            encoder = None
            if mode in {'embedding', 'hybrid'}:
                if _rag_encoder is None or _rag_encoder_signature != signature:
                    _rag_encoder = LocalEncoder(device='cpu')
                    _rag_encoder_signature = signature
                encoder = _rag_encoder
            _rag_retrievers[key] = create_general_retriever(root, mode, vision_mode=PROFILE_ID,
                                                          encoder=encoder, allow_fallback=mode is None)
            while len(_rag_retrievers) > 12:
                del _rag_retrievers[next(iter(_rag_retrievers))]
        return _rag_retrievers[key]


@app.get('/api/capabilities')
def capabilities():
    return jsonify({'profile_id': PROFILE_ID, 'ontology_version': ONTOLOGY_VERSION,
                    'supported_classes': [{'name': name, 'label': CLASS_NAMES.get(name, name)} for name in COCO_CLASSES],
                    'supported_inputs': ['JPEG 8-bit', 'PNG 8-bit'],
                    'scope': 'ordinary_natural_photos_not_specialized_domains',
                    'memory_namespace': PROFILE_ID})


@app.get('/knowledge')
def knowledge_page():
    return render_template('knowledge.html')


@app.post('/api/knowledge/search')
def knowledge_search():
    payload=request.get_json(silent=True)
    if not isinstance(payload,dict): return jsonify({'error':'请求必须是JSON对象'}),400
    try:
        query=payload.get('query')
        if not isinstance(query,str) or not query.strip() or len(query)>2000:
            raise ValueError('请输入1—2000字符的问题')
        top_k=payload.get('top_k',3)
        if type(top_k) is not int or not 1<=top_k<=10: raise ValueError('top_k须为1—10的整数')
        vision=payload.get('vision_mode','general_rgb_v1')
        if vision not in {'legacy','general_rgb_v1'}: raise ValueError('视觉配置已归档或无效')
        modes=payload.get('modes',['tfidf','embedding','hybrid'])
        if not isinstance(modes,list) or not modes or len(modes)>4 or any(m not in {'legacy','tfidf','embedding','hybrid'} for m in modes):
            raise ValueError('检索模式无效')
        outputs={}
        for mode in modes:
            started=perf_counter()
            try:
                retriever=get_retriever(mode,vision)
                load_ms=(perf_counter()-started)*1000
                started=perf_counter(); results=retriever.search(query,top_k)
                query_ms=(perf_counter()-started)*1000
                outputs[mode]={'available':True,'results':results,'query_ms':query_ms,'load_ms':load_ms,
                    'metadata':retriever.metadata(), 'empty_reason':'当前拒绝阈值下未交付知识片段' if not results else None}
            except RagUnavailable as exc:
                outputs[mode]={'available':False,'error':str(exc),'results':None}
        return jsonify({'query':query,'vision_mode':vision,'comparisons':outputs,
                        'paid_api_requests':0,'score_is_probability':False})
    except ValueError as exc: return jsonify({'error':str(exc)}),400


@app.get('/agent')
def conversation():
    return render_template('conversation.html', cloud_configured=CloudClient().configured,
                           langgraph_available=find_spec('langgraph') is not None,
                           capabilities=probe_capabilities(MODEL, DATA_ROOT/'models/bge-small-zh-v1.5', experimental=False))


@app.get('/memory')
def memory_page():
    return render_template('memory.html', general_classes=[(name, CLASS_NAMES.get(name, name)) for name in COCO_CLASSES])


@app.route('/api/memory/preferences', methods=['GET', 'PUT'])
def memory_preferences():
    try:
        if request.method == 'PUT':
            prefs = _memory_store.set_preferences(request.get_json(silent=True))
        else:
            prefs = _memory_store.preferences()
        return jsonify({'available': True, 'preferences': prefs, 'namespace': PROFILE_ID,
                        'migration_notice': '通用照片使用独立偏好与历史；旧水下类别不再应用。'})
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except MemoryUnavailable as exc:
        return jsonify({'available': False, 'error': str(exc)}), 503


@app.route('/api/memory/history', methods=['GET', 'DELETE'])
def memory_history():
    try:
        if request.method == 'DELETE':
            return jsonify({'deleted': _memory_store.delete()})
        rows = _memory_store.list_history(request.args.get('q', ''))
        return jsonify({'available': True, 'history': rows, 'count': len(rows),
                        'evidence_scope': 'historical_summary_only_not_current_evidence'})
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except MemoryUnavailable as exc:
        return jsonify({'available': False, 'error': str(exc)}), 503


@app.delete('/api/memory/history/<memory_id>')
def delete_memory(memory_id):
    try:
        count = _memory_store.delete(memory_id)
        return jsonify({'deleted': count}), 200 if count else 404
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except MemoryUnavailable as exc:
        return jsonify({'available': False, 'error': str(exc)}), 503


@app.post('/api/sessions')
def create_session():
    try:
        if request.form.get('demo') == 'true':
            rgb, provenance = demo_image(request.form.get('demo_variant') or 'normal')
        else:
            uploads = request.files.getlist('image')
            if len(uploads) != 1 or not uploads[0].filename:
                raise ValueError('对话入口每次需要一张JPEG或PNG图片')
            rgb = read_image_bytes(uploads[0].read())
            provenance = {'kind': 'uploaded_natural_image', 'preprocessing': 'exif_oriented_rgb8'}
        vision_mode = request.form.get('vision_mode', PROFILE_ID)
        detector = get_vision_agent(vision_mode).detector
        rag_mode = request.form.get('rag_mode') or None
        knowledge_scope = (PROFILE_ID if getattr(detector, 'profile_id', None) == PROFILE_ID
                           else 'legacy' if vision_mode == PROFILE_ID else vision_mode)
        retriever = get_retriever(rag_mode, knowledge_scope)
        session = _sessions.create(detector, rgb, retriever)
        session.rag_metadata = retriever.metadata() if hasattr(retriever, 'metadata') else {'mode': 'legacy'}
        session.vision_mode = vision_mode
        session.input_provenance = provenance
        return jsonify({'session_id': session.id, 'expires_after_idle_seconds': 1800,
                        'vision_mode': vision_mode, 'profile_id': getattr(detector, 'profile_id', 'legacy'),
                        'ontology_version': getattr(detector, 'ontology_version', None),
                        'rag_metadata': session.rag_metadata,
                        'vision_models': detector.metadata() if hasattr(detector, 'metadata') else {'mode':'legacy','experimental':False},
                        'input_provenance': provenance,
                        'image_refs': {'original': f'/api/sessions/{session.id}/images/original'}}), 201
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        return jsonify({'error': str(exc)}), 400


@app.post('/api/chat')
def chat():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({'error': '请求必须是JSON对象'}), 400
    try:
        if not isinstance(payload.get('session_id'), str):
            raise ValueError('缺少会话编号')
        session = _sessions.get(payload['session_id'])
        mode = payload.get('mode', 'cloud')
        if mode not in {'cloud', 'scripted'}:
            raise ValueError('mode必须为cloud或scripted')
        load_local_env(Path(__file__).parent / '.env')
        agent_mode = payload.get('agent_mode', 'legacy')
        if agent_mode not in {'legacy', 'adaptive'}:
            raise ValueError('agent_mode必须为legacy或adaptive')
        engine = payload.get('runtime_engine', 'native')
        if engine not in {'native', 'langgraph'}:
            raise ValueError('runtime_engine必须为native或langgraph')
        if engine == 'langgraph' and agent_mode != 'adaptive':
            raise ValueError('LangGraph仅支持动态调度，请选择adaptive。')
        if hasattr(session.context.detector, 'branch_names') and agent_mode != 'adaptive':
            raise ValueError('0.4.0实验模型需要选择“动态目标与规划”，以保证按目标类别调用分支；原有调度对照请使用现有默认模型')
        client = (AdaptiveScriptedClient() if agent_mode == 'adaptive' else ScriptedClient()) if mode == 'scripted' else CloudClient()
        runtime = AdaptiveRuntime if agent_mode == 'adaptive' else AgentRuntime
        if engine == 'langgraph':
            try:
                from langgraph.graph import StateGraph
                from optical_agent.langgraph_runtime import LangGraphRuntime
            except (ImportError, RuntimeError) as exc:
                raise ValueError('LangGraph依赖不可用，请使用D盘隔离环境或双击LangGraph对照启动入口。') from None
            runtime = LangGraphRuntime
        if not session.lock.acquire(blocking=False):
            raise ValueError('当前会话正在执行，请等待完成。')
        try:
            previous = getattr(session, 'runtime_engine', None)
            if previous is not None and previous != engine:
                raise ValueError('此会话已固定运行器；切换运行器请创建新图片会话。')
            session.runtime_engine = engine
        finally:
            session.lock.release()
        log_dir = Path(r'D:\CodexData\optical_agent\general_v060\runs\sessions') / engine
        result = runtime(client, log_dir=log_dir,
                         memory_store=_memory_store).run(session, payload.get('message'))
        result['profile_id'] = getattr(session.context.detector, 'profile_id', 'legacy')
        result['ontology_version'] = getattr(session.context.detector, 'ontology_version', None)
        result['agent_mode'] = agent_mode
        result['runtime_engine'] = engine
        result.setdefault('graph_trace', [])
        result['runtime_revision'] = DELIVERY_REVISION
        result['vision_mode'] = getattr(session,'vision_mode','legacy')
        result['rag_metadata'] = getattr(session,'rag_metadata',{'mode':'legacy'})
        result['vision_models'] = session.context.detector.metadata() if hasattr(session.context.detector,'metadata') else {'mode':'legacy','experimental':False}
        result['input_provenance'] = getattr(session, 'input_provenance', None)
        result['image_refs'] = {key: f'/api/sessions/{session.id}/images/{key}' for key in result['image_refs']}
        return jsonify(result)
    except KeyError:
        return jsonify({'error': '会话不存在或已过期，请重新上传'}), 404
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400


@app.get('/api/sessions/<session_id>/images/<image_id>')
def session_image(session_id, image_id):
    try:
        session = _sessions.get(session_id)
        if not session.lock.acquire(timeout=1):
            return jsonify({'error': '图片正在更新，请稍后重试'}), 409
        try:
            images = (session.adaptive.images if hasattr(session, 'adaptive') and image_id in session.adaptive.images
                      and image_id != 'original' else session.context.images)
            if image_id not in images:
                return jsonify({'error': '此会话中没有该图片'}), 404
            image = images[image_id].copy()
        finally:
            session.lock.release()
        success, encoded = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not success:
            raise ValueError('无法生成预览')
        return Response(encoded.tobytes(), mimetype='image/jpeg', headers={'Cache-Control': 'no-store'})
    except KeyError:
        return jsonify({'error': '会话不存在或已过期'}), 404
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Local natural-image agent')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    app.run(host="127.0.0.1", port=args.port, debug=False)
