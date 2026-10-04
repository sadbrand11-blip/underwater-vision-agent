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
from optical_agent.public_config import MODEL_PATH, PUBLIC_ROOT, UnavailableDetector
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
from optical_agent.memory import MemoryStore, MemoryUnavailable
from optical_agent import __version__
from optical_agent.ui_capabilities import probe_capabilities
from importlib.util import find_spec


MODEL = MODEL_PATH
SODD_TEST = Path(__file__).parent / 'assets' / 'examples'
DEMO_SAMPLE = 'frame_1431_44s_797ms_original.jpg'
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
_memory_store = MemoryStore()  # Lazy: import and health checks do not open the database.
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
        if (not MODEL.exists() or MODEL.stat().st_size < 1_000_000
                or find_spec('torch') is None or find_spec('torchvision') is None):
            _agent = DeliveryOpticalAgent(UnavailableDetector())
        else:
            from detector import TorchDetector
            detector = TorchDetector(MODEL, device='cpu')
            if tuple(detector.classes) != UnavailableDetector.classes or detector.dataset != 'sodd':
                raise RuntimeError('Detector class mapping differs from the released SODD model.')
            _agent = DeliveryOpticalAgent(detector)
    return _agent


def get_vision_agent(mode):
    if mode == 'legacy':
        return get_agent()
    if mode not in {'experiment_control', 'experiment_candidate'}:
        raise ValueError('vision_mode must be legacy, experiment_control or experiment_candidate')
    if mode not in _vision_agents:
        from optical_agent.vision_router import load_experimental
        _vision_agents[mode] = DeliveryOpticalAgent(load_experimental(mode))
    return _vision_agents[mode]


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


def render_analysis(images, vision_mode='legacy'):
    if vision_mode != 'legacy' and len(images) != 1:
        raise ValueError('0.4.0实验配置仅支持单张原图及校正候选；多帧请使用现有默认配置')
    result, views = get_vision_agent(vision_mode).inspect(images)
    return render_template('analysis.html', result=result, images={name: encode(img) for name, img in views.items()},
                                  result_json=json.dumps(result, ensure_ascii=False, indent=2),
                                  status_labels=STATUS_LABELS, error=None)


@app.get("/demo")
def demo():
    try:
        samples = sorted(SODD_TEST.glob("*_original.jpg"))
        if not samples:
            raise FileNotFoundError("D 盘没有水下示例图。请先按 README 下载 SODD 样本，或上传自己的水下图像。")
        preferred = SODD_TEST / DEMO_SAMPLE
        bgr = cv2.imread(str(preferred if preferred.exists() else samples[0]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError("内置示例图片不存在")
        return render_analysis([cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)])
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        return render_template('analysis.html', result=None, status_labels=STATUS_LABELS, error=str(exc)), 500


@app.post("/")
def inspect():
    try:
        files = [item for item in request.files.getlist("images") if item.filename]
        if not files:
            raise ValueError("请选择至少一张图像")
        images = []
        for item in files:
            raw = np.frombuffer(item.read(), dtype=np.uint8)
            bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"无法读取图像：{item.filename}")
            images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return render_analysis(images, request.form.get('vision_mode','legacy'))
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
                    "rag_release": release_config(Path(__file__).parent),
                    "cloud_configured": CloudClient().configured})


def get_retriever(mode=None, vision_mode='legacy'):
    global _retriever, _rag_encoder, _rag_encoder_signature
    if _retriever is not None and mode is None and vision_mode=='legacy':
        return _retriever  # old test/application injection remains supported.
    config=release_config(Path(__file__).parent)
    selected=mode or config['mode']
    corpus_root=Path(__file__).parent
    if selected=='legacy':
        old_root=Path(__file__).parent
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
        vision=payload.get('vision_mode','legacy')
        if vision not in {'legacy','experiment_control','experiment_candidate'}: raise ValueError('视觉配置无效')
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
    return render_template('memory.html')


@app.route('/api/memory/preferences', methods=['GET', 'PUT'])
def memory_preferences():
    try:
        if request.method == 'PUT':
            prefs = _memory_store.set_preferences(request.get_json(silent=True))
        else:
            prefs = _memory_store.preferences()
        return jsonify({'available': True, 'preferences': prefs})
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
            variant = request.form.get('demo_variant')
            if variant is not None and variant not in {'normal', 'dark', 'lost'}:
                raise ValueError('演示图类型必须为normal、dark或lost')
            samples = sorted(SODD_TEST.glob('*_original.jpg'))
            if not samples:
                raise ValueError('D盘未找到示例图，请上传自己的图片')
            preferred = SODD_TEST / DEMO_SAMPLE
            bgr = cv2.imread(str(preferred if preferred.exists() else samples[0]))
            if variant is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if variant == 'dark':
                    rgb = np.uint8(rgb.astype(float)*.6)
                elif variant == 'lost':
                    rgb = np.zeros_like(rgb)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            uploads = request.files.getlist('image')
            if len(uploads) != 1 or not uploads[0].filename:
                raise ValueError('对话入口每次需要一张图，请选择图片')
            bgr = cv2.imdecode(np.frombuffer(uploads[0].read(), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError('图片无法读取')
        vision_mode = request.form.get('vision_mode','legacy')
        detector = get_vision_agent(vision_mode).detector
        rag_mode=request.form.get('rag_mode') or None
        retriever=get_retriever(rag_mode,vision_mode)
        session = _sessions.create(detector, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), retriever)
        session.rag_metadata=retriever.metadata() if hasattr(retriever,'metadata') else {'mode':'legacy'}
        session.vision_mode = vision_mode
        session.input_provenance = ({'kind':'real_sodd' if variant == 'normal' else 'simulated_exposure_perturbation',
            'variant':variant, 'source_sample': DEMO_SAMPLE, 'source_subset_index': 7} if request.form.get('demo') == 'true' and variant else {'kind':'uploaded_or_legacy_demo'})
        return jsonify({'session_id': session.id, 'expires_after_idle_seconds': 1800,
                        'vision_mode': vision_mode,
                        'rag_metadata':session.rag_metadata,
                        'vision_models': detector.metadata() if hasattr(detector,'metadata') else {'mode':'legacy','experimental':False},
                        'input_provenance': session.input_provenance,
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
        log_dir = PUBLIC_ROOT / 'runs' / engine / agent_mode
        result = runtime(client, log_dir=log_dir,
                         memory_store=_memory_store).run(session, payload.get('message'))
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
    app.run(host="127.0.0.1", port=7860, debug=False)
