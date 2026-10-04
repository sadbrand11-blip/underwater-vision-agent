"""Process-local MCP image registry; visual work stays in the adaptive executor."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError

from agent import _overlay
from optical_agent.adaptive_goals import ALIASES, CLASSES, Goal
from optical_agent.adaptive_tools import AdaptiveState, execute, pixel_hash
from optical_agent.diagnostics import redact
from optical_agent.tools import ToolContext

from optical_agent.public_config import DATA_ROOT, PUBLIC_ROOT
MCP_ROOT = PUBLIC_ROOT / 'mcp'
MAX_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 16_000_000


class AdapterError(ValueError):
    def __init__(self, category, message):
        super().__init__(message)
        self.category = category


class UnavailableDetector:
    """Quality remains measurable; this object must never claim empty detection."""
    dataset, threshold, calibration = 'sodd', .35, None
    classes = ('background', *CLASSES)
    model_available = False

    def predict(self, image):
        raise AdapterError('model_unavailable', '检测权重不可用；这不表示图中没有目标。')


def load_default_detector():
    from optical_agent.public_config import MODEL_PATH
    path = MODEL_PATH
    if not path.is_file() or path.stat().st_size < 1_000_000:
        return UnavailableDetector()
    try:
        from detector import TorchDetector
        detector = TorchDetector(path)
        if tuple(detector.classes) != ('background', *CLASSES) or detector.dataset != 'sodd':
            return UnavailableDetector()
        return detector
    except Exception as exc:
        logging.getLogger('optical_mcp').warning('检测权重加载失败：%s', type(exc).__name__)
        return UnavailableDetector()


@dataclass
class ImageContext:
    context_id: str
    state: AdaptiveState
    source_path: Path
    file_hash: str
    last_used: float
    ids: dict = field(default_factory=dict)
    lock: RLock = field(default_factory=RLock)
    active: int = 0


class VisionMCPAdapter:
    def __init__(self, detector_factory=load_default_detector, *, data_root=DATA_ROOT,
                 output_root=MCP_ROOT / 'images', ttl=1800, max_contexts=20, clock=time.monotonic):
        self.data_root = Path(data_root).resolve()
        self.output_root = Path(output_root).resolve()
        if not self.output_root.is_relative_to(self.data_root):
            raise ValueError('MCP outputs must stay inside the data root')
        self.detector_factory, self.detector = detector_factory, None
        self.ttl, self.max_contexts, self.clock = ttl, max_contexts, clock
        if ttl <= 0 or max_contexts < 1:
            raise ValueError('Registry limits must be positive')
        self.contexts, self.index, self.by_file = {}, {}, {}
        self.registry_lock, self.gpu_lock = RLock(), RLock()

    def _read_image(self, value):
        if not isinstance(value, str) or not value.strip() or '\x00' in value:
            raise AdapterError('parameters', 'image_path 必须是有效的图片绝对路径。')
        try:
            path = Path(value)
            if not path.is_absolute():
                raise AdapterError('invalid_path', '请提供 D 盘数据目录内的绝对路径。')
            path = path.resolve(strict=True)
            if not path.is_relative_to(self.data_root) or not path.is_file():
                raise AdapterError('invalid_path', '输入图片必须位于指定的 D 盘数据目录内。')
            if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}:
                raise AdapterError('invalid_image', '仅接受 JPEG、PNG、BMP、TIFF 或 WebP 图片。')
            with path.open('rb') as stream:
                if path.stat().st_size > MAX_BYTES:
                    raise AdapterError('size_limit', '图片文件超过 32 MiB 上限。')
                content = stream.read(MAX_BYTES + 1)
            if len(content) > MAX_BYTES:
                raise AdapterError('size_limit', '图片文件超过 32 MiB 上限。')
            with Image.open(io.BytesIO(content)) as image:
                width, height = image.size
                if width * height > MAX_PIXELS:
                    raise AdapterError('pixel_limit', '图片超过 1600 万像素上限。')
                if min(width, height) < 8 or getattr(image, 'n_frames', 1) != 1:
                    raise AdapterError('invalid_image', '需要至少 8×8 像素的单帧图片。')
                image.verify()
            # Decode exactly as data.read_rgb (including its EXIF convention).
            decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise AdapterError('invalid_image', '图片无法解码。')
            pixels = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            return path, hashlib.sha256(content).hexdigest(), pixels
        except AdapterError:
            raise
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError, RuntimeError):
            raise AdapterError('invalid_image', '图片不存在、无法读取或无法解码。') from None

    def _remove(self, ctx):
        del self.contexts[ctx.context_id]
        self.by_file.pop(ctx.file_hash, None)
        for public_id in ctx.ids.values():
            self.index.pop(public_id, None)

    def _expire(self):
        now = self.clock()
        for ctx in list(self.contexts.values()):
            if not ctx.active and now - ctx.last_used >= self.ttl:
                self._remove(ctx)

    def _register(self, value):
        path, digest, pixels = self._read_image(value)
        with self.registry_lock:
            self._expire()
            existing = self.by_file.get(digest)
            if existing:
                return self.contexts[existing].ids['original']
            if len(self.contexts) >= self.max_contexts:
                idle = [c for c in self.contexts.values() if not c.active]
                if not idle:
                    raise AdapterError('capacity', '所有图片上下文正在使用，请稍后重试。')
                self._remove(min(idle, key=lambda c: c.last_used))
            if self.detector is None:
                # Initialization and inference use the same process-wide GPU lock.
                with self.gpu_lock:
                    self.detector = self.detector_factory()
            state = AdaptiveState(ToolContext(self.detector, pixels))
            ctx = ImageContext(uuid.uuid4().hex, state, path, digest, self.clock())
            self.contexts[ctx.context_id] = ctx
            self.by_file[digest] = ctx.context_id
            return self._public_id(ctx, 'original')

    def _public_id(self, ctx, internal_id):
        if internal_id not in ctx.ids:
            public_id = 'img_' + uuid.uuid4().hex
            ctx.ids[internal_id] = public_id
            self.index[public_id] = (ctx.context_id, internal_id)
        return ctx.ids[internal_id]

    @contextmanager
    def _lease(self, image_id):
        with self.registry_lock:
            self._expire()
            if not isinstance(image_id, str) or image_id not in self.index:
                raise AdapterError('unknown_image', '图片编号不存在、已过期或属于已退出的服务；请重新登记图片。')
            context_id, internal_id = self.index[image_id]
            ctx = self.contexts[context_id]
            ctx.active += 1  # Protect queued as well as currently executing calls.
        try:
            with ctx.lock:
                yield ctx, internal_id
        finally:
            with self.registry_lock:
                ctx.last_used = self.clock()
                ctx.active -= 1

    def _run(self, ctx, kind, image_id, tool, arguments, classes=CLASSES):
        state = ctx.state
        state.check_integrity()
        state.begin(Goal(kind, image_id, tuple(classes), 'always' if kind == 'correction' else 'never', False))
        result = execute(state, tool, arguments)
        if not result['ok']:
            # The adapter already checks every public prerequisite. The legacy
            # executor labels in-function failures "prerequisite" as well;
            # here those are computation failures, not requests to retry a step.
            category = result.get('error_category', 'computation')
            if category == 'prerequisite':
                category = 'computation'
            raise AdapterError(category, '视觉工具执行失败；请检查模型、图片和配置。')
        state.check_integrity()
        cached = bool(state.events[-1]['cached'])
        state.events[:] = state.events[-1:]  # Keep caches, without growing a conversation log.
        return result['data']['data'], cached

    def _save(self, ctx, name, pixels):
        folder = self.output_root / ctx.context_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (name + '.png')
        if not path.exists():
            temporary = folder / (uuid.uuid4().hex + '.tmp')
            try:
                Image.fromarray(pixels).save(temporary, format='PNG')
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return str(path)

    def _targets(self, values):
        if values is None:
            return CLASSES
        if not isinstance(values, list) or not 1 <= len(values) <= 12 or any(not isinstance(x, str) for x in values):
            raise AdapterError('parameters', 'target_classes 须为非空类别列表；省略时检测全部六类。')
        selected = set()
        for value in values:
            for name in ALIASES.get(value, (value,)):
                if name not in CLASSES:
                    raise AdapterError('unsupported_class', '该服务只支持现有 SODD 六类；管道包含 pipe 和 pipe_type2。')
                selected.add(name)
        return tuple(x for x in CLASSES if x in selected)

    def call(self, name, arguments):
        """Uniform secret-safe result envelope, including operation failures."""
        start = time.perf_counter()
        try:
            if name not in {'assess_image_quality', 'generate_exposure_candidate', 'detect_objects'}:
                raise AdapterError('unknown_tool', '工具不存在。')
            if not isinstance(arguments, dict):
                raise AdapterError('parameters', '工具参数须为对象。')
            data, summary, cached = getattr(self, '_' + name)(**arguments)
            result = {'ok': True, 'tool': name, 'summary': summary, 'data': data, 'cached': cached}
        except AdapterError as exc:
            result = {'ok': False, 'tool': name, 'summary': str(exc),
                      'error': {'category': exc.category, 'message': str(exc)}}
        except TypeError:
            result = {'ok': False, 'tool': name, 'summary': '工具参数不符合要求。',
                      'error': {'category': 'parameters', 'message': '工具参数不符合要求。'}}
        except Exception as exc:
            # Never include arbitrary third-party exception bodies (they can contain secrets).
            result = {'ok': False, 'tool': name, 'summary': '视觉计算或结果文件写入失败，已保留原图。',
                      'error': {'category': 'storage' if isinstance(exc, OSError) else 'computation',
                                'message': '视觉计算或结果文件写入失败。', 'exception_type': type(exc).__name__}}
        result['elapsed_ms'] = round((time.perf_counter() - start) * 1000, 2)
        safe = redact(result)
        # Logs hold identifiers and status only, never pixels or full request/response bodies.
        logging.getLogger('optical_mcp').info(json.dumps(redact({
            'tool': name, 'ok': safe['ok'], 'cached': safe.get('cached'),
            'image_id': safe.get('data', {}).get('image_id'),
            'error_category': safe.get('error', {}).get('category'), 'elapsed_ms': safe['elapsed_ms']})))
        return safe

    def _assess_image_quality(self, image_path=None, image_id=None):
        if (image_path is None) == (image_id is None):
            raise AdapterError('parameters', 'image_path 和 image_id 必须且只能提供一个。')
        if image_path is not None:
            image_id = self._register(image_path)
        with self._lease(image_id) as (ctx, internal):
            data, cached = self._run(ctx, 'quality', internal, 'assess_image_quality', {'image_id': internal})
            quality = data['quality']
            heat = self._save(ctx, image_id + '_heat', ctx.state.heatmaps[internal])
            path = str(ctx.source_path) if internal == 'original' else self._save(ctx, image_id, ctx.state.images[internal])
            return {'image_id': image_id, 'image_path': path, 'quality': quality, 'heatmap_path': heat,
                    'pixels_sha256': pixel_hash(ctx.state.images[internal]), 'configuration_sha256': ctx.state.config_hash,
                    'quality_status': 'quality_unassessable' if quality['quality_pass'] is None else
                                      'quality_pass' if quality['quality_pass'] else 'quality_failure'}, \
                '曝光评估已完成；这是图像质量状态，不是目标识别可靠性结论。', cached

    def _generate_exposure_candidate(self, image_id, method):
        if method not in {'gamma_only', 'local_bounded'}:
            raise AdapterError('parameters', 'method 只能是 gamma_only 或 local_bounded。')
        with self._lease(image_id) as (ctx, internal):
            if internal != 'original':
                raise AdapterError('prerequisite', '只能从原图生成候选，禁止继续增强候选图。')
            before = ctx.state.qualities.get('original')
            if before is None:
                raise AdapterError('prerequisite', '请先评估原图曝光。')
            if before['quality_pass'] is not True:
                raise AdapterError('quality_failure', '原图严重信息丢失或质量不可评估，停止生成校正候选。')
            data, cached = self._run(ctx, 'correction', 'original', 'generate_exposure_candidate',
                                     {'image_id': 'original', 'method': method})
            with self.registry_lock:
                candidate = self._public_id(ctx, data['image_id'])
            path = self._save(ctx, candidate, ctx.state.images[data['image_id']])
            return {'image_id': candidate, 'source_image_id': image_id, 'image_path': path,
                    'method': method, 'parameters': data['parameters'], 'applied': data['parameters']['applied'],
                    'pixels_sha256': data['pixels_sha256'], 'configuration_sha256': ctx.state.config_hash}, \
                '校正候选已生成。' if data['parameters']['applied'] else '本图无需调整，返回未调整候选。', cached

    def _detect_objects(self, image_id, target_classes=None):
        targets = self._targets(target_classes)
        with self._lease(image_id) as (ctx, internal):
            detector = ctx.state.detector
            if (not getattr(detector, 'model_available', True)
                    or tuple(getattr(detector, 'classes', ('background', *CLASSES))) != ('background', *CLASSES)):
                raise AdapterError('model_unavailable', '检测权重不可用；这不表示图中没有目标。')
            with self.gpu_lock:
                data, cached = self._run(ctx, 'detection', internal, 'detect_objects', {'image_id': internal}, targets)
            # The annotation and count use exactly the same program-filtered boxes.
            scope = hashlib.sha256(json.dumps(targets).encode()).hexdigest()[:12]
            view = self._save(ctx, image_id + '_boxes_' + scope,
                              _overlay(ctx.state.images[internal], data['detections'], []))
            return {'image_id': image_id, 'target_classes': list(targets), 'detections': data['detections'],
                    'count': data['count'], 'status': 'candidate_only', 'annotated_image_path': view,
                    'model': {'dataset': getattr(detector, 'dataset', 'unknown'),
                              'sha256': getattr(detector, 'model_sha256', None),
                              'threshold': float(detector.threshold), 'classes': list(CLASSES)},
                    'calibration': {'available': bool(detector.calibration),
                                    'status': copy.deepcopy(getattr(detector, 'calibration_status', None)),
                                    'warning': '模型分数及校准统计不等于当前图的可靠识别结论。'},
                    'configuration_sha256': ctx.state.config_hash}, \
                f'候选识别完成，共 {data["count"]} 个候选；尚未进行完整可靠性验收。', cached
