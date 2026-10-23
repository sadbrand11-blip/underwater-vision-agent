"""Lazy official COCO detector for ordinary photos; no download on import/use.

The detector keeps the official 91-slot prediction head, transform and category
IDs. This adapter does not reuse the six-class underwater head or calibration.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import threading

import numpy as np

from optical_agent.general_catalog import (COCO_CLASSES, COCO_ID_TO_NAME,
    ONTOLOGY_SHA256, ONTOLOGY_VERSION, PROFILE_ID)
from optical_agent.image_input import PREPROCESSING_VERSION
from quality import ExposureProfile, GENERAL_QUALITY_CONFIG

MODEL_FILENAME = 'fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth'
MODEL_SHA256 = '907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5'
MODEL_URL = 'https://download.pytorch.org/models/' + MODEL_FILENAME
MODEL_VERSION = 'torchvision_fasterrcnn_mobilenet_v3_large_320_fpn_coco_v1'
DEFAULT_DATA_ROOT = Path(os.environ.get('OPTICAL_AGENT_DATA_ROOT', r'D:\CodexData\optical_agent'))
DEFAULT_CHECKPOINT = Path(os.environ.get('OPTICAL_GENERAL_MODEL_PATH', str(DEFAULT_DATA_ROOT / 'pretrained' / 'downloads' / MODEL_FILENAME)))
DEFAULT_CALIBRATION = DEFAULT_DATA_ROOT / 'general_v060' / 'calibration.json'
DEFAULT_CALIBRATION_PATH = DEFAULT_CALIBRATION
_INFERENCE_LOCK = threading.RLock()


class ModelUnavailableError(RuntimeError):
    reason_code = 'model_unavailable'


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _curve(value):
    if not isinstance(value, dict):
        return None
    try:
        x, y = np.asarray(value['x'], dtype=float), np.asarray(value['y'], dtype=float)
        if (x.ndim != 1 or y.ndim != 1 or len(x) < 2 or len(x) != len(y)
                or not np.isfinite(x).all() or not np.isfinite(y).all()
                or np.any(np.diff(x) < 0) or np.any(x < 0) or np.any(x > 1)
                or np.any(y < 0) or np.any(y > 1)):
            return None
        return {'x': x.tolist(), 'y': y.tolist(), 'method': value.get('method', 'empirical_precision')}
    except (KeyError, TypeError, ValueError):
        return None


class GeneralDetector:
    classes = ('background', *COCO_CLASSES)
    dataset = 'coco2017'
    profile_id = PROFILE_ID
    ontology_version = ONTOLOGY_VERSION
    ontology_sha256 = ONTOLOGY_SHA256
    preprocessing_version = PREPROCESSING_VERSION

    def __init__(self, checkpoint=None, calibration_path=None, device=None):
        self.model_path = str(Path(checkpoint) if checkpoint is not None else DEFAULT_CHECKPOINT)
        self.model_sha256 = MODEL_SHA256
        self.device = device
        self.threshold = .35
        self.quality_config = asdict(GENERAL_QUALITY_CONFIG)
        self.exposure_profile = ExposureProfile(name='general_rgb_bounded_v1')
        self.calibration_path = Path(calibration_path) if calibration_path is not None else DEFAULT_CALIBRATION
        self.calibration = None
        self.calibration_status = {'status': 'unavailable', 'reason': '未进行通用模型校准',
            'class_status': {name: 'insufficient' for name in COCO_CLASSES}}
        self._class_calibration = {}
        self._model = self._weights = None
        self._model_error = None
        self._verified = False
        # No Torch import, CUDA initialization or checkpoint parsing here.
        self._load_calibration()
        self.exposure_policy = {'status': 'not_evaluated', 'selected_method': None,
                                'reason': '尚无通用开发集的校正采用结论'}
        policy_path = self.calibration_path.parent / 'exposure_policy.json'
        if policy_path.is_file():
            try:
                policy = json.loads(policy_path.read_text(encoding='utf8'))
                if (policy.get('schema_version') != 1 or policy.get('profile_id') != PROFILE_ID
                        or policy.get('model_sha256') != MODEL_SHA256
                        or policy.get('ontology_version') != ONTOLOGY_VERSION
                        or policy.get('selected_method') not in {'original', 'gamma_only', 'local_bounded'}):
                    raise ValueError('校正策略的通用模型或类别身份不匹配')
                self.exposure_policy = {'status': 'development_selected', 'selected_method': policy['selected_method'],
                    'reason': policy.get('reason'), 'source_sha256': _digest(policy_path)}
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                self.exposure_policy['reason'] = '校正策略不可用：' + str(exc)

    @property
    def model_available(self):
        return Path(self.model_path).is_file() and self._model_error is None

    @property
    def available(self):
        return self.model_available

    @property
    def unavailable_reason(self):
        if self._model_error:
            return self._model_error
        if not Path(self.model_path).is_file():
            return '未找到官方COCO检测权重；曝光评估和校正仍可使用'
        return None

    def _load_calibration(self):
        if not self.calibration_path.is_file():
            return
        try:
            payload = json.loads(self.calibration_path.read_text(encoding='utf-8'))
            if (payload.get('schema_version') != 1 or payload.get('profile_id') != PROFILE_ID
                    or payload.get('model_sha256') != MODEL_SHA256
                    or payload.get('ontology_version') != ONTOLOGY_VERSION
                    or payload.get('preprocessing_version') != PREPROCESSING_VERSION
                    or payload.get('calibration_scope') != 'coco_originals'):
                raise ValueError('校准文件的模型、配置或类别版本不匹配')
            if set(payload.get('classes', {})) - set(COCO_CLASSES):
                raise ValueError('校准文件包含当前COCO模型不支持的类别')
            statuses, thresholds = {}, {}
            for name in COCO_CLASSES:
                item = payload.get('classes', {}).get(name, {})
                state = item.get('status', payload.get('class_status', {}).get(name, 'insufficient'))
                curve = _curve(item.get('calibration', payload.get('calibration')))
                threshold = float(item.get('threshold', payload.get('class_thresholds', {}).get(name, self.threshold)))
                if not np.isfinite(threshold) or not 0 < threshold < 1:
                    raise ValueError('校准阈值必须在0和1之间')
                validated = state == 'validated' and curve is not None
                statuses[name] = 'validated' if validated else 'insufficient'
                thresholds[name] = threshold
                if validated:
                    self._class_calibration[name] = {'threshold': threshold, **curve}
            self.calibration = {'schema_version': 1, 'classes': self._class_calibration,
                                'source_sha256': _digest(self.calibration_path)} if self._class_calibration else None
            self.calibration_status = {'status': 'available' if self._class_calibration else 'insufficient',
                'class_status': statuses, 'class_thresholds': thresholds,
                'source_sha256': _digest(self.calibration_path),
                'reason': '仅样本充分且通过独立校准规则的类别可用于可靠性检查'}
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            self._class_calibration.clear()
            self.calibration = None
            self.calibration_status['reason'] = '通用校准不可用：' + str(exc)

    def _ensure_model(self):
        if self._model is not None:
            return
        if not self.model_available:
            raise ModelUnavailableError(self.unavailable_reason + '；不能解释为没有目标')
        try:
            if _digest(self.model_path) != MODEL_SHA256:
                raise ValueError('官方COCO权重SHA256不匹配，拒绝加载')
            self._verified = True
            import torch
            from torchvision.models.detection import (
                FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                fasterrcnn_mobilenet_v3_large_320_fpn)
            weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
            categories = weights.meta['categories']
            if len(categories) != 91 or any(categories[k] != v for k, v in COCO_ID_TO_NAME.items()):
                raise ValueError('安装的Torchvision类别映射与冻结COCO本体不一致')
            model = fasterrcnn_mobilenet_v3_large_320_fpn(
                weights=None, weights_backbone=None, num_classes=91)
            model.load_state_dict(torch.load(self.model_path, map_location='cpu', weights_only=True))
            self.device = torch.device(self.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
            self._model = model.to(self.device).eval()
            self._weights = weights
        except (OSError, RuntimeError, ImportError, ValueError) as exc:
            self._model_error = 'COCO检测模型不可用：' + str(exc)
            self._model = self._weights = None
            raise ModelUnavailableError(self._model_error) from exc

    def predict(self, rgb, threshold=None, target_classes=None):
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
            raise ValueError('检测输入必须是非空RGB uint8图像')
        chosen = float(self.threshold if threshold is None else threshold)
        if not np.isfinite(chosen) or not 0 <= chosen <= 1:
            raise ValueError('检测分数阈值必须在0到1之间')
        targets = set(COCO_CLASSES if target_classes is None else target_classes)
        if targets - set(COCO_CLASSES):
            raise ValueError('当前COCO模型不支持目标类别：' + ', '.join(sorted(targets - set(COCO_CLASSES))))
        with _INFERENCE_LOCK:
            self._ensure_model()
            import torch
            from PIL import Image
            with torch.inference_mode():
                tensor = self._weights.transforms()(Image.fromarray(rgb)).to(self.device)
                prediction = self._model([tensor])[0]
            result = []
            for box, score, label in zip(prediction['boxes'].detach().cpu().tolist(),
                                         prediction['scores'].detach().cpu().tolist(),
                                         prediction['labels'].detach().cpu().tolist()):
                name = COCO_ID_TO_NAME.get(label)
                if name not in targets:
                    continue
                fitted = self._class_calibration.get(name)
                minimum = fitted['threshold'] if threshold is None and fitted else chosen
                if score < minimum:
                    continue
                precision = float(np.interp(score, fitted['x'], fitted['y'])) if fitted else 0.0
                result.append({'box': [round(float(v), 1) for v in box], 'score': round(float(score), 6),
                    'class_id': int(label), 'category_id': int(label), 'class_name': name,
                    'estimated_box_precision': round(precision, 6),
                    'calibration_status': 'validated' if fitted else 'insufficient',
                    'operating_threshold': fitted['threshold'] if fitted else self.threshold,
                    'profile_id': PROFILE_ID, 'model_sha256': MODEL_SHA256,
                    'ontology_version': ONTOLOGY_VERSION, 'status': 'candidate_only'})
            return result

    def metadata(self):
        return {'profile_id': PROFILE_ID, 'dataset': self.dataset, 'model_version': MODEL_VERSION,
            'exposure_policy': self.exposure_policy,
            'model_sha256': MODEL_SHA256, 'weight_source': MODEL_URL,
            'weight_sha256_verified': self._verified, 'available': self.model_available,
            'unavailable_reason': self.unavailable_reason, 'loaded': self._model is not None,
            'ontology_version': ONTOLOGY_VERSION, 'ontology_sha256': ONTOLOGY_SHA256,
            'category_count': len(COCO_CLASSES), 'category_id_policy': 'official_COCO_91_slots_no_remapping',
            'preprocessing_version': PREPROCESSING_VERSION,
            'preprocessing': 'official_COCO_V1_weights.transforms_then_model_GeneralizedRCNNTransform',
            'threshold': self.threshold, 'calibration_status': self.calibration_status,
            'quality_profile': self.quality_config['calibration_label'],
            'exposure_profile': asdict(self.exposure_profile)}
