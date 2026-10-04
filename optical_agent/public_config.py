"""Public delivery paths, independent of private experiment caches."""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get('OPTICAL_AGENT_DATA_ROOT', r'D:\CodexData\optical_agent')).expanduser().resolve()
PUBLIC_ROOT = DATA_ROOT / 'public'
MODEL_PATH = Path(os.environ.get('OPTICAL_AGENT_MODEL_PATH', str(PUBLIC_ROOT / 'models/sodd_detector.pt')))


class ModelUnavailable(RuntimeError):
    reason_code = 'model_unavailable'


class UnavailableDetector:
    dataset, threshold, calibration = 'sodd', .35, None
    classes = ('background', 'propeller', 'pipe_type2', 'red_fin', 'net', 'qr_codes', 'pipe')
    model_available = False

    def predict(self, image, **kwargs):
        raise ModelUnavailable('Detection unavailable: start python run_demo.py --with-detector. No detections were computed.')

    def metadata(self):
        return {'available': False, 'dataset': 'sodd', 'status': 'model_unavailable', 'candidate_only': True}
