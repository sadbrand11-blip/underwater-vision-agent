"""A small, explicit object detector adapter for DeepPCB."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from data import PCB_CLASSES


def build_model(class_count: int, coco_weights: str | Path | None = None):
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None)
    if coco_weights is not None:
        state = torch.load(str(coco_weights), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, class_count)
    return model


class TorchDetector:
    def __init__(self, checkpoint: str | Path, device: str | None = None):
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Detector checkpoint is missing: {checkpoint}")
        self.model_path = str(checkpoint.resolve())
        with checkpoint.open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        self.model_sha256 = digest.hexdigest()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        self.classes = payload.get("classes", PCB_CLASSES)
        self.dataset = payload.get("dataset", "unknown")
        self.validation = payload.get("validation", {})
        self.calibration = payload.get("calibration")
        self.calibration_status = payload.get('calibration_status')
        self.quality_config = payload.get("quality_config")
        self.model = build_model(len(self.classes))
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device).eval()
        self.threshold = float(payload.get("threshold", 0.35))

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray, threshold: float | None = None) -> list[dict]:
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().to(self.device) / 255.0
        prediction = self.model([tensor])[0]
        chosen = float(self.threshold if threshold is None else threshold)
        boxes = prediction["boxes"].detach().cpu().numpy()
        scores = prediction["scores"].detach().cpu().numpy()
        labels = prediction["labels"].detach().cpu().numpy()
        result = []
        for box, score, label in zip(boxes, scores, labels):
            if score < chosen:
                continue
            item = {
                "box": [round(float(v), 1) for v in box],
                "score": round(float(score), 4),
                "class_id": int(label),
                "class_name": self.classes[int(label)],
            }
            if self.calibration:
                item["estimated_box_precision"] = round(float(np.interp(
                    score, self.calibration["x"], self.calibration["y"])), 4)
            if self.calibration_status:
                item['calibration_status'] = self.calibration_status['class_status'].get(item['class_name'],'insufficient')
                if item['calibration_status'] == 'insufficient':
                    item['estimated_box_precision'] = 0.0
                item['model_sha256'] = self.model_sha256
                item['operating_threshold'] = self.threshold
            result.append(item)
        return result
