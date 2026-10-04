"""Perception → exposure correction → object detection → reliability feedback."""

from __future__ import annotations

import time
from typing import Protocol

import cv2
import numpy as np

from metrics import iou
from quality import ExposureProfile, QualityConfig, UNDERWATER_QUALITY_CONFIG, assess, correct_exposure


class Detector(Protocol):
    threshold: float

    def predict(self, rgb: np.ndarray, threshold: float | None = None) -> list[dict]: ...


def _high_confidence(boxes: list[dict], threshold: float) -> list[dict]:
    return [box for box in boxes if box["score"] >= box.get('operating_threshold', threshold)
            and box.get("estimated_box_precision", 1.0) >= 0.5]


def _consistent(a: list[dict], b: list[dict], overlap: float = 0.3) -> bool:
    if not a and not b:
        return True
    if not a or not b:
        return False
    def covered(left, right):
        return all(any(x["class_id"] == y["class_id"] and iou(x["box"], y["box"]) >= overlap
                       for y in right) for x in left)
    return covered(a, b) and covered(b, a)


def _lost_boxes(quality: dict) -> list[list[int]]:
    return [tile["box"] for tile in quality["tiles"] if tile["information_loss"]]


def _box_intersects(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _overlay(rgb: np.ndarray, boxes: list[dict], lost: list[list[int]]) -> np.ndarray:
    output = rgb.copy()
    for x0, y0, x1, y1 in lost:
        cv2.rectangle(output, (x0, y0), (x1 - 1, y1 - 1), (240, 60, 50), 2)
    for detection in boxes:
        x0, y0, x1, y1 = [int(round(v)) for v in detection["box"]]
        color = (30, 230, 50) if detection.get("estimated_box_precision", 1.0) >= 0.5 else (245, 155, 30)
        cv2.rectangle(output, (x0, y0), (x1, y1), color, 2)
        label = f'{detection["class_name"]} {detection["score"]:.2f}'
        cv2.putText(output, label, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, cv2.LINE_AA)
    return output


class OpticalAgent:
    def __init__(self, detector: Detector):
        self.detector = detector
        self.exposure_profile = getattr(detector, 'exposure_profile', None) or ExposureProfile()
        self.quality_config = (QualityConfig(**detector.quality_config)
                               if getattr(detector, "quality_config", None) else
                               UNDERWATER_QUALITY_CONFIG if getattr(detector, "dataset", None) == "sodd" else
                               QualityConfig())

    def _one(self, rgb: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
        start = time.perf_counter()
        before, raw_heat = assess(rgb, self.quality_config)
        corrected, correction = correct_exposure(rgb, before, self.exposure_profile)
        after, corrected_heat = assess(corrected, self.quality_config)
        raw_boxes = self.detector.predict(rgb)
        corrected_boxes = raw_boxes if not correction["applied"] else self.detector.predict(corrected)
        return compare_evidence(self.detector, rgb, corrected, before, after, correction, raw_boxes, corrected_boxes, raw_heat, corrected_heat, start)

    def inspect(self, images: list[np.ndarray]) -> tuple[dict, dict[str, np.ndarray]]:
        if not images:
            raise ValueError("At least one image is required")
        if any(image.shape != images[0].shape for image in images):
            raise ValueError("Multi-exposure images must have identical dimensions and be aligned")
        runs = [self._one(image) for image in images]
        if len(runs) == 1:
            return runs[0]
        # Evidence fusion is conservative: choose the clearest valid exposure,
        # and require object-box agreement across valid frames.
        valid = [(index, report) for index, (report, _) in enumerate(runs)
                 if report["quality_selected"]["quality_pass"] is True]
        pool = valid if valid else [(index, report) for index, (report, _) in enumerate(runs)]
        selected, report = min(pool, key=lambda item: (item[1]["quality_selected"]["lost_tile_fraction"],
                                                       -item[1]["quality_selected"]["global"]["contrast"]))
        output = dict(report)
        output["multi_frame"] = {
            "frame_count": len(images), "selected_frame": selected,
            "frame_statuses": [r["status"] for r, _ in runs],
            "frame_summaries": [{"index": index, "status": r["status"],
                                  "exposure_state": r["quality_before"]["exposure_state"],
                                  "lost_tile_fraction_before": r["quality_before"]["lost_tile_fraction"],
                                  "lost_tile_fraction_selected": r["quality_selected"]["lost_tile_fraction"],
                                  "accepted_detection_count": len(r["accepted_detections"])}
                                 for index, (r, _) in enumerate(runs)],
            "mode": "select_frame_and_cross_check_detection_evidence",
            "alignment": "assumed_by_input_contract_not_geometrically_verified"}
        accepted = [(index, r) for index, r in valid if r["status"] == "reliable"]
        stable = len(accepted) >= 1 and all(
            _consistent(accepted[0][1]["accepted_detections"],
                        other["accepted_detections"])
            for _, other in accepted[1:])
        if not stable:
            output["status"] = "unreliable" if valid else "quality_failure"
            output["target_conclusion"] = "unreliable"
            if output["defect_conclusion"] is not None:
                output["defect_conclusion"] = "unreliable"
            output["next_action"] = "provide_more_aligned_exposure_frames"
            output["reasons"] = list(output["reasons"]) + ["没有通过质量检查的有效帧，或有效帧的检测相互冲突"]
        else:
            output["multi_frame"]["consistent_frames"] = len(accepted)
            if output["status"] != "reliable":
                selected = accepted[0][0]
                output = dict(runs[selected][0])
                output["multi_frame"] = {"frame_count": len(images), "selected_frame": selected,
                                         "frame_statuses": [r["status"] for r, _ in runs],
                                         "frame_summaries": [{"index": index, "status": r["status"],
                                                               "exposure_state": r["quality_before"]["exposure_state"],
                                                               "lost_tile_fraction_before": r["quality_before"]["lost_tile_fraction"],
                                                               "lost_tile_fraction_selected": r["quality_selected"]["lost_tile_fraction"],
                                                               "accepted_detection_count": len(r["accepted_detections"])}
                                                              for index, (r, _) in enumerate(runs)],
                                         "consistent_frames": len(accepted),
                                         "mode": "select_frame_and_cross_check_detection_evidence",
                                         "alignment": "assumed_by_input_contract_not_geometrically_verified"}
        output["latency_ms"] = round(sum(r["latency_ms"] for r, _ in runs), 1)
        return output, runs[selected][1]


def compare_evidence(detector, rgb, corrected, before, after, correction, raw_boxes, corrected_boxes, raw_heat, corrected_heat, start=None):
    """Apply the same deterministic gate to already computed observations."""
    import copy
    correction = copy.deepcopy(correction)
    start = time.perf_counter() if start is None else start
    threshold = detector.threshold
    raw_high = _high_confidence(raw_boxes, threshold)
    corrected_high = _high_confidence(corrected_boxes, threshold)
    selected_variant = "corrected" if correction["applied"] else "original"
    if correction["applied"] and before["quality_pass"] is True:
        raw_score = sum(d.get("estimated_box_precision", d["score"]) for d in raw_high)
        corrected_score = sum(d.get("estimated_box_precision", d["score"]) for d in corrected_high)
        if after["quality_pass"] is False or corrected_score + 0.02 < raw_score:
            selected_variant = "original"
            correction["selection_reason"] = "原图质量可用，校正候选使质量或检测证据下降"
    correction["selected_as_candidate"] = selected_variant == "corrected" and correction["applied"]
    selected_boxes = corrected_boxes if selected_variant == "corrected" else raw_boxes
    accepted_boxes = corrected_high if selected_variant == "corrected" else raw_high
    tentative_boxes = [d for d in selected_boxes if d not in accepted_boxes]
    original_lost = _lost_boxes(before)
    corrected_lost = _lost_boxes(after)
    reasons = []
    domain = getattr(detector, "dataset", "unknown")
    if domain == "neu" and (rgb.shape[:2] != (200, 200) or
                             float(np.mean(np.max(rgb, axis=2) - np.min(rgb, axis=2))) > 3):
        reasons.append("输入与 NEU-DET 灰度钢材训练域不符，属于待验证分布外样本")
    if domain == "deeppcb" and rgb.shape[:2] != (640, 640):
        reasons.append("输入尺寸与 DeepPCB 训练域不符，属于待验证分布外样本")
    if before["exposure_state"] == "preprocessed_binary":
        reasons.append("输入已接近二值化，原始曝光质量无法评估")
    if selected_variant == "corrected" and after["quality_pass"] is False:
        reasons.append("校正后仍存在大面积不可辨区域")
    if before["quality_pass"] is False:
        reasons.append("原图存在大面积饱和或暗部信息丢失，单帧不能恢复真实纹理")
    if any(_box_intersects(d["box"], lost) for d in accepted_boxes for lost in original_lost):
        reasons.append("检测框与原图不可恢复的信息丢失区域重叠")
    if not _consistent(raw_high, corrected_high):
        reasons.append("原图与校正图的高分检测结果不一致")
    if not accepted_boxes:
        reasons.append("未检出达到验证集阈值的目标；不能据此证明画面中没有目标")
    status = "unreliable" if reasons else "reliable"
    if (selected_variant == "corrected" and after["quality_pass"] is False) or before["quality_pass"] is False:
        status = "quality_failure"
    if before["exposure_state"] == "preprocessed_binary":
        status = "quality_unassessable"
    if detector.threshold <= 0 or detector.threshold >= 1:
        reasons.append("检测阈值无效")
        status = "unreliable"
    correction["accepted"] = correction["selected_as_candidate"] and status == "reliable"
    next_action = ("accept_result" if status == "reliable" else
                   "provide_unprocessed_image" if status == "quality_unassessable" else
                   "provide_aligned_alternative_exposure" if status == "quality_failure" else
                   "inspect_original_and_corrected_or_add_exposure_frame")
    if status == "reliable" and tentative_boxes:
        next_action = "provide_aligned_alternative_exposure_for_tentative_boxes"
    elapsed = (time.perf_counter() - start) * 1000
    conclusion = ("unreliable" if status != "reliable" else
                  "automatic_with_tentative_candidates" if tentative_boxes else "automatic")
    result = {
        "status": status,
        "target_conclusion": conclusion,
        "defect_conclusion": conclusion if domain in {"neu", "deeppcb"} else None,
        "next_action": next_action,
        "reasons": reasons,
        "quality_before": before,
        "quality_after": after,
        "quality_selected": after if selected_variant == "corrected" else before,
        "correction": correction,
        "detections_original": raw_boxes,
        "detections_corrected": corrected_boxes,
        "detections_selected": selected_boxes,
        "accepted_detections": accepted_boxes,
        "tentative_detections": tentative_boxes,
        "selected_image": selected_variant,
        "irrecoverable_regions": original_lost,
        "remaining_problem_regions": corrected_lost,
        "detector_threshold": threshold,
        "detector_training_domain": domain,
        "detector_checkpoint_sha256": getattr(detector, "model_sha256", None),
        "detector_calibration_status": getattr(detector, 'calibration_status', None),
        "detector_models": detector.metadata() if hasattr(detector,'metadata') else None,
        "detector_score_warning": (
            "estimated_box_precision来自验证集全局等距回归，不保证分布外样本可靠"
            if getattr(detector, "calibration", None)
            else "检测分数是模型排序分数，未经概率校准"),
        "latency_ms": round(elapsed, 1),
    }
    images = {"original": rgb, "corrected": corrected,
              "selected": corrected if selected_variant == "corrected" else rgb,
              "quality_heatmap_before": raw_heat, "quality_heatmap_after": corrected_heat,
              "detections_original": _overlay(rgb, raw_boxes, original_lost),
              "detections_corrected": _overlay(corrected, corrected_boxes, corrected_lost),
              "detections_selected": _overlay(corrected if selected_variant == "corrected" else rgb,
                                              accepted_boxes, corrected_lost if selected_variant == "corrected" else original_lost)}
    return result, images

