"""Detection metrics with explicit treatment of abstentions."""

from __future__ import annotations

import numpy as np


def iou(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def match_counts(predictions: list[list[dict]], targets: list[list[dict]], threshold: float,
                 overlap: float = 0.5) -> dict:
    tp = fp = fn = small_total = small_tp = 0
    for preds, truth in zip(predictions, targets):
        matched = set()
        for pred in sorted((p for p in preds if p["score"] >= threshold),
                           key=lambda p: p["score"], reverse=True):
            candidates = [(iou(pred["box"], target["box"]), j)
                          for j, target in enumerate(truth)
                          if j not in matched and pred["class_id"] == target["class_id"]]
            if candidates and max(candidates)[0] >= overlap:
                _, j = max(candidates)
                matched.add(j)
                tp += 1
            else:
                fp += 1
        fn += len(truth) - len(matched)
        for j, target in enumerate(truth):
            area = (target["box"][2] - target["box"][0]) * (target["box"][3] - target["box"][1])
            if area < 32 * 32:
                small_total += 1
                small_tp += int(j in matched)
    return {"tp": tp, "fp": fp, "fn": fn,
            "recall": tp / max(tp + fn, 1),
            "precision": tp / max(tp + fp, 1),
            "fp_per_image": fp / max(len(targets), 1),
            "small_recall": small_tp / small_total if small_total else None,
            "small_count": small_total}


def average_precision(predictions: list[list[dict]], targets: list[list[dict]],
                      class_id: int, overlap: float) -> float | None:
    total_truth = sum(sum(t["class_id"] == class_id for t in image) for image in targets)
    if not total_truth:
        return None
    ranked = sorted(((pred["score"], image_id, pred) for image_id, image in enumerate(predictions)
                     for pred in image if pred["class_id"] == class_id), reverse=True,
                    key=lambda item: item[0])
    used: set[tuple[int, int]] = set()
    tps, fps = [], []
    for _, image_id, pred in ranked:
        candidates = [(iou(pred["box"], truth["box"]), j)
                      for j, truth in enumerate(targets[image_id])
                      if truth["class_id"] == class_id and (image_id, j) not in used]
        if candidates and max(candidates)[0] >= overlap:
            _, j = max(candidates)
            used.add((image_id, j))
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)
    if not ranked:
        return 0.0
    recall = np.cumsum(tps) / total_truth
    precision = np.cumsum(tps) / np.maximum(np.cumsum(tps) + np.cumsum(fps), 1)
    # COCO style 101-point interpolation at a single IoU threshold.
    return float(np.mean([np.max(precision[recall >= r], initial=0) for r in np.linspace(0, 1, 101)]))


def summarize(predictions: list[list[dict]], targets: list[list[dict]],
              class_count: int, threshold: float = 0.35, max_fp_per_image: float = 1.0) -> dict:
    if len(predictions) != len(targets):
        raise ValueError("Prediction and target counts differ")
    counts = match_counts(predictions, targets, threshold)
    by_class = {}
    for class_id in range(1, class_count):
        p = [[item for item in image if item["class_id"] == class_id] for image in predictions]
        t = [[item for item in image if item["class_id"] == class_id] for image in targets]
        by_class[str(class_id)] = match_counts(p, t, threshold)
    # A fixed score grid keeps this sweep bounded for full test sets.
    thresholds = [float(x) for x in np.linspace(0.99, 0.01, 99)]
    if threshold not in thresholds:
        thresholds.append(float(threshold))
    feasible = [(t, match_counts(predictions, targets, t)) for t in thresholds]
    feasible = [(t, m) for t, m in feasible if m["fp_per_image"] <= max_fp_per_image]
    best = max(feasible, key=lambda item: (item[1]["recall"], -item[1]["fp_per_image"])) if feasible else None
    aps_50 = [average_precision(predictions, targets, c, 0.5) for c in range(1, class_count)]
    aps = [average_precision(predictions, targets, c, float(overlap))
           for overlap in np.arange(0.5, 1.0, 0.05) for c in range(1, class_count)]
    return {**counts, "by_class": by_class,
            "map50": float(np.mean([x for x in aps_50 if x is not None])) if any(x is not None for x in aps_50) else None,
            "map50_95": float(np.mean([x for x in aps if x is not None])) if any(x is not None for x in aps) else None,
            "recall_at_max_fp_per_image": None if best is None else {
                "fp_limit": max_fp_per_image, "threshold": best[0], **best[1]}}
