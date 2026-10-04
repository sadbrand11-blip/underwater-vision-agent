"""Fit an empirical score-to-box-precision curve on held-out validation images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

from data import DeepPCBDataset, NEUDataset, SODDDataset
from detector import build_model
from metrics import iou, summarize
from optical_agent.experiments.train_detector import evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-under", action="store_true",
                        help="Calibrate on clean and source-matched synthetic underexposure validation")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dataset = (NEUDataset(args.root, "val") if payload["dataset"] == "neu" else
               SODDDataset(args.root, "val") if payload["dataset"] == "sodd" else
               DeepPCBDataset(args.root, "val"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(payload["classes"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    predictions, targets, _ = evaluate_model(model, dataset, device, score_floor=0.01)
    if args.include_under:
        if payload["dataset"] not in {"neu", "sodd"}:
            raise ValueError("--include-under requires continuous-tone images")
        under = (NEUDataset(args.root, "val", exposure="under") if payload["dataset"] == "neu"
                 else SODDDataset(args.root, "val", exposure="under"))
        under_predictions, under_targets, _ = evaluate_model(model, under, device, score_floor=0.01)
        predictions += under_predictions
        targets += under_targets
        operating_point = summarize(predictions, targets, len(payload["classes"]))["recall_at_max_fp_per_image"]
        if operating_point:
            payload["threshold"] = operating_point["threshold"]
    scores, correctness = [], []
    for image_predictions, image_targets in zip(predictions, targets):
        used = set()
        for prediction in sorted(image_predictions, key=lambda item: item["score"], reverse=True):
            candidates = [(iou(prediction["box"], truth["box"]), j)
                          for j, truth in enumerate(image_targets)
                          if j not in used and prediction["class_id"] == truth["class_id"]]
            is_true = bool(candidates and max(candidates)[0] >= 0.5)
            if is_true:
                used.add(max(candidates)[1])
            scores.append(prediction["score"])
            correctness.append(int(is_true))
    if len(set(correctness)) < 2:
        raise RuntimeError("Validation predictions lack both true and false boxes; calibration unavailable")
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    calibrator.fit(scores, correctness)
    payload["calibration"] = {"method": "global_isotonic_precision_at_iou50",
                              "x": calibrator.X_thresholds_.tolist(),
                              "y": calibrator.y_thresholds_.tolist(),
                              "sample_count": len(scores),
                              "positive_fraction": float(np.mean(correctness)),
                              "source_split": "validation",
                              "conditions": ["clean", "synthetic_under"] if args.include_under else ["clean"],
                              "caution": "Box precision estimate on validation distribution; not an OOD guarantee"}
    torch.save(payload, args.output)
    report = {"checkpoint": str(args.output), "calibration": payload["calibration"]}
    Path(args.output).with_suffix(".calibration.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"sample_count": len(scores), "positive_fraction": float(np.mean(correctness)),
                      "output": args.output}))


if __name__ == "__main__":
    main()
