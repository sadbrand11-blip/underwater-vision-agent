"""Tune conservative exposure triggers on source-disjoint validation images.

Synthetic JPEG-domain exposure is only a stress signal for trigger selection.
The resulting thresholds are not a real-camera exposure calibration.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch

from data import NEUDataset, read_rgb, simulate_exposure
from quality import QualityConfig


def features(rgb, grid=8):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    values = []
    for row in range(grid):
        for col in range(grid):
            tile = gray[height * row // grid: height * (row + 1) // grid,
                        width * col // grid: width * (col + 1) // grid]
            values.append([np.mean(tile <= 12), np.mean(tile >= 250), np.median(tile) / 255])
    return np.asarray(values, dtype=np.float32)


def states(feat, dfrac, bfrac, dmed, bmed, share):
    dark = np.mean((feat[:, :, 0] > dfrac) | (feat[:, :, 2] < dmed), axis=1) > share
    bright = np.mean((feat[:, :, 1] > bfrac) | (feat[:, :, 2] > bmed), axis=1) > share
    extreme_mixed = ((np.sum(feat[:, :, 2] < 0.15, axis=1) >= 3)
                     & (np.sum(feat[:, :, 2] > 0.95, axis=1) >= 3))
    result = dark.astype(int) + bright.astype(int) * 2
    result[extreme_mixed] = 3
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dataset = NEUDataset(args.root, "val")
    arrays = {mode: [] for mode in ("none", "under", "over", "mixed")}
    for image_path, _ in dataset.items:
        clean = read_rgb(image_path)
        for mode in arrays:
            arrays[mode].append(features(simulate_exposure(clean, mode)))
    arrays = {mode: np.stack(items) for mode, items in arrays.items()}
    best = None
    for dfrac, bfrac, dmed, bmed, share in itertools.product(
            [0.3, 0.5, 0.7], [0.2, 0.4, 0.6],
            [0.06, 0.08, 0.10, 0.12, 0.14],
            [0.86, 0.90, 0.94, 0.97], [0.10, 0.15, 0.20, 0.25]):
        result = {mode: states(items, dfrac, bfrac, dmed, bmed, share)
                  for mode, items in arrays.items()}
        false_trigger = float(np.mean(result["none"] != 0))
        if false_trigger > 0.05:
            continue
        recalls = {"under": float(np.mean((result["under"] & 1) > 0)),
                   "over": float(np.mean((result["over"] & 2) > 0)),
                   "mixed": float(np.mean(result["mixed"] == 3))}
        score = float(np.mean(list(recalls.values()))) - false_trigger * 0.2
        item = (score, -false_trigger, dfrac, bfrac, dmed, bmed, share, recalls)
        if best is None or item[:2] > best[:2]:
            best = item
    if best is None:
        raise RuntimeError("No quality trigger satisfied the 5% clean-image false-trigger cap")
    _, neg_false, dfrac, bfrac, dmed, bmed, share, recalls = best
    config = QualityConfig(dark_fraction_warn=dfrac, bright_fraction_warn=bfrac,
                           dark_median_warn=dmed, bright_median_warn=bmed,
                           exposure_tile_share_warn=share,
                           calibration_label="neu_validation_synthetic_trigger_v1")
    cfg = asdict(config)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    payload["quality_config"] = cfg
    payload["quality_calibration"] = {"source_split": "NEU_validation", "source_count": len(dataset),
                                      "stress_type": "synthetic_JPEG_domain",
                                      "clean_false_trigger_rate": -neg_false,
                                      "stress_recall": recalls,
                                      "not_real_camera_calibration": True}
    torch.save(payload, args.output)
    report = {"quality_config": cfg, "calibration": payload["quality_calibration"]}
    Path(args.output).with_suffix(".quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
