"""Evaluate a detector on official or reserved images without exposure claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import DeepPCBDataset, NEUDataset, SODDDataset
from detector import build_model
from metrics import summarize
from train_detector import evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max-sources", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dataset = (NEUDataset(args.root, args.split, max_images=args.max_sources)
               if payload["dataset"] == "neu" else
               SODDDataset(args.root, args.split, max_images=args.max_sources)
               if payload["dataset"] == "sodd" else
               DeepPCBDataset(args.root, args.split, max_pairs=args.max_sources))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(payload["classes"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    predictions, targets, latencies = evaluate_model(model, dataset, device)
    metrics = summarize(predictions, targets, len(payload["classes"]), payload.get("threshold", 0.35))
    report = {"dataset": payload["dataset"], "split": args.split,
              "source_count": len(dataset) if payload["dataset"] != "deeppcb" else len(dataset) // 2,
              "image_count": len(dataset), "threshold": payload.get("threshold", 0.35),
              "classes": payload["classes"], "metrics": metrics,
              "latency_p95_ms": float(np.percentile(latencies, 95) * 1000),
              "latency_unit_note": "per-image model time; excludes quality assessment",
              "metric_policy": "Primary recall and FP/image use validation-selected checkpoint threshold; test-set threshold sweep is diagnostic only"}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"recall": metrics["recall"], "map50": metrics["map50"],
                      "fp_per_image": metrics["fp_per_image"], "image_count": len(dataset)}))


if __name__ == "__main__":
    main()
