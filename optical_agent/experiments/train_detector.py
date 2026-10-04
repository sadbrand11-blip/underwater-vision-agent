"""Fine tune and evaluate the bundled TorchVision defect detector."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import DeepPCBDataset, NEUDataset, SODDDataset, collate
from detector import build_model
from metrics import summarize


def build_dataset(kind, root, split, limit=None, augment=False, exposure="none", exposure_augment=False):
    if kind == "neu":
        return NEUDataset(root, split, max_images=limit, augment=augment, exposure=exposure,
                          exposure_augment=exposure_augment)
    if kind == "sodd":
        return SODDDataset(root, split, max_images=limit, augment=augment, exposure=exposure,
                           exposure_augment=exposure_augment)
    if exposure != "none":
        raise ValueError("DeepPCB is preprocessed binary and is not valid for exposure simulation")
    return DeepPCBDataset(root, split, max_pairs=limit, augment=augment)


def target_dict(target) -> list[dict]:
    return [{"box": box.tolist(), "class_id": int(label)}
            for box, label in zip(target["boxes"], target["labels"])]


@torch.inference_mode()
def evaluate_model(model, dataset, device, batch_size=2, score_floor=0.01):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    predictions, targets, latency = [], [], []
    for images, batch_targets, _ in loader:
        start = time.perf_counter()
        outputs = model([image.to(device) for image in images])
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / len(images)
        for output, target in zip(outputs, batch_targets):
            predictions.append([{"box": box.tolist(), "score": float(score), "class_id": int(label)}
                                for box, score, label in zip(output["boxes"].cpu(), output["scores"].cpu(),
                                                             output["labels"].cpu()) if score >= score_floor])
            targets.append(target_dict(target))
            latency.append(elapsed)
    return predictions, targets, latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sodd", "neu", "deeppcb"], default="sodd")
    parser.add_argument("--root", required=True)
    parser.add_argument("--weights", required=True, help="Path to TorchVision COCO weights")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--resume", help="Continue from a fine-tuned checkpoint instead of COCO initialization")
    parser.add_argument("--exposure-augment", action="store_true")
    parser.add_argument("--validate-under", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(2026)
    np.random.seed(2026)
    random.seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = build_dataset(args.dataset, args.root, "train", args.max_train, augment=True,
                              exposure_augment=args.exposure_augment)
    val_set = build_dataset(args.dataset, args.root, "val", args.max_val)
    val_under = build_dataset(args.dataset, args.root, "val", args.max_val, exposure="under") if args.validate_under else None
    model = build_model(len(train_set.classes), None if args.resume else args.weights).to(device)
    if args.resume:
        previous = torch.load(args.resume, map_location="cpu", weights_only=False)
        if previous["classes"] != train_set.classes:
            raise ValueError("Resume checkpoint class list does not match dataset")
        model.load_state_dict(previous["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_score = -1.0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        epoch_start = time.perf_counter()
        for step, (images, targets, _) in enumerate(loader, 1):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss: {loss_dict}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if step % 50 == 0:
                print(json.dumps({"epoch": epoch + 1, "step": step, "steps": len(loader),
                                  "loss": round(float(np.mean(losses[-50:])), 4)}), flush=True)
        preds, truth, latency = evaluate_model(model, val_set, device, args.batch_size)
        report = summarize(preds, truth, len(train_set.classes), threshold=0.35)
        under_report = None
        if val_under is not None:
            under_preds, under_truth, _ = evaluate_model(model, val_under, device, args.batch_size)
            under_report = summarize(under_preds, under_truth, len(train_set.classes), threshold=0.35)
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                  "train_seconds": time.perf_counter() - epoch_start,
                  "validation": report, "validation_p95_ms": float(np.percentile(latency, 95) * 1000)}
        if under_report is not None:
            record["validation_under"] = under_report
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        fixed_fp = report["recall_at_max_fp_per_image"]
        score = (fixed_fp["recall"] if fixed_fp else 0.0) + (report["map50"] or 0.0) * 0.001
        if under_report is not None:
            under_fixed = under_report["recall_at_max_fp_per_image"]
            score = 0.5 * score + 0.5 * (under_fixed["recall"] if under_fixed else 0.0)
        if score > best_score:
            best_score = score
            torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "classes": train_set.classes, "dataset": args.dataset,
                        "threshold": report["recall_at_max_fp_per_image"]["threshold"]
                        if report["recall_at_max_fp_per_image"] else 0.35,
                        "validation": record, "training_samples": len(train_set),
                        "validation_samples": len(val_set)}, output)
        output.with_suffix(".history.json").write_text(
            json.dumps({"dataset": args.dataset, "train_samples": len(train_set),
                        "val_samples": len(val_set), "history": history}, ensure_ascii=False, indent=2),
            encoding="utf-8")


if __name__ == "__main__":
    main()
