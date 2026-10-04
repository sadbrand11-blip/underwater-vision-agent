"""Evaluate original, corrected, and synthetic exposure conditions separately."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from agent import OpticalAgent
from data import NEUDataset, SODDDataset, simulate_exposure
from detector import TorchDetector
from metrics import match_counts, summarize


def truth_from_target(target):
    return [{"box": box.tolist(), "class_id": int(label)}
            for box, label in zip(target["boxes"], target["labels"])]


def assess_exposure(clean, exposed, corrected):
    return {"input_psnr": float(peak_signal_noise_ratio(clean, exposed, data_range=255)),
            "corrected_psnr": float(peak_signal_noise_ratio(clean, corrected, data_range=255)),
            "input_ssim": float(structural_similarity(clean, exposed, channel_axis=2, data_range=255)),
            "corrected_ssim": float(structural_similarity(clean, corrected, channel_axis=2, data_range=255))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Dataset directory")
    parser.add_argument("--dataset", choices=["sodd", "neu"], default="sodd")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--modes", nargs="+", choices=["none", "under", "over", "mixed"],
                        default=["none", "under", "over", "mixed"])
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--multi-frame", action="store_true",
                        help="Also evaluate aligned synthetic under/normal/over stacks")
    args = parser.parse_args()
    dataset = (SODDDataset(args.root, args.split, max_images=args.max_images)
               if args.dataset == "sodd" else NEUDataset(args.root, args.split, max_images=args.max_images))
    detector = TorchDetector(args.model, device=args.device)
    agent = OpticalAgent(detector)
    result = {"dataset": args.dataset, "split": args.split, "source_count": len(dataset),
              "exposure_type": "synthetic_JPEG_domain_not_real_camera", "model": str(args.model),
              "device": str(detector.device), "conditions": {}}
    if args.dataset == "sodd":
        result["split_caveat"] = ("Official SODD video-frame splits are retained; adjacent frames may cross splits. "
                                  "This is a pilot, not an independent scene generalization estimate.")
    result["metric_policy"] = ("The primary recall and FP/image use the checkpoint threshold selected on validation. "
                               "The recall_at_max_fp_per_image sweep inside each condition is diagnostic only; "
                               "do not select a new test threshold from it.")
    for mode in args.modes:
        raw_preds, corrected_preds, selected_preds, safe_preds, targets, latencies = [], [], [], [], [], []
        statuses, psnr, ssim, exposure_before, exposure_after = [], [], [], [], []
        new_fp = 0
        correction_accepted = 0
        for index in range(len(dataset)):
            tensor, target, _ = dataset[index]
            clean = np.uint8(np.round(tensor.permute(1, 2, 0).numpy() * 255))
            exposed = simulate_exposure(clean, mode)
            report, images = agent.inspect([exposed])
            truth = truth_from_target(target)
            raw_preds.append(report["detections_original"])
            corrected_preds.append(report["detections_corrected"])
            selected_preds.append(report["detections_selected"])
            safe_preds.append(report["accepted_detections"] if report["status"] == "reliable" else [])
            correction_accepted += int(report["correction"]["accepted"])
            targets.append(truth)
            latencies.append(report["latency_ms"])
            statuses.append(report["status"])
            exposure_before.append(report["quality_before"]["global"])
            exposure_after.append(report["quality_after"]["global"])
            if mode != "none":
                fidelity = assess_exposure(clean, exposed, images["corrected"])
                psnr.append([fidelity["input_psnr"], fidelity["corrected_psnr"]])
                ssim.append([fidelity["input_ssim"], fidelity["corrected_ssim"]])
            # Count additional false detections after correction at the chosen threshold.
            def false_count(preds):
                return match_counts([preds], [truth], detector.threshold)["fp"]
            new_fp += max(0, false_count(report["detections_corrected"]) - false_count(report["detections_original"]))
        condition = {"original_detection": summarize(raw_preds, targets, len(detector.classes), detector.threshold),
                     "corrected_detection": summarize(corrected_preds, targets, len(detector.classes), detector.threshold),
                     "selected_detection": summarize(selected_preds, targets, len(detector.classes), detector.threshold),
                     "end_to_end_reliable_only": summarize(safe_preds, targets, len(detector.classes), detector.threshold),
                     "correction_accept_rate": correction_accepted / len(dataset),
                     "status_counts": {key: statuses.count(key) for key in sorted(set(statuses))},
                     "unreliable_rate": float(np.mean([s != "reliable" for s in statuses])),
                     "new_false_detections_after_correction": int(new_fp),
                     "new_fp_definition": "sum_of_positive_per_image_fp_increase_one_to_one_iou50",
                     "latency_p95_ms": float(np.percentile(latencies, 95)),
                     "quality_metric_means": {key: {"before": float(np.mean([x[key] for x in exposure_before])),
                                                    "after": float(np.mean([x[key] for x in exposure_after]))}
                                              for key in ["dark_fraction", "bright_fraction", "contrast",
                                                          "sharpness", "noise_estimate"]}}
        if mode != "none":
            condition["paired_reference_metrics"] = {
                "psnr_input": float(np.mean([x[0] for x in psnr])),
                "psnr_corrected": float(np.mean([x[1] for x in psnr])),
                "ssim_input": float(np.mean([x[0] for x in ssim])),
                "ssim_corrected": float(np.mean([x[1] for x in ssim])),
                "reference": "unperturbed_source_image"}
        result["conditions"][mode] = condition
        print(json.dumps({"mode": mode, "raw_recall": condition["original_detection"]["recall"],
                          "corrected_recall": condition["corrected_detection"]["recall"],
                          "reliable_only_recall": condition["end_to_end_reliable_only"]["recall"],
                          "unreliable_rate": condition["unreliable_rate"]}), flush=True)
    if args.multi_frame:
        fused_preds, targets, statuses, latencies = [], [], [], []
        for index in range(len(dataset)):
            tensor, target, _ = dataset[index]
            clean = np.uint8(np.round(tensor.permute(1, 2, 0).numpy() * 255))
            frames = [simulate_exposure(clean, mode) for mode in ("under", "none", "over")]
            report, _ = agent.inspect(frames)
            fused_preds.append(report["accepted_detections"] if report["status"] == "reliable" else [])
            targets.append(truth_from_target(target))
            statuses.append(report["status"])
            latencies.append(report["latency_ms"])
        result["conditions"]["multi_frame_synthetic"] = {
            "end_to_end_reliable_only": summarize(fused_preds, targets, len(detector.classes), detector.threshold),
            "status_counts": {key: statuses.count(key) for key in sorted(set(statuses))},
            "unreliable_rate": float(np.mean([s != "reliable" for s in statuses])),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "alignment": "exact_by_generation_not_a_real_camera_stack"}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
