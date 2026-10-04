"""Evaluate real paired exposure images and optional external learned outputs.

Manifest CSV columns: scene_id, exposed, reference (optional), learned (optional).
All paths are relative to the manifest's directory unless absolute.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from data import read_rgb
from quality import assess, correct_exposure


def paired(a, b):
    if a.shape != b.shape:
        raise ValueError("Reference and candidate must be aligned and have the same dimensions")
    return {"psnr": float(peak_signal_noise_ratio(a, b, data_range=255)),
            "ssim": float(structural_similarity(a, b, channel_axis=2, data_range=255))}


def tensor(rgb):
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().unsqueeze(0) / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lpips", action="store_true", help="Requires optional lpips package and weights")
    parser.add_argument("--musiq", action="store_true", help="Requires optional pyiqa package and weights")
    args = parser.parse_args()
    lpips_metric = musiq_metric = None
    if args.lpips:
        import lpips
        lpips_metric = lpips.LPIPS(net="alex").eval()
    if args.musiq:
        import pyiqa
        musiq_metric = pyiqa.create_metric("musiq", device="cpu")
    manifest = Path(args.manifest)
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "exposed" not in rows[0]:
        raise ValueError("CSV must have an exposed column and at least one row")
    def resolve(value):
        candidate = Path(value)
        return candidate if candidate.is_absolute() else manifest.parent / candidate
    results = []
    for index, row in enumerate(rows):
        exposed = read_rgb(resolve(row["exposed"]))
        before, _ = assess(exposed)
        classical, correction = correct_exposure(exposed, before)
        after, _ = assess(classical)
        item = {"scene_id": row.get("scene_id") or str(index), "exposed": row["exposed"],
                "exposure_state": before["exposure_state"],
                "classical_method": correction["method"],
                "classical_quality_pass": after["quality_pass"],
                "original_lost_tile_fraction": before["lost_tile_fraction"],
                "classical_lost_tile_fraction": after["lost_tile_fraction"],
                "before": before["global"], "classical_after": after["global"]}
        if musiq_metric is not None:
            with torch.inference_mode():
                item["musiq"] = {"input": float(musiq_metric(tensor(exposed)).item()),
                                 "classical": float(musiq_metric(tensor(classical)).item())}
        if row.get("reference"):
            reference = read_rgb(resolve(row["reference"]))
            item["paired_reference"] = {"input": paired(reference, exposed),
                                        "classical": paired(reference, classical)}
            if lpips_metric is not None:
                with torch.inference_mode():
                    ref_tensor = tensor(reference) * 2 - 1
                    item["paired_reference"]["input"]["lpips"] = float(
                        lpips_metric(ref_tensor, tensor(exposed) * 2 - 1).item())
                    item["paired_reference"]["classical"]["lpips"] = float(
                        lpips_metric(ref_tensor, tensor(classical) * 2 - 1).item())
        if row.get("learned"):
            learned = read_rgb(resolve(row["learned"]))
            learned_quality, _ = assess(learned)
            item["learned_quality_pass"] = learned_quality["quality_pass"]
            item["learned_after"] = learned_quality["global"]
            if row.get("reference"):
                item["paired_reference"]["learned"] = paired(reference, learned)
                if lpips_metric is not None:
                    with torch.inference_mode():
                        item["paired_reference"]["learned"]["lpips"] = float(
                            lpips_metric(tensor(reference) * 2 - 1, tensor(learned) * 2 - 1).item())
            if musiq_metric is not None:
                with torch.inference_mode():
                    item["musiq"]["learned"] = float(musiq_metric(tensor(learned)).item())
        results.append(item)
    groups = defaultdict(list)
    for item in results:
        groups[item["scene_id"]].append(item)
    selection = {scene: min(items, key=lambda item: (
        item["original_lost_tile_fraction"], item["classical_lost_tile_fraction"],
        -float(item["classical_after"]["contrast"])))["exposed"] for scene, items in groups.items()}
    report = {"source": str(manifest), "image_count": len(results), "scene_count": len(groups),
              "real_or_simulated": "must_be_declared_by_manifest_provider",
              "reference_metrics_condition": "computed_only_when_aligned_reference_is_provided",
              "detection_benefit": "not_measured_without_defect_ground_truth",
              "selected_exposure_by_scene": selection, "images": results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"image_count": len(results), "scene_count": len(groups),
                      "output": str(Path(args.output).resolve())}))


if __name__ == "__main__":
    main()
