"""Measure exposure decisions on real MOUD lamp-lighting levels.

The three folders are real illumination conditions, but not registered image
pairs or ground-truth camera exposures. This script therefore reports quality
and correction behavior without PSNR or detection claims.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import cv2

from data import read_rgb
from quality import UNDERWATER_QUALITY_CONFIG, assess, correct_exposure


def summarize(records: list[dict]) -> dict:
    if not records:
        return {"count": 0}
    def mean(name: str) -> float:
        return round(float(np.mean([item[name] for item in records])), 4)
    return {
        "count": len(records),
        "exposure_states": dict(Counter(item["exposure_state"] for item in records)),
        "correction_applied_rate": mean("correction_applied"),
        "quality_failure_rate_before": mean("failure_before"),
        "quality_failure_rate_after": mean("failure_after"),
        "median_luminance_before": mean("median_before"),
        "median_luminance_after": mean("median_after"),
        "dark_fraction_before": mean("dark_before"),
        "dark_fraction_after": mean("dark_after"),
        "bright_fraction_before": mean("bright_before"),
        "bright_fraction_after": mean("bright_after"),
        "contrast_before": mean("contrast_before"),
        "contrast_after": mean("contrast_after"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"D:\CodexData\optical_agent\moud\Scene_1\Images"))
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for light in ("Low light", "Mid light", "High light"):
        for image_path in sorted((args.root / light).rglob("*.jpg")):
            rgb = read_rgb(image_path)
            if max(rgb.shape[:2]) > args.max_side:
                scale = args.max_side / max(rgb.shape[:2])
                rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            before, _ = assess(rgb, UNDERWATER_QUALITY_CONFIG)
            corrected, decision = correct_exposure(rgb, before)
            after, _ = assess(corrected, UNDERWATER_QUALITY_CONFIG)
            records.append({
                "file": str(image_path), "illumination": light,
                "area": image_path.parent.name.rsplit("_", 1)[-1],
                "exposure_state": before["exposure_state"],
                "correction_applied": int(decision["applied"]),
                "failure_before": int(before["quality_pass"] is False),
                "failure_after": int(after["quality_pass"] is False),
                "median_before": before["global"]["median"],
                "median_after": after["global"]["median"],
                "dark_before": before["global"]["dark_fraction"],
                "dark_after": after["global"]["dark_fraction"],
                "bright_before": before["global"]["bright_fraction"],
                "bright_after": after["global"]["bright_fraction"],
                "contrast_before": before["global"]["contrast"],
                "contrast_after": after["global"]["contrast"],
            })
    if not records:
        raise FileNotFoundError(f"No MOUD images found in {args.root}")
    report = {
        "dataset": "MOUD Scene_1 range-downloaded subset",
        "real_condition": "controlled_underwater_lamp_illumination",
        "not_camera_exposure_time": True,
        "not_registered_pairs": True,
        "quality_thresholds": UNDERWATER_QUALITY_CONFIG.calibration_label,
        "analysis_max_side": args.max_side,
        "groups": {light: summarize([item for item in records if item["illumination"] == light])
                   for light in ("Low light", "Mid light", "High light")},
        "by_area": {area: {light: summarize([item for item in records
                                               if item["area"] == area and item["illumination"] == light])
                           for light in ("Low light", "Mid light", "High light")}
                    for area in sorted({item["area"] for item in records})},
        "images": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["groups"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
