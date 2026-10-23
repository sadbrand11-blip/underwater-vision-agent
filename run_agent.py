"""Run the offline agent on one image or a registered exposure stack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from optical_agent.vision_delivery import DeliveryOpticalAgent
from optical_agent.vision_data import ROOT
from data import save_rgb
from optical_agent.general_vision import GeneralDetector, DEFAULT_CHECKPOINT
from optical_agent.image_input import read_image_bytes


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="Single image or aligned exposure frames")
    parser.add_argument("--model", default=str(DEFAULT_CHECKPOINT), help="Official verified COCO checkpoint")
    parser.add_argument("--output", default=str(ROOT / 'general_v060' / 'cli_latest'))
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    detector = GeneralDetector(args.model, device=args.device)
    report, images = DeliveryOpticalAgent(detector).inspect([read_image_bytes(Path(path).read_bytes()) for path in args.images])
    for name, image in images.items():
        save_rgb(output / f"{name}.png", image)
    report["input_files"] = [str(Path(path).resolve()) for path in args.images]
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "reasons": report["reasons"],
                      "accepted_detections": report["accepted_detections"],
                      "tentative_detections": report["tentative_detections"],
                      "candidate_detections": report["detections_corrected"],
                      "output": str(output.resolve())},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
