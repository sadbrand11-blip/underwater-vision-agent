"""Run a small, local example without a camera or API key."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from optical_agent.tools import ToolContext, execute_tool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/tool_layer_demo.json")
    parser.add_argument("--image", help="Use a real local image and the SODD detector")
    parser.add_argument("--model", default="models/sodd_detector.pt")
    args = parser.parse_args()
    image = np.tile(np.arange(20, 148, dtype=np.uint8), (128, 1))
    image = np.repeat(image[:, :, None], 3, axis=2)
    detector = SimpleNamespace(dataset='sodd', threshold=0.33)
    if args.image:
        from data import read_rgb
        from detector import TorchDetector
        image = read_rgb(args.image)
        detector = TorchDetector(args.model)
    context = ToolContext(detector, image)
    actions = [('assess_image_quality', {'image_id': 'original'}),
               ('correct_image_exposure', {'image_id': 'original'}),
               ('assess_image_quality', {'image_id': 'corrected'})]
    if args.image:
        actions.extend([('detect_objects', {'image_id': 'original'}),
                        ('detect_objects', {'image_id': 'corrected'}),
                        ('compare_detection_evidence', {'original_image_id': 'original', 'corrected_image_id': 'corrected'})])
    observations = []
    for tool, arguments in actions:
        result = execute_tool(context, tool, arguments)
        if not result['ok']:
            raise RuntimeError(result['error'])
        data = result['data']
        if 'quality' in data:
            data = {'image_id': data['image_id'], 'exposure_state': data['quality']['exposure_state'],
                    'metric_policy': data['quality']['metric_policy'], 'global': data['quality']['global']}
        observations.append({'action': {'tool': tool, 'arguments': arguments}, 'observation': data})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"mode": "manual_tool_calls_no_llm", 'real_detector_used': bool(args.image),
                                 'input': args.image or 'generated_gradient_fixture', 'steps': observations,
                                 'trace': context.events},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print("Quality example saved:", output)


if __name__ == "__main__":
    main()
