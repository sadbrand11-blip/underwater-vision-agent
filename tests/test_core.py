import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import OpticalAgent
from data import simulate_exposure
from metrics import iou, summarize
from quality import assess


class FakeDetector:
    threshold = 0.5
    calibration = None

    def predict(self, rgb, threshold=None):
        return [{"box": [20, 20, 40, 40], "score": 0.9,
                 "class_id": 1, "class_name": "defect"}]


class BrightnessSensitiveDetector(FakeDetector):
    def predict(self, rgb, threshold=None):
        score = 0.9 if float(rgb.mean()) < 60 else 0.3
        return [{"box": [20, 20, 40, 40], "score": score,
                 "class_id": 1, "class_name": "defect"}]


class CoreTests(unittest.TestCase):
    def test_blank_saturated_frame_is_quality_failure(self):
        image = np.full((64, 64, 3), 255, dtype=np.uint8)
        quality, _ = assess(image)
        self.assertEqual(quality["exposure_state"], "overexposed")
        self.assertFalse(quality["quality_pass"])
        report, _ = OpticalAgent(FakeDetector()).inspect([image])
        self.assertEqual(report["status"], "quality_failure")
        self.assertEqual(report["target_conclusion"], "unreliable")

    def test_two_tone_input_does_not_claim_exposure(self):
        image = np.full((64, 64, 3), 255, dtype=np.uint8)
        image[:, :32] = 0
        quality, _ = assess(image)
        self.assertEqual(quality["exposure_state"], "preprocessed_binary")
        self.assertIsNone(quality["quality_pass"])
        self.assertEqual(quality["lost_tile_fraction"], 0)

    def test_synthetic_exposure_does_not_modify_source(self):
        image = np.full((16, 16, 3), 128, dtype=np.uint8)
        dark = simulate_exposure(image, "under")
        self.assertTrue(np.all(image == 128))
        self.assertLess(int(dark.mean()), 128)

    def test_median_catches_dark_frame_without_clipped_pixels(self):
        image = np.full((64, 64, 3), 36, dtype=np.uint8)
        quality, heatmap = assess(image)
        self.assertEqual(quality["exposure_state"], "underexposed")
        self.assertTrue(quality["quality_pass"])
        self.assertGreater(int(heatmap[:, :, 2].mean()), 0)

    def test_abstention_is_not_scored_as_correct_detection(self):
        truth = [[{"box": [0, 0, 10, 10], "class_id": 1}]]
        report = summarize([[]], truth, 2)
        self.assertEqual(report["recall"], 0)
        self.assertEqual(report["fn"], 1)

    def test_correction_candidate_rejected_when_detection_degrades(self):
        image = np.full((64, 64, 3), 35, dtype=np.uint8)
        report, _ = OpticalAgent(BrightnessSensitiveDetector()).inspect([image])
        self.assertEqual(report["selected_image"], "original")
        self.assertFalse(report["correction"]["accepted"])
        self.assertEqual(report["status"], "unreliable")

    def test_multi_frame_consistent_evidence(self):
        image = np.full((64, 64, 3), 130, dtype=np.uint8)
        image[20:40, 20:40] = 90
        report, _ = OpticalAgent(FakeDetector()).inspect([image, image.copy()])
        self.assertEqual(report["multi_frame"]["consistent_frames"], 2)
        self.assertEqual(report["status"], "reliable")


if __name__ == "__main__":
    unittest.main()
