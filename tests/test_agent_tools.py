import unittest

import numpy as np

from agent import OpticalAgent
from optical_agent.tools import ToolContext, execute_tool
from quality import UNDERWATER_QUALITY_CONFIG, assess


class FakeDetector:
    dataset = 'sodd'
    threshold = 0.33

    def __init__(self):
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        return [{'class_id': 5, 'class_name': 'qr_codes', 'box': [10, 10, 30, 30],
                 'score': 0.9, 'estimated_box_precision': 0.8}]


class ToolsTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((64, 64, 3), 100, np.uint8)
        self.detector = FakeDetector()
        self.context = ToolContext(self.detector, self.image)

    def test_quality_matches_underwater_baseline_and_does_not_detect(self):
        result = execute_tool(self.context, 'assess_image_quality', {'image_id': 'original'})
        quality, _ = assess(self.image, UNDERWATER_QUALITY_CONFIG)
        self.assertEqual(result['data']['quality'], quality)
        self.assertEqual(self.detector.calls, 0)

    def test_full_tool_matches_baseline_except_timing(self):
        baseline, _ = OpticalAgent(FakeDetector()).inspect([self.image])
        report = execute_tool(self.context, 'analyze_image_full_pipeline', {'image_id': 'original'})['data']['report']
        baseline.pop('latency_ms')
        report.pop('latency_ms')
        self.assertEqual(report, baseline)

    def test_cache_and_one_correction(self):
        execute_tool(self.context, 'detect_objects', {'image_id': 'original'})
        execute_tool(self.context, 'detect_objects', {'image_id': 'original'})
        self.assertEqual(self.detector.calls, 1)
        execute_tool(self.context, 'assess_image_quality', {'image_id': 'original'})
        execute_tool(self.context, 'correct_image_exposure', {'image_id': 'original'})
        corrected = self.context.images['corrected'].copy()
        execute_tool(self.context, 'correct_image_exposure', {'image_id': 'original'})
        np.testing.assert_array_equal(self.context.images['corrected'], corrected)
        np.testing.assert_array_equal(self.context.images['original'], self.image)
        self.assertFalse(execute_tool(self.context, 'correct_image_exposure', {'image_id': 'corrected'})['ok'])

    def test_validation_and_missing_dependencies(self):
        for tool, args in [('fake', {}), ('detect_objects', {}), ('detect_objects', {'image_id': 3}),
                           ('detect_objects', {'image_id': 'other'}), ('detect_objects', {'image_id': 'corrected'}),
                           ('detect_objects', {'image_id': 'original', 'threshold': 0})]:
            self.assertFalse(execute_tool(self.context, tool, args)['ok'])
        self.assertFalse(execute_tool(self.context, 'compare_detection_evidence',
            {'original_image_id': 'original', 'corrected_image_id': 'corrected'})['ok'])

    def test_quality_failures_remain_failures(self):
        for value in (0, 255):
            ctx = ToolContext(FakeDetector(), np.full((64, 64, 3), value, np.uint8))
            result = execute_tool(ctx, 'analyze_image_full_pipeline', {'image_id': 'original'})
            self.assertEqual(result['data']['report']['status'], 'quality_failure')
