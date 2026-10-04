import io
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as webapp
from agent import OpticalAgent


class FakeDetector:
    threshold = 0.5
    calibration = None

    def predict(self, rgb, threshold=None):
        return [{"box": [12, 12, 30, 30], "class_id": 1,
                 "class_name": "sample", "score": 0.9}]


class WebSmokeTests(unittest.TestCase):
    def test_upload_renders_images_and_result(self):
        webapp._agent = OpticalAgent(FakeDetector())
        image = np.full((48, 48, 3), 130, dtype=np.uint8)
        image[12:30, 12:30] = 90
        success, buffer = cv2.imencode(".png", image)
        self.assertTrue(success)
        client = webapp.app.test_client()
        response = client.post("/", data={"images": (io.BytesIO(buffer.tobytes()), "sample.png")},
                               content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"data:image/jpeg;base64,", response.data)
        self.assertIn(b"reliable", response.data)


if __name__ == "__main__":
    unittest.main()
