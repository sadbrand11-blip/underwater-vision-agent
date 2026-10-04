import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from data import SODDDataset


class SODDAdapterTests(unittest.TestCase):
    def test_author_class_mapping_and_yolo_box_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "validation" / "images"
            labels = root / "validation" / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            image = np.full((100, 200, 3), 80, dtype=np.uint8)
            cv2.imwrite(str(images / "frame_1_original.jpg"), image)
            (labels / "frame_1_original.txt").write_text("5 0.5 0.5 0.4 0.2\n", encoding="utf-8")
            dataset = SODDDataset(root, "val")
            _, target, _ = dataset[0]
            self.assertEqual(dataset.classes[int(target["labels"][0])], "pipe")
            self.assertEqual(target["boxes"].tolist(), [[60.0, 40.0, 140.0, 60.0]])


if __name__ == "__main__":
    unittest.main()
