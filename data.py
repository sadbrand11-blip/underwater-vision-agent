"""Dataset adapters. Exposure variants are always grouped by source image."""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


PCB_CLASSES = ["background", "open", "short", "mousebite", "spur", "copper", "pin-hole"]
NEU_CLASSES = ["background", "crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]
SODD_CLASSES = ["background", "propeller", "pipe_type2", "red_fin", "net", "qr_codes", "pipe"]
_NEU_NAMES = {"crazing": 1, "inclusion": 2, "patches": 3, "pitted_surface": 4,
              "pitted surface": 4, "rolled-in_scale": 5, "rolled-in scale": 5,
              "rolled_in_scale": 5, "scratches": 6}


def read_rgb(path: str | Path) -> np.ndarray:
    raw = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if raw is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)


def save_rgb(path: str | Path, rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


class DeepPCBDataset(Dataset):
    """DeepPCB's official train/test pairs, with an internal grouped validation split.

    Each defective ``_test`` image and its normal ``_temp`` image share one pair
    ID. A pair never crosses train, validation, or test boundaries.
    """

    classes = PCB_CLASSES

    def __init__(self, root: str | Path, split: str, max_pairs: int | None = None, augment: bool = False):
        self.root = Path(root)
        self.augment = augment
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        list_name = "test.txt" if split == "test" else "trainval.txt"
        lines = (self.root / list_name).read_text(encoding="utf-8").splitlines()
        pairs = []
        for line in lines:
            image_rel, annotation_rel = line.split()
            image_base = self.root / image_rel
            test_image = image_base.with_name(image_base.stem + "_test.jpg")
            template = image_base.with_name(image_base.stem + "_temp.jpg")
            annotation = self.root / annotation_rel
            if not (test_image.exists() and template.exists() and annotation.exists()):
                raise FileNotFoundError(f"Incomplete DeepPCB pair: {line}")
            pairs.append((test_image, template, annotation))
        if split != "test":
            rng = random.Random(2026)
            rng.shuffle(pairs)
            boundary = int(0.8 * len(pairs))
            pairs = pairs[:boundary] if split == "train" else pairs[boundary:]
        if max_pairs is not None:
            pairs = pairs[:max_pairs]
        self.items = []
        for test_image, template, annotation in pairs:
            self.items.append((test_image, annotation, True))
            self.items.append((template, annotation, False))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        image_path, annotation_path, defective = self.items[index]
        image = read_rgb(image_path)
        boxes = []
        labels = []
        if defective:
            for line in annotation_path.read_text(encoding="utf-8").splitlines():
                x0, y0, x1, y1, label = map(int, line.split())
                if x1 > x0 and y1 > y0:
                    boxes.append([x0, y0, x1, y1])
                    labels.append(label)
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(labels, dtype=np.int64)
        height, width = image.shape[:2]
        if self.augment and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            if len(boxes):
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        if self.augment and random.random() < 0.5:
            image = np.ascontiguousarray(image[::-1])
            if len(boxes):
                boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(index),
            "area": torch.as_tensor((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), dtype=torch.float32),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }
        return tensor, target, str(image_path)


def collate(batch):
    return tuple(zip(*batch))


class NEUDataset(Dataset):
    """NEU-DET continuous-tone steel images with source-disjoint splits.

    The 30 images in Validation_Images are reserved for the test set. Remaining
    images are deterministically split 80/20 within each class. Synthetic
    exposure is applied only after a source image has been assigned a split.
    """

    classes = NEU_CLASSES

    def __init__(self, root: str | Path, split: str, max_images: int | None = None,
                 augment: bool = False, exposure: str = "none", exposure_augment: bool = False):
        self.root = Path(root)
        self.augment = augment
        self.exposure = exposure
        self.exposure_augment = exposure_augment
        if split not in {"train", "val", "test"}:
            raise ValueError(split)
        if exposure not in {"none", "under", "over", "mixed"}:
            raise ValueError(exposure)
        if split == "test":
            image_dir, annotation_dir = self.root / "Validation_Images", self.root / "Validation_Annotations"
            images = sorted(image_dir.glob("*.jpg"))
        else:
            image_dir, annotation_dir = self.root / "IMAGES", self.root / "ANNOTATIONS"
            images = []
            for category in ["crazing", "inclusion", "patches", "pitted", "rolled-in", "scratches"]:
                group = sorted(image_dir.glob(f"{category}_*.jpg"))
                random.Random(2026).shuffle(group)
                boundary = int(0.8 * len(group))
                images.extend(group[:boundary] if split == "train" else group[boundary:])
        if max_images is not None:
            # Take a balanced prefix from each category, then preserve a stable order.
            buckets: dict[str, list[Path]] = {}
            for image in images:
                buckets.setdefault(image.stem.rsplit("_", 1)[0], []).append(image)
            selected = []
            while len(selected) < max_images and any(buckets.values()):
                for bucket in buckets.values():
                    if bucket and len(selected) < max_images:
                        selected.append(bucket.pop(0))
            images = selected
        self.items = [(image, annotation_dir / f"{image.stem}.xml") for image in images]
        for image, annotation in self.items:
            if not annotation.exists():
                raise FileNotFoundError(annotation)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        image_path, annotation_path = self.items[index]
        image = read_rgb(image_path)
        height, width = image.shape[:2]
        root = ET.parse(annotation_path).getroot()
        boxes, labels = [], []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip().lower().replace(" ", "_")
            name = name.replace("pitted_surface", "pitted_surface")
            if name not in _NEU_NAMES:
                raise ValueError(f"Unknown NEU class {name} in {annotation_path}")
            box = obj.find("bndbox")
            x0, y0 = float(box.findtext("xmin")), float(box.findtext("ymin"))
            x1, y1 = float(box.findtext("xmax")), float(box.findtext("ymax"))
            x0, x1 = sorted((max(0, min(width, x0)), max(0, min(width, x1))))
            y0, y1 = sorted((max(0, min(height, y0)), max(0, min(height, y1))))
            if x1 > x0 and y1 > y0:
                boxes.append([x0, y0, x1, y1])
                labels.append(_NEU_NAMES[name])
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(labels, dtype=np.int64)
        if self.augment and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        if self.augment and random.random() < 0.5:
            image = np.ascontiguousarray(image[::-1])
            boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
        if self.exposure != "none":
            image = simulate_exposure(image, self.exposure)
        elif self.exposure_augment and random.random() < 0.55:
            gain = random.uniform(0.22, 0.85) if random.random() < 0.75 else random.uniform(1.1, 2.0)
            offset = 0.0 if gain < 1 else random.uniform(0.02, 0.12)
            image = np.uint8(np.clip(np.round((image.astype(np.float32) / 255 * gain + offset) * 255), 0, 255))
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32),
                  "labels": torch.as_tensor(labels, dtype=torch.int64),
                  "image_id": torch.tensor(index),
                  "area": torch.as_tensor((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), dtype=torch.float32),
                  "iscrowd": torch.zeros(len(boxes), dtype=torch.int64)}
        return tensor, target, str(image_path)


def simulate_exposure(image: np.ndarray, mode: str) -> np.ndarray:
    """Deterministic JPEG-domain stress test, not a physical camera simulation."""
    x = image.astype(np.float32) / 255.0
    if mode == "under":
        x = x * 0.28
    elif mode == "over":
        x = np.minimum(1.0, x * 4.0 + 0.45)
    elif mode == "mixed":
        mask = np.linspace(0.2, 5.0, image.shape[1], dtype=np.float32)[None, :, None]
        offset = np.linspace(0.0, 0.4, image.shape[1], dtype=np.float32)[None, :, None]
        x = np.minimum(1.0, x * mask + offset)
    elif mode != "none":
        raise ValueError(mode)
    return np.uint8(np.clip(np.round(x * 255), 0, 255))


class SODDDataset(Dataset):
    """Original SODD underwater frames with YOLO box labels.

    The supplied directory may be a range-downloaded subset. Augmented copies
    from the source archive are excluded so derived frames are not counted as
    independent observations. The official split is retained, with the caveat
    that nearby video frames can still appear across splits.
    """

    classes = SODD_CLASSES

    def __init__(self, root: str | Path, split: str, max_images: int | None = None,
                 augment: bool = False, exposure: str = "none", exposure_augment: bool = False):
        self.root = Path(root)
        self.augment = augment
        self.exposure = exposure
        self.exposure_augment = exposure_augment
        if split not in {"train", "val", "test"}:
            raise ValueError(split)
        if exposure not in {"none", "under", "over", "mixed"}:
            raise ValueError(exposure)
        folder = {"train": "train", "val": "validation", "test": "test"}[split]
        directory = self.root / folder
        images = sorted((directory / "images").glob("*_original.jpg"))
        if max_images is not None:
            images = images[:max_images]
        if not images:
            raise FileNotFoundError(f"No original SODD images found in {directory / 'images'}")
        self.items = []
        for image in images:
            label = directory / "labels" / f"{image.stem}.txt"
            if not label.exists():
                raise FileNotFoundError(label)
            self.items.append((image, label))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        image_path, label_path = self.items[index]
        image = read_rgb(image_path)
        height, width = image.shape[:2]
        boxes, labels = [], []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Malformed YOLO label in {label_path}: {line}")
            category = int(fields[0])
            if category not in range(6):
                raise ValueError(f"Unknown SODD category {category} in {label_path}")
            cx, cy, bw, bh = map(float, fields[1:])
            x0, x1 = sorted((np.clip((cx - bw / 2) * width, 0, width),
                             np.clip((cx + bw / 2) * width, 0, width)))
            y0, y1 = sorted((np.clip((cy - bh / 2) * height, 0, height),
                             np.clip((cy + bh / 2) * height, 0, height)))
            if x1 > x0 and y1 > y0:
                boxes.append([x0, y0, x1, y1])
                labels.append(category + 1)
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(labels, dtype=np.int64)
        if self.augment and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        if self.exposure != "none":
            image = simulate_exposure(image, self.exposure)
        elif self.exposure_augment and random.random() < 0.55:
            gain = random.uniform(0.3, 0.9) if random.random() < 0.8 else random.uniform(1.1, 1.7)
            image = np.uint8(np.clip(np.round(image.astype(np.float32) * gain), 0, 255))
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32),
                  "labels": torch.as_tensor(labels, dtype=torch.int64),
                  "image_id": torch.tensor(index),
                  "area": torch.as_tensor((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), dtype=torch.float32),
                  "iscrowd": torch.zeros(len(boxes), dtype=torch.int64)}
        return tensor, target, str(image_path)
