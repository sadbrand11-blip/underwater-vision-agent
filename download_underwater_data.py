"""Download a small, reproducible subset of official underwater datasets.

Only selected ZIP members are fetched through HTTP range requests. Data stays
outside the Git repository; the default destination is on drive D on Windows.
"""

from __future__ import annotations

import argparse
import binascii
import json
import struct
import time
import zlib
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import requests


SODD_URL = "https://zenodo.org/api/records/10230328/files/SODD.zip/content"
MOUD_URL = "https://zenodo.org/api/records/15836103/files/Scene_1.zip/content"


def get_range(url: str, start: int, end: int) -> bytes:
    for attempt in range(7):
        try:
            response = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                pause = min(90, max(5, float(retry_after))) if retry_after else min(90, 5 * 2 ** attempt)
                time.sleep(pause)
                continue
            response.raise_for_status()
            if response.status_code != 206 or len(response.content) != end - start + 1:
                raise IOError(f"Server did not return requested byte range: {response.status_code}")
            return response.content
        except (requests.RequestException, IOError):
            if attempt == 6:
                raise
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError("Official dataset server is still rate-limiting requests; retry later")


class RemoteReader:
    """Minimal seekable file needed to load a remote ZIP central directory."""

    def __init__(self, url: str):
        self.url = url
        for attempt in range(5):
            response = requests.head(url, timeout=30)
            if response.status_code != 429:
                break
            time.sleep(min(60, 5 * 2 ** attempt))
        response.raise_for_status()
        self.size = int(response.headers["Content-Length"])
        self.position = 0

    def tell(self) -> int:
        return self.position

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.position
        start = self.position
        end = min(self.size - 1, start + size - 1)
        if end < start:
            return b""
        data = get_range(self.url, start, end)
        self.position += len(data)
        return data


def spaced(items: list[str], count: int) -> list[str]:
    items = sorted(items)
    if len(items) <= count:
        return items
    if count == 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


def select_sodd(names: set[str], train: int, val: int, test: int) -> list[str]:
    selected = []
    for split, count in (("train", train), ("validation", val), ("test", test)):
        images = [name for name in names if name.startswith(f"SODD/data/{split}/images/")
                  and name.endswith("_original.jpg")]
        for image in spaced(images, count):
            label = image.replace("/images/", "/labels/").removesuffix(".jpg") + ".txt"
            if label not in names:
                raise FileNotFoundError(f"Annotation missing: {label}")
            selected.extend((image, label))
    return selected


def select_moud(names: set[str], per_area: int) -> list[str]:
    selected = []
    for light in ("Low light", "Mid light", "High light"):
        for area in range(1, 5):
            prefix = f"Scene_1/Images/{light}/{light} area_{area}/"
            images = [name for name in names if name.startswith(prefix) and name.endswith(".jpg")]
            if light == "High light" and area in (1, 4):
                annotated = [name for name in images if
                             f"Scene_1/Annotation/labels_{area}/{Path(name).stem}.json" in names]
                images = annotated or images
            for image in spaced(images, per_area):
                selected.append(image)
                annotation = f"Scene_1/Annotation/labels_{area}/{Path(image).stem}.json"
                if annotation in names:
                    selected.append(annotation)
    return selected


def member_bytes(url: str, info: zipfile.ZipInfo, next_offset: int) -> bytes:
    region = get_range(url, info.header_offset, next_offset - 1)
    if len(region) < 30:
        raise IOError(f"Truncated ZIP member: {info.filename}")
    header = struct.unpack("<IHHHHHIIIHH", region[:30])
    if header[0] != 0x04034B50:
        raise IOError(f"Invalid ZIP member header: {info.filename}")
    filename_length, extra_length = header[-2:]
    start = 30 + filename_length + extra_length
    compressed = region[start:start + info.compress_size]
    if len(compressed) != info.compress_size:
        raise IOError(f"Truncated compressed data: {info.filename}")
    if info.compress_type == zipfile.ZIP_STORED:
        raw = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        raw = zlib.decompress(compressed, -15)
    else:
        raise ValueError(f"Unsupported ZIP compression {info.compress_type}")
    if len(raw) != info.file_size or binascii.crc32(raw) != info.CRC:
        raise IOError(f"ZIP integrity check failed: {info.filename}")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("sodd", "moud"))
    parser.add_argument("--root", type=Path, default=Path(r"D:\CodexData\optical_agent"))
    parser.add_argument("--train", type=int, default=160, help="SODD original train images")
    parser.add_argument("--val", type=int, default=40, help="SODD original validation images")
    parser.add_argument("--test", type=int, default=40, help="SODD original test images")
    parser.add_argument("--per-area", type=int, default=2, help="MOUD images per lighting level and area")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if __import__("os").name == "nt" and args.root.drive.upper() != "D:":
        raise ValueError("Dataset downloads must be stored on drive D")
    if min(args.train, args.val, args.test, args.per_area, args.workers) < 1:
        raise ValueError("All counts must be positive")
    url = SODD_URL if args.dataset == "sodd" else MOUD_URL
    root = args.root / args.dataset
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RemoteReader(url)) as archive:
        members = archive.infolist()
        by_name = {member.filename: member for member in members}
        names = set(by_name)
        selected = (select_sodd(names, args.train, args.val, args.test) if args.dataset == "sodd"
                    else select_moud(names, args.per_area))
        offsets = sorted(member.header_offset for member in members)
        next_offsets = {offset: offsets[i + 1] if i + 1 < len(offsets) else archive.start_dir
                        for i, offset in enumerate(offsets)}

    def obtain(name: str) -> dict:
        member = by_name[name]
        output = root.joinpath(*Path(name).parts)
        if not output.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Unsafe archive path: {name}")
        if (output.exists() and output.stat().st_size == member.file_size
                and binascii.crc32(output.read_bytes()) == member.CRC):
            return {"path": name, "bytes": member.file_size, "already_present": True}
        raw = member_bytes(url, member, next_offsets[member.header_offset])
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".part")
        temporary.write_bytes(raw)
        temporary.replace(output)
        return {"path": name, "bytes": len(raw), "already_present": False}

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(obtain, name): name for name in selected}
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 20 == 0 or len(results) == len(selected):
                print(f"{args.dataset}: {len(results)}/{len(selected)} files", flush=True)
    manifest = {"dataset": args.dataset, "source": url, "date": date.today().isoformat(),
                "selection": "spaced_original_frames_by_official_split" if args.dataset == "sodd"
                else "spaced_frames_per_area_and_real_illumination_level",
                "note": "MOUD lighting levels are lamp illumination, not camera exposure time"
                if args.dataset == "moud" else "SODD augmented copies excluded",
                "files": sorted(results, key=lambda item: item["path"])}
    (root / "subset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved to {root}; {len(results)} files", flush=True)


if __name__ == "__main__":
    main()
