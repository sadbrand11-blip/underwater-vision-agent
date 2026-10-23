"""Constrained ordinary-photo input; legacy data readers remain unchanged."""
from __future__ import annotations

from io import BytesIO
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 16_000_000
PREPROCESSING_VERSION = 'jpeg_png_8bit_exif_rgb_v1'


def read_image_bytes(payload: bytes) -> np.ndarray:
    """Decode an 8-bit JPEG/PNG, orient with EXIF, and return owned RGB pixels.

    Alpha PNGs are composited onto white explicitly. Palette PNGs are supported;
    1-bit, 16-bit, float, animated and non-JPEG/PNG inputs are rejected.
    """
    if not isinstance(payload, bytes) or not payload:
        raise ValueError('图片内容为空或格式无效')
    if len(payload) > MAX_FILE_BYTES:
        raise ValueError('图片超过32 MiB限制')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as image:
                if image.format not in {'JPEG', 'PNG'}:
                    raise ValueError('仅支持8位 JPEG、PNG 图片')
                if image.width <= 0 or image.height <= 0 or image.width * image.height > MAX_PIXELS:
                    raise ValueError('图片超过1600万像素限制')
                if getattr(image, 'n_frames', 1) != 1:
                    raise ValueError('不支持动画图片，请提供单张8位 JPEG、PNG')
                # Pillow may represent 16-bit grayscale PNG as mode I.
                if image.mode not in {'L', 'LA', 'P', 'RGB', 'RGBA', 'CMYK'}:
                    raise ValueError('仅支持8位图片，不支持二值、16位或浮点图像')
                if image.format == 'PNG' and len(payload) > 24 and payload[12:16] == b'IHDR' and payload[24] != 8:
                    raise ValueError('PNG必须为8位图像')
                image.load()
                oriented = ImageOps.exif_transpose(image)
                if 'A' in oriented.getbands() or (oriented.mode == 'P' and 'transparency' in oriented.info):
                    rgba = oriented.convert('RGBA')
                    canvas = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
                    oriented = Image.alpha_composite(canvas, rgba)
                return np.array(oriented.convert('RGB'), dtype=np.uint8, copy=True)
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError('图片无法解码或尺寸不安全，请提供有效的8位 JPEG、PNG') from exc
