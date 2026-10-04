"""Exposure and visibility measurements for the offline inspection agent.

The thresholds below are engineering defaults, not calibrated probabilities.
All measurements are kept separately so a later domain calibration can replace
the defaults without changing the output contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ExposureProfile:
    name: str = 'legacy_v030'
    target_median: float = 0.45
    gamma_min: float = 0.65
    gamma_max: float = 1.45
    clahe_blend: float = 0.25

    def __post_init__(self):
        if not (0.05 < self.target_median < 0.95 and 0 < self.gamma_min <= self.gamma_max <= 2
                and 0 <= self.clahe_blend <= 0.25):
            raise ValueError('Exposure parameters exceed bounded correction policy')


@dataclass(frozen=True)
class QualityConfig:
    grid: int = 8
    dark_level: int = 12
    bright_level: int = 250
    binary_fraction: float = 0.97
    dark_fraction_warn: float = 0.30
    bright_fraction_warn: float = 0.20
    dark_median_warn: float = 0.18
    bright_median_warn: float = 0.88
    exposure_tile_share_warn: float = 0.10
    lost_tile_fraction: float = 0.92
    max_lost_tiles_fraction: float = 0.12
    max_global_dark_fraction: float = 1.0
    max_global_bright_fraction: float = 1.0
    calibration_label: str = "uncalibrated_heuristic_v1"


UNDERWATER_QUALITY_CONFIG = QualityConfig(
    max_lost_tiles_fraction=0.50,
    max_global_dark_fraction=0.65,
    max_global_bright_fraction=0.75,
    calibration_label="underwater_background_tolerant_heuristic_pilot_v1",
)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _tile_metrics(tile: np.ndarray, cfg: QualityConfig) -> dict:
    values = tile.astype(np.float32) / 255.0
    sx = cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(values, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(sx * sx + sy * sy)
    blurred = cv2.GaussianBlur(values, (3, 3), 0)
    flat = gradient < np.quantile(gradient, 0.40)
    residual = (values - blurred)[flat]
    noise = float(1.4826 * np.median(np.abs(residual - np.median(residual)))) if residual.size else 0.0
    return {
        "dark_fraction": float(np.mean(tile <= cfg.dark_level)),
        "bright_fraction": float(np.mean(tile >= cfg.bright_level)),
        "median": float(np.median(values)),
        "contrast": float(np.std(values)),
        "edge_strength": float(np.mean(gradient)),
        "sharpness": float(cv2.Laplacian(values, cv2.CV_32F).var()),
        "noise_estimate": noise,
    }


def assess(rgb: np.ndarray, cfg: QualityConfig | None = None) -> tuple[dict, np.ndarray]:
    """Return a JSON-ready report and an RGB problem heatmap.

    Red means likely information loss at the bright end; blue means likely
    information loss in darkness. This map visualizes raw-image evidence and
    does not pretend to localize a defect.
    """
    cfg = cfg or QualityConfig()
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an RGB uint8 image")
    gray = _luminance(rgb)
    height, width = gray.shape
    global_metrics = _tile_metrics(gray, cfg)
    binary_fraction = float(np.mean((gray <= 10) | (gray >= 245)))
    # A blank saturated frame is information loss, not a two-tone PCB input.
    is_binary = (binary_fraction >= cfg.binary_fraction
                 and float(np.mean(gray <= 10)) > 0.05
                 and float(np.mean(gray >= 245)) > 0.05)
    heat = np.zeros((height, width, 3), dtype=np.uint8)
    tiles = []
    lost_count = 0
    dark_count = 0
    bright_count = 0
    for row in range(cfg.grid):
        y0, y1 = height * row // cfg.grid, height * (row + 1) // cfg.grid
        for col in range(cfg.grid):
            x0, x1 = width * col // cfg.grid, width * (col + 1) // cfg.grid
            if x1 <= x0 or y1 <= y0:
                continue
            values = _tile_metrics(gray[y0:y1, x0:x1], cfg)
            dark = (values["dark_fraction"] > cfg.dark_fraction_warn
                    or values["median"] < cfg.dark_median_warn)
            bright = (values["bright_fraction"] > cfg.bright_fraction_warn
                      or values["median"] > cfg.bright_median_warn)
            lost = (not is_binary and
                    max(values["dark_fraction"], values["bright_fraction"]) >= cfg.lost_tile_fraction)
            dark_count += int(dark)
            bright_count += int(bright)
            lost_count += int(lost)
            if not is_binary:
                bright_visual = max(values["bright_fraction"],
                                    max(0.0, (values["median"] - cfg.bright_median_warn) /
                                        max(1.0 - cfg.bright_median_warn, 1e-6)))
                dark_visual = max(values["dark_fraction"],
                                  max(0.0, (cfg.dark_median_warn - values["median"]) /
                                      max(cfg.dark_median_warn, 1e-6)))
                heat[y0:y1, x0:x1, 0] = min(255, round(255 * bright_visual))
                heat[y0:y1, x0:x1, 2] = min(255, round(255 * dark_visual))
            tiles.append({"row": row, "col": col, "box": [x0, y0, x1, y1], "information_loss": lost, **values})
    tile_count = max(len(tiles), 1)
    if is_binary:
        state = "preprocessed_binary"
        passed = None
        reason = "输入接近二值图，无法从像素分布判断原始曝光"
    else:
        has_dark = dark_count > tile_count * cfg.exposure_tile_share_warn
        has_bright = bright_count > tile_count * cfg.exposure_tile_share_warn
        extreme_mixed = (sum(t["median"] < 0.15 for t in tiles) >= 3
                         and sum(t["median"] > 0.95 for t in tiles) >= 3)
        state = "mixed" if (has_dark and has_bright) or extreme_mixed else "underexposed" if has_dark else "overexposed" if has_bright else "normal"
        passed = (lost_count / tile_count <= cfg.max_lost_tiles_fraction
                  and global_metrics["dark_fraction"] <= cfg.max_global_dark_fraction
                  and global_metrics["bright_fraction"] <= cfg.max_global_bright_fraction)
        reason = "整图仍有大面积不可辨像素" if not passed else "图像质量通过启发式检查"
    report = {
        "exposure_state": state,
        "quality_pass": passed,
        "reason": reason,
        "metric_policy": cfg.calibration_label,
        "binary_fraction": binary_fraction,
        "lost_tile_fraction": lost_count / tile_count,
        "global": global_metrics,
        "tiles": tiles,
    }
    return report, heat


def correct_exposure(rgb: np.ndarray, before: dict, profile: ExposureProfile | None = None,
                     method: str = 'local_bounded') -> tuple[np.ndarray, dict]:
    """One bounded, non-generative correction pass; never invents details."""
    profile = profile or ExposureProfile()
    if method not in {'gamma_only', 'local_bounded'}:
        raise ValueError('Unknown bounded correction method')
    if before.get('quality_pass') is False:
        return rgb.copy(), {'applied': False, 'method': 'identity', 'profile': profile.name,
                            'reason': '原图信息丢失，停止恢复；增强不能成为可靠证据'}
    if before["exposure_state"] in {"normal", "preprocessed_binary"}:
        return rgb.copy(), {"applied": False, "method": "identity", "reason": "无需校正或输入不适合曝光估计"}
    gray = _luminance(rgb).astype(np.float32) / 255.0
    median = float(np.median(gray))
    # Brightening needs gamma < 1; darkening needs gamma > 1.
    gamma = float(np.clip(np.log(profile.target_median) / np.log(np.clip(median, 0.05, 0.95)), profile.gamma_min, profile.gamma_max))
    lut = np.clip(np.round((np.arange(256) / 255.0) ** gamma * 255), 0, 255).astype(np.uint8)
    mapped = cv2.LUT(rgb, lut)
    if method == 'gamma_only' or profile.clahe_blend == 0:
        return mapped, {'applied': True, 'method': 'bounded_gamma', 'gamma': gamma,
                        'profile': profile.name, 'target_median': profile.target_median,
                        'clahe_blend': 0.0, 'denoise_applied': False}
    lab = cv2.cvtColor(mapped, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    local = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(l)
    # Limited blending keeps the correction from emphasizing every faint edge.
    l = cv2.addWeighted(l, 1-profile.clahe_blend, local, profile.clahe_blend, 0)
    corrected = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    denoise = (before["exposure_state"] in {"underexposed", "mixed"}
               and before["global"]["noise_estimate"] >= 0.01)
    if denoise:
        corrected = cv2.fastNlMeansDenoisingColored(corrected, None, 2, 2, 5, 15)
    return corrected, {
        "applied": True,
        "method": "bounded_gamma_clahe_denoise" if denoise else "bounded_gamma_clahe",
        "denoise_applied": denoise,
        "gamma": gamma,
        "profile": profile.name,
        "target_median": profile.target_median,
        "clahe_blend": profile.clahe_blend,
        "irrecoverable_original_pixels": {
            "dark_fraction": before["global"]["dark_fraction"],
            "bright_fraction": before["global"]["bright_fraction"],
        },
    }
