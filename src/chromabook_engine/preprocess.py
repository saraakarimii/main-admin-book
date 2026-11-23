"""Preprocessing operations (ported from the reference notebook, plus brightness/contrast)."""
from __future__ import annotations
import cv2
import numpy as np
from skimage.color import rgb2lab, lab2rgb


def brightness_contrast(rgb, brightness=0.0, contrast=1.0):
    if brightness == 0 and contrast == 1.0:
        return rgb.copy()
    lab = rgb2lab(rgb.astype(np.float32) / 255.0).astype(np.float32)
    lab[:, :, 0] = np.clip((lab[:, :, 0] - 50.0) * float(contrast) + 50.0 + float(brightness), 0, 100)
    return np.clip(lab2rgb(lab) * 255.0, 0, 255).astype(np.uint8)


def illumination_correction(rgb, strength=0.0, sigma_fraction=0.10):
    if strength <= 0:
        return rgb.copy(), np.zeros(rgb.shape[:2], np.uint8)
    lab = rgb2lab(rgb.astype(np.float32) / 255.0).astype(np.float32)
    L = lab[:, :, 0]
    sigma = max(3.0, min(rgb.shape[:2]) * float(sigma_fraction))
    background = cv2.GaussianBlur(L, (0, 0), sigmaX=sigma, sigmaY=sigma)
    target = float(np.median(background))
    lab[:, :, 0] = np.clip(L - float(strength) * (background - target), 0, 100)
    out = np.clip(lab2rgb(lab) * 255.0, 0, 255).astype(np.uint8)
    shade = background - target
    lo, hi = np.percentile(shade, [2, 98])
    vis = np.clip((shade - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)
    return out, vis


def clahe_luminance(rgb, clip_limit=0.0, tile_grid=8):
    if clip_limit <= 0:
        return rgb.copy()
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def denoise_rgb(rgb, method="bilateral", strength=0.0):
    if method == "none" or strength <= 0:
        return rgb.copy()
    if method == "bilateral":
        d = 5 if strength < 35 else 7
        return cv2.bilateralFilter(rgb, d, float(strength), max(1.0, float(strength) / 8))
    if method == "nlm":
        h = float(strength)
        return cv2.fastNlMeansDenoisingColored(rgb, None, h, h, 7, 21)
    if method == "median":
        return cv2.medianBlur(rgb, 3 if strength < 35 else 5)
    raise ValueError(f"unknown denoise method {method!r}")


def multiscale_sharpen(rgb, amount_fine=0.0, amount_coarse=0.0):
    if amount_fine <= 0 and amount_coarse <= 0:
        return rgb.copy()
    lab = rgb2lab(rgb.astype(np.float32) / 255.0).astype(np.float32)
    L = lab[:, :, 0]
    fine = L - cv2.GaussianBlur(L, (0, 0), 0.8)
    coarse = L - cv2.GaussianBlur(L, (0, 0), 2.0)
    lab[:, :, 0] = np.clip(L + amount_fine * fine + amount_coarse * coarse, 0, 100)
    return np.clip(lab2rgb(lab) * 255.0, 0, 255).astype(np.uint8)


def preprocess(reference_rgb, p) -> dict:
    """Returns every intermediate stage (kept for reports) - the input is never modified."""
    bc = brightness_contrast(reference_rgb, p.brightness, p.contrast)
    illum, shade = illumination_correction(bc, p.illumination_strength, p.illumination_sigma)
    local = clahe_luminance(illum, p.clahe_clip, p.clahe_tiles)
    den = denoise_rgb(local, p.denoise_method, p.denoise_strength)
    sharp = multiscale_sharpen(den, p.sharpen_fine, p.sharpen_coarse)
    return {"reference": reference_rgb, "brightness_contrast": bc, "illumination_corrected": illum,
            "clahe": local, "denoised": den, "processed": sharp, "shade_map": shade}
