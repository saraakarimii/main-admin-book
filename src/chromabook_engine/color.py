"""Colour-space helpers. Delta E 2000 comes from scikit-image (reference-grade)."""
from __future__ import annotations
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000


def rgb_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    return rgb2lab(rgb_u8.astype(np.float32) / 255.0).astype(np.float32)


def delta_e00_matrix(pixel_labs: np.ndarray, palette_labs: np.ndarray) -> np.ndarray:
    """(N,3) x (K,3) -> (N,K) CIEDE2000."""
    return deltaE_ciede2000(pixel_labs[:, None, :], palette_labs[None, :, :]).astype(np.float32)


def delta_e76(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1)
