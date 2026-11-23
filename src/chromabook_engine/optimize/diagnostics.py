"""Image diagnostics on an analysis-resolution copy (never touches the original)."""
from __future__ import annotations
import cv2
import numpy as np
from ..color import rgb_to_lab, delta_e00_matrix


def _small(rgb, side=512):
    h, w = rgb.shape[:2]
    s = min(1.0, side / max(h, w))
    return cv2.resize(rgb, (max(8, round(w * s)), max(8, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else rgb


def diagnose(rgb: np.ndarray, palette=None) -> dict:
    sm = _small(rgb)
    lab = rgb_to_lab(sm)
    L = lab[:, :, 0]
    gray = cv2.cvtColor(sm, cv2.COLOR_RGB2GRAY)
    chroma = np.hypot(lab[:, :, 1], lab[:, :, 2])
    # noise: robust sigma of the Laplacian residual (Immerkaer-style)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    resid = cv2.filter2D(gray.astype(np.float32), -1, k)
    noise = float(np.median(np.abs(resid)) / 0.6745 * np.sqrt(np.pi / 2) / 6.0)
    sharp = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    edges = cv2.Canny(gray, 55, 150)
    p5, p50, p95 = np.percentile(L, [5, 50, 95])
    d = {"size": [int(rgb.shape[1]), int(rgb.shape[0])], "aspect_h_over_w": round(rgb.shape[0] / rgb.shape[1], 4),
         "L_mean": float(L.mean()), "L_p5": float(p5), "L_p50": float(p50), "L_p95": float(p95),
         "L_range_p5_p95": float(p95 - p5), "chroma_mean": float(chroma.mean()),
         "noise_sigma_est": noise, "sharpness_laplacian_var": sharp,
         "edge_density_pct": float(100.0 * (edges > 0).mean()),
         "white_fraction_pct": float(100.0 * np.mean((sm > 245).all(axis=2)))}
    flags = []
    if d["L_range_p5_p95"] < 45: flags.append("low_contrast")
    if d["L_mean"] < 38: flags.append("dark")
    if d["L_mean"] > 78: flags.append("bright")
    if noise > 4.0: flags.append("noisy")
    if sharp < 60: flags.append("soft_or_blurry")
    if d["chroma_mean"] < 8: flags.append("near_monochrome")
    if d["white_fraction_pct"] > 40: flags.append("large_white_areas")
    if palette is not None:
        px = lab.reshape(-1, 3)[:: max(1, lab.shape[0] * lab.shape[1] // 3000)]
        de = delta_e00_matrix(px, palette.lab).min(axis=1)
        d["palette_gap_mean_dE00"] = float(de.mean()); d["palette_gap_p95_dE00"] = float(np.percentile(de, 95))
        if d["palette_gap_p95_dE00"] > 25: flags.append("palette_cannot_reach_some_colors")
    d["flags"] = flags
    return d
