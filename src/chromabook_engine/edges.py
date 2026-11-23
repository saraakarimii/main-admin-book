"""Edge extraction: Scharr gradient + Canny (auto/manual) + closing -> boundary-strength map."""
from __future__ import annotations
import cv2
import numpy as np


def extract_edges(rgb, e):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    grad = np.clip(mag / max(1e-6, float(np.percentile(mag, 99.5))), 0, 1)
    if e.auto_canny:
        med = float(np.median(gray))
        lo = int(max(0, (1.0 - 0.33) * med))
        hi = int(min(255, max(lo + 1, (1.0 + 0.33) * med)))
    else:
        lo, hi = int(e.canny_low), int(e.canny_high)
        if hi <= lo:
            hi = min(255, lo + 1)
    canny = cv2.Canny(gray, lo, hi, L2gradient=True)
    if e.close_iter > 0:
        canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8),
                                 iterations=int(e.close_iter))
    combined = np.maximum(grad * float(e.scharr_weight), canny.astype(np.float32) / 255.0)
    combined = cv2.GaussianBlur(combined, (0, 0), 0.65)
    combined = np.clip(combined * float(e.sensitivity), 0, 1).astype(np.float32)
    return combined, canny, grad, (lo, hi)
