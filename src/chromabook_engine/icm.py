"""Boundary-weighted pairwise terms and deterministic raster-scan ICM (approximate, not Graph Cut)."""
from __future__ import annotations
import numpy as np
from .grid import Geometry


def boundary_pairwise_weights(boundary_map, geo: Geometry, beta=4.0):
    h, w = boundary_map.shape
    rows, cols, xe, ye = geo.rows, geo.cols, geo.xe, geo.ye
    wh = np.ones((rows, max(0, cols - 1)), np.float32)
    wv = np.ones((max(0, rows - 1), cols), np.float32)
    rad = 2
    for r in range(rows):
        y0, y1 = ye[r], ye[r + 1]
        for c in range(cols - 1):
            x = int(xe[c + 1]); xa, xb = max(0, x - rad), min(w, x + rad + 1)
            s = float(np.mean(boundary_map[y0:y1, xa:xb])) if y1 > y0 and xb > xa else 0.0
            wh[r, c] = np.exp(-float(beta) * np.clip(s, 0, 1))
    for r in range(rows - 1):
        y = int(ye[r + 1]); ya, yb = max(0, y - rad), min(h, y + rad + 1)
        for c in range(cols):
            x0, x1 = xe[c], xe[c + 1]
            s = float(np.mean(boundary_map[ya:yb, x0:x1])) if yb > ya and x1 > x0 else 0.0
            wv[r, c] = np.exp(-float(beta) * np.clip(s, 0, 1))
    return wh, wv


def _icm_py(unary, wh, wv, lam, iterations):
    rows, cols, k = unary.shape
    labels = np.argmin(unary, axis=2).astype(np.int32)
    ids = np.arange(k)
    for _ in range(int(iterations)):
        changes = 0
        for r in range(rows):
            for c in range(cols):
                costs = unary[r, c].copy()
                if c > 0:        costs += lam * wh[r, c - 1] * (ids != labels[r, c - 1])
                if c < cols - 1: costs += lam * wh[r, c] * (ids != labels[r, c + 1])
                if r > 0:        costs += lam * wv[r - 1, c] * (ids != labels[r - 1, c])
                if r < rows - 1: costs += lam * wv[r, c] * (ids != labels[r + 1, c])
                new = int(np.argmin(costs))
                if new != labels[r, c]:
                    labels[r, c] = new; changes += 1
        if changes == 0:
            break
    return labels


def icm_optimize(unary, wh, wv, lam=1.8, iterations=5):
    return _icm_py(unary, wh, wv, float(lam), iterations)
