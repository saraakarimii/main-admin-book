"""Per-cell colour cost volumes (unary terms).

sampling="legacy"      -> notebook-identical flat linspace sampling (used for parity tests).
sampling="stratified"  -> fixed S = sy*sx grid of sample positions inside every cell; unbiased with
                          respect to cell shape and fully vectorised across cells.
"""
from __future__ import annotations
import numpy as np
from .color import delta_e00_matrix
from .grid import Geometry


def sampled_pixels(arr, max_samples):
    flat = arr.reshape(-1, arr.shape[-1])
    if len(flat) <= max_samples:
        return flat
    idx = np.linspace(0, len(flat) - 1, int(max_samples)).astype(np.int32)
    return flat[idx]


def trimmed_mean(costs, frac=0.0, axis=0):
    """Trimmed mean of costs along `axis` (sorted)."""
    frac = float(np.clip(frac, 0, 0.40))
    n = costs.shape[axis]
    if frac <= 0 or n < 5:
        return costs.mean(axis=axis)
    cut = min(int(n * frac), max(0, (n - 1) // 2))
    if not cut:
        return costs.mean(axis=axis)
    s = np.sort(costs, axis=axis)
    sl = [slice(None)] * costs.ndim
    sl[axis] = slice(cut, n - cut)
    return s[tuple(sl)].mean(axis=axis)


def _legacy_unary(lab, geo: Geometry, pal_lab, max_samples, trim):
    out = np.zeros((geo.rows, geo.cols, len(pal_lab)), np.float32)
    for r in range(geo.rows):
        for c in range(geo.cols):
            cell = lab[geo.ye[r]:geo.ye[r + 1], geo.xe[c]:geo.xe[c + 1]]
            pix = sampled_pixels(cell, max_samples).astype(np.float32)
            out[r, c] = trimmed_mean(delta_e00_matrix(pix, pal_lab), trim, axis=0)
    return out


def _stratified_unary(lab, geo: Geometry, pal_lab, max_samples, trim):
    sy = max(1, int(np.floor(np.sqrt(max_samples))))
    sx = max(1, int(np.ceil(max_samples / sy)))
    fy = (np.arange(sy) + 0.5) / sy
    fx = (np.arange(sx) + 0.5) / sx
    K = len(pal_lab)
    out = np.zeros((geo.rows, geo.cols, K), np.float32)
    x0, x1 = geo.xe[:-1], geo.xe[1:]
    xs = np.minimum(x0[:, None] + np.floor(fx[None, :] * (x1 - x0)[:, None]).astype(int), x1[:, None] - 1)
    xs = np.maximum(xs, x0[:, None])                                   # (C, sx)
    step = max(1, int(2_000_000 // (geo.cols * sy * sx * K)))          # row-chunking bounds memory
    for r0 in range(0, geo.rows, step):
        r1 = min(geo.rows, r0 + step)
        ya, yb = geo.ye[r0:r1], geo.ye[r0 + 1:r1 + 1]
        ys = np.minimum(ya[:, None] + np.floor(fy[None, :] * (yb - ya)[:, None]).astype(int), yb[:, None] - 1)
        ys = np.maximum(ys, ya[:, None])                               # (R, sy)
        pix = lab[ys[:, None, :, None], xs[None, :, None, :]]          # (R, C, sy, sx, 3)
        pix = pix.reshape(r1 - r0, geo.cols, sy * sx, 3)
        d = delta_e00_matrix(pix.reshape(-1, 3), pal_lab).reshape(r1 - r0, geo.cols, sy * sx, K)
        out[r0:r1] = trimmed_mean(d, trim, axis=2)
    return out


def build_cell_unary(lab, geo, pal_lab, max_samples=96, trim=0.03, sampling="stratified"):
    if sampling == "legacy":
        return _legacy_unary(lab, geo, pal_lab, max_samples, trim)
    if sampling == "stratified":
        return _stratified_unary(lab, geo, pal_lab, max_samples, trim)
    raise ValueError(f"unknown sampling mode {sampling!r}")


def build_region_balanced_unary(lab, slic_labels, geo: Geometry, pal_lab, max_samples=72, trim=0.03):
    out = np.zeros((geo.rows, geo.cols, len(pal_lab)), np.float32)
    for r in range(geo.rows):
        for c in range(geo.cols):
            y0, y1, x0, x1 = geo.ye[r], geo.ye[r + 1], geo.xe[c], geo.xe[c + 1]
            cell_lab, cell_seg = lab[y0:y1, x0:x1], slic_labels[y0:y1, x0:x1]
            ids, counts = np.unique(cell_seg, return_counts=True)
            costs, weights = [], []
            for sid, count in zip(ids, counts):
                region = cell_lab[cell_seg == sid]
                pix = sampled_pixels(region, max(12, min(max_samples, int(max_samples * count / max(1, cell_seg.size) * 2))))
                costs.append(trimmed_mean(delta_e00_matrix(pix.astype(np.float32), pal_lab), trim, axis=0))
                weights.append(np.sqrt(float(count)))
            w = np.asarray(weights, np.float32)
            w /= max(1e-8, w.sum())
            out[r, c] = np.sum(np.asarray(costs) * w[:, None], axis=0)
    return out
