"""Quality metrics. They compare candidates; they do not certify a printed result."""
from __future__ import annotations
import cv2
import numpy as np
from scipy import ndimage
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import structural_similarity

_STRUCT = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


def grid_metrics(labels):
    total = int(labels.size)
    singleton = tiny = comps = 0
    for idx in np.unique(labels):
        cc, n = ndimage.label(labels == idx, structure=_STRUCT)
        if n:
            sizes = np.bincount(cc.ravel())[1:]
            singleton += int(np.sum(sizes == 1))
            tiny += int(np.sum(sizes <= 2))   # NOTE: includes singletons (notebook definition)
            comps += int(n)
    h = int(np.sum(labels[:, 1:] != labels[:, :-1]))
    v = int(np.sum(labels[1:, :] != labels[:-1, :]))
    adj = labels.shape[0] * max(0, labels.shape[1] - 1) + max(0, labels.shape[0] - 1) * labels.shape[1]
    return {"singleton_ratio_pct": 100.0 * singleton / max(1, total),
            "tiny_component_ratio_pct": 100.0 * tiny / max(1, total),
            "connected_components": comps,
            "transition_rate_pct": 100.0 * (h + v) / max(1, adj)}


def edge_f1(reference_rgb, mapped_rgb, lo=55, hi=150):
    a = cv2.Canny(cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY), lo, hi)
    b = cv2.Canny(cv2.cvtColor(mapped_rgb, cv2.COLOR_RGB2GRAY), lo, hi)
    k = np.ones((3, 3), np.uint8)
    ad, bd = cv2.dilate(a, k) > 0, cv2.dilate(b, k) > 0
    ab, bb = a > 0, b > 0
    p = float(np.sum(bb & ad) / max(1, np.sum(bb)))
    r = float(np.sum(ab & bd) / max(1, np.sum(ab)))
    return 2 * p * r / max(1e-12, p + r), p, r


def evaluate_grid(reference_rgb, processed_rgb, mapped_rgb, labels, palette_size):
    ref, proc, mp = (rgb2lab(x.astype(np.float32) / 255.0) for x in (reference_rgb, processed_rgb, mapped_rgb))
    de_ref, de_proc = deltaE_ciede2000(ref, mp), deltaE_ciede2000(proc, mp)
    try:
        ssim = float(structural_similarity(reference_rgb, mapped_rgb, channel_axis=-1, data_range=255, win_size=7))
    except Exception:
        ssim = float("nan")
    f1, p, r = edge_f1(reference_rgb, mapped_rgb)
    used = len(np.unique(labels))
    return {"DeltaE00_mean_vs_source": float(de_ref.mean()),
            "DeltaE00_median_vs_source": float(np.median(de_ref)),
            "DeltaE00_p95_vs_source": float(np.percentile(de_ref, 95)),
            "DeltaE00_mean_vs_processed": float(de_proc.mean()),
            "SSIM_vs_source": ssim, "Edge_F1_1px": f1, "Edge_precision": p, "Edge_recall": r,
            "colors_used": int(used), "palette_utilization_pct": 100.0 * used / max(1, palette_size),
            **grid_metrics(labels)}, de_ref


def delta_map_rgb(de):
    vmax = max(1.0, float(np.percentile(de, 99)))
    x = np.clip(de / vmax * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(x, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def palette_usage(labels, palette):
    u, c = np.unique(labels, return_counts=True)
    rows = [{"color_id": palette.id_of(i), "name": palette.colors[int(i)].name,
             "hex": palette.colors[int(i)].hex, "palette_index": int(i),
             "cells": int(n), "share_pct": 100.0 * float(n) / labels.size} for i, n in zip(u, c)]
    return sorted(rows, key=lambda x: -x["cells"])


def grid_integrity(labels, palette, preview_cells=None):
    """Hard checks used by tests and reports."""
    ok_range = bool(labels.min() >= 0 and labels.max() < len(palette))
    return {"rows": int(labels.shape[0]), "cols": int(labels.shape[1]),
            "all_cells_have_one_valid_index": ok_range,
            "dtype_fits_uint8": bool(len(palette) <= 256)}
