"""Framing PROPOSALS. Different crops show different content, so their metrics are not comparable;
the framing is therefore chosen by a conservative rule and the alternatives are shown, not ranked."""
from __future__ import annotations
import cv2
import numpy as np
from ..color import rgb_to_lab
from .diagnostics import _small


def _border_trim(lab, max_frac=0.20, thr=7.0):
    h, w = lab.shape[:2]
    ref = np.median(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]), axis=0)
    dev = np.linalg.norm(lab - ref, axis=2)
    def run(vec, limit):
        n = 0
        for v in vec[:limit]:
            if v < 0.015: n += 1       # <1.5 % of pixels in the band differ from the border colour
            else: break
        return n
    rowdev = (dev > thr).mean(axis=1); coldev = (dev > thr).mean(axis=0)
    t = run(rowdev, int(h * max_frac)); b = run(rowdev[::-1], int(h * max_frac))
    l = run(coldev, int(w * max_frac)); r = run(coldev[::-1], int(w * max_frac))
    return l / w, t / h, 1 - r / w, 1 - b / h


def _subject_box(lab, margin=0.04):
    h, w = lab.shape[:2]
    ref = np.median(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]), axis=0)
    dev = np.linalg.norm(lab - ref, axis=2)
    g = cv2.GaussianBlur(lab[:, :, 0], (0, 0), 2)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    sal = dev / (dev.max() + 1e-6) + 0.5 * grad / (grad.max() + 1e-6)
    sal8 = np.clip(sal / (sal.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
    _, m = cv2.threshold(cv2.GaussianBlur(sal8, (0, 0), 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    ys, xs = np.where(m > 0)
    if len(xs) < 50: return None
    x0, x1 = np.percentile(xs, [1, 99]); y0, y1 = np.percentile(ys, [1, 99])
    return (max(0.0, x0 / w - margin), max(0.0, y0 / h - margin), min(1.0, x1 / w + margin), min(1.0, y1 / h + margin))


def _area(b): return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def propose_framings(rgb) -> list[dict]:
    """Returns [{id, crop (normalised x0,y0,x1,y1), reason, default}]. 'full' is always present."""
    lab = rgb_to_lab(_small(rgb, 384))
    out = [{"id": "full", "crop": (0.0, 0.0, 1.0, 1.0), "reason": "whole image", "default": True}]
    trim = _border_trim(lab)
    if _area(trim) < 0.97:
        out.append({"id": "trim_border", "crop": tuple(round(float(v), 4) for v in trim),
                    "reason": "removes near-uniform border bands (<=20 % per side)", "default": False})
    box = _subject_box(lab)
    if box is not None and 0.15 <= _area(box) <= 0.80:
        base = trim if _area(trim) < 0.97 else (0, 0, 1, 1)
        if _area(box) < 0.92 * _area(base):
            out.append({"id": "subject", "crop": tuple(round(float(v), 4) for v in box),
                        "reason": "tight box around the salient subject (saliency = colour distance from border + gradient)",
                        "default": False})
    # safe default: trimming empty borders loses no content; a subject crop is an owner decision
    if len(out) > 1 and out[1]["id"] == "trim_border":
        out[0]["default"] = False; out[1]["default"] = True
    return out
