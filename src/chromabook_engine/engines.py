"""The four mapping engines. Pure functions of explicit inputs (no globals)."""
from __future__ import annotations
import time
from dataclasses import dataclass
import numpy as np
from .grid import Geometry
from .icm import boundary_pairwise_weights, icm_optimize
from .unary import build_region_balanced_unary

ENGINE_NAMES = {"A": "A - Legacy Mean-Lab / dE76", "B": "B - Cell-wise CIEDE2000",
                "C": "C - Edge-Aware ICM", "D": "D - SLIC Region-Aware ICM"}
# What the stored per-cell cost actually means for each engine (never label all as dE00).
COST_KINDS = {
    "A": "deltaE76_cell_mean_lab_to_selected_palette_color",
    "B": "ciede2000_mean_per_pixel_trimmed_selected_label_unary",
    "C": "ciede2000_mean_per_pixel_trimmed_selected_label_unary",
    "D": "ciede2000_blend(cellwise,region_balanced)_selected_label_unary",
}


@dataclass
class EngineResult:
    key: str
    name: str
    labels: np.ndarray            # (rows, cols) int32 palette INDICES (not printed IDs)
    selected_cost: np.ndarray     # (rows, cols) float32
    cost_kind: str
    runtime_seconds: float


def legacy_mean_lab(lab, geo: Geometry, pal_lab):
    labels = np.zeros(geo.shape, np.int32)
    cost = np.zeros(geo.shape, np.float32)
    for r in range(geo.rows):
        for c in range(geo.cols):
            cell = lab[geo.ye[r]:geo.ye[r + 1], geo.xe[c]:geo.xe[c + 1]]
            d = np.linalg.norm(pal_lab - cell.reshape(-1, 3).mean(0)[None, :], axis=1)
            labels[r, c] = int(np.argmin(d)); cost[r, c] = float(d[labels[r, c]])
    return labels, cost


def _take(unary, labels):
    return np.take_along_axis(unary, labels[:, :, None], axis=2)[:, :, 0]


def run_engine(key, *, lab, geo, pal_lab, unary, edge_map, slic_labels, slic_boundary, mapping) -> EngineResult:
    t0 = time.time()
    if key == "A":
        labels, cost = legacy_mean_lab(lab, geo, pal_lab)
    elif key == "B":
        labels = np.argmin(unary, axis=2).astype(np.int32); cost = _take(unary, labels)
    elif key == "C":
        wh, wv = boundary_pairwise_weights(edge_map, geo, mapping.boundary_beta)
        labels = icm_optimize(unary, wh, wv, mapping.smoothness_lambda, mapping.icm_iterations)
        cost = _take(unary, labels)
    elif key == "D":
        if slic_labels is None:
            raise RuntimeError("Engine D needs SLIC labels")
        reg = build_region_balanced_unary(lab, slic_labels, geo, pal_lab, mapping.max_samples,
                                          mapping.trim_fraction)
        b = float(mapping.region_balance)
        du = (1.0 - b) * unary + b * reg
        combined = np.maximum(edge_map, 0.95 * slic_boundary)
        wh, wv = boundary_pairwise_weights(combined, geo, mapping.boundary_beta)
        labels = icm_optimize(du, wh, wv, mapping.smoothness_lambda, mapping.icm_iterations)
        cost = _take(du, labels)
    else:
        raise ValueError(f"unknown engine {key!r}")
    return EngineResult(key, ENGINE_NAMES[key], labels.astype(np.int32), cost.astype(np.float32),
                        COST_KINDS[key], time.time() - t0)
