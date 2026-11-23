"""Physical-size constraints applied INSIDE the candidate search.

The classification rule (Phase 5) uses cell size only. Here the minimum cell size is a hard filter, so
no candidate is generated that cannot reach a printable layout. Panel size is an explicit input
(defaults are assumptions until Phase 5 fixes the layout classes).
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PrintSpec:
    panel_width_mm: float = 130.0
    panel_height_mm: float = 180.0
    min_cell_mm: float = 3.0
    number_height_ratio: float = 0.5      # printed number height as a fraction of the cell
    min_number_height_mm: float = 1.5     # legibility floor for a 2-character ID
    id_chars: int = 2


def rows_for(cols: int, aspect_h_over_w: float) -> int:
    return max(4, int(round(cols * aspect_h_over_w)))


def cell_size_mm(spec: PrintSpec, cols: int, rows: int) -> float:
    if hasattr(spec, "cell_mm"):          # geometry-backed spec (layout.LayoutPrintSpec)
        return spec.cell_mm(cols, rows)
    return min(spec.panel_width_mm / cols, spec.panel_height_mm / rows)


def feasible(spec: PrintSpec, cols: int, rows: int) -> dict:
    cell = cell_size_mm(spec, cols, rows)
    num_h = cell * spec.number_height_ratio
    # a 3-character ID needs roughly 1.5x the width of a 2-character one
    need = spec.min_number_height_mm * (1.0 if spec.id_chars <= 2 else 1.5)
    return {"cols": cols, "rows": rows, "cell_mm": round(cell, 3),
            "meets_min_cell": cell + 1e-9 >= spec.min_cell_mm,
            "number_height_mm": round(num_h, 3), "number_legible": num_h + 1e-9 >= need}


def feasible_cols(spec: PrintSpec, aspect_h_over_w: float, lo: int = 16, hi: int = 120, step: int = 6) -> list[int]:
    out = []
    for c in range(lo, hi + 1, step):
        f = feasible(spec, c, rows_for(c, aspect_h_over_w))
        if f["meets_min_cell"] and f["number_legible"]:
            out.append(c)
    return out
