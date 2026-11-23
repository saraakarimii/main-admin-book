from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Geometry:
    rows: int
    cols: int
    xe: np.ndarray   # column edges in working-image pixels (cols+1)
    ye: np.ndarray   # row edges (rows+1)

    @property
    def shape(self):
        return (self.rows, self.cols)


def make_geometry(h: int, w: int, cols: int, rows: int | None = None) -> Geometry:
    cols = int(np.clip(cols, 4, 180))
    if rows is None:
        rows = max(4, int(round(cols * h / w)))
    xe = np.linspace(0, w, cols + 1).round().astype(int)
    ye = np.linspace(0, h, rows + 1).round().astype(int)
    return Geometry(int(rows), cols, xe, ye)
