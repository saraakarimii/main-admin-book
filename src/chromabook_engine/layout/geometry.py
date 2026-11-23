"""Exact A4 layout geometry. Nothing here is hardcoded to a panel size: slots are DERIVED from the page,
margins and gap; the artwork area is what remains after the palette guide and spacing are allocated.

Classification uses physical cell size only (never image-quality metrics)."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from ..optimize.constraints import PrintSpec

MM = 72.0 / 25.4   # PDF points per millimetre


@dataclass(frozen=True)
class PageLayout:
    name: str
    page_w: float
    page_h: float
    inner: float            # margin on the spiral (left) edge
    outer: float
    top: float
    bottom: float
    slots: int              # artworks per page
    gap: float = 0.0        # centre gap between slots
    reading_order: str = "right_to_left"   # slot fill order for 2-slot pages (as stated: right then left)

    @property
    def usable_w(self): return self.page_w - self.inner - self.outer
    @property
    def usable_h(self): return self.page_h - self.top - self.bottom
    @property
    def slot_w(self): return (self.usable_w - self.gap * (self.slots - 1)) / self.slots
    @property
    def slot_h(self): return self.usable_h

    def slot_boxes(self):
        """Left-to-right physical boxes (x, y_from_top, w, h) in mm. Spiral is on the LEFT edge."""
        return [(self.inner + i * (self.slot_w + self.gap), self.top, self.slot_w, self.slot_h) for i in range(self.slots)]

    def fill_order(self):
        idx = list(range(self.slots))
        return idx[::-1] if self.reading_order == "right_to_left" else idx


SIDE_BY_SIDE = PageLayout("side_by_side", 297.0, 210.0, 18.0, 12.0, 12.0, 12.0, slots=2, gap=8.0)
FULL_PAGE = PageLayout("full_page", 210.0, 297.0, 18.0, 12.0, 12.0, 12.0, slots=1)


@dataclass(frozen=True)
class GuideSpec:
    entry_w: float = 36.0       # swatch + ID + colour name
    entry_h: float = 7.0
    swatch: float = 6.0
    spacing: float = 4.0        # between artwork and its guide
    column_gap: float = 2.0
    min_artwork: float = 30.0   # refuse absurd leftovers


def _guide_dims(n_colors, avail_w, avail_h, placement, g: GuideSpec):
    """Smallest guide footprint for n colours. below: spans the slot width; side: spans the slot height."""
    if n_colors <= 0: return None
    if placement == "below":
        cols = max(1, int((avail_w + g.column_gap) // (g.entry_w + g.column_gap)))
        rows = math.ceil(n_colors / cols)
        used_cols = min(cols, n_colors)
        w = used_cols * g.entry_w + (used_cols - 1) * g.column_gap
        return {"cols": used_cols, "rows": rows, "w": w, "h": rows * g.entry_h}
    per_col = max(1, int(avail_h // g.entry_h))
    cols = math.ceil(n_colors / per_col)
    rows = math.ceil(n_colors / cols)
    return {"cols": cols, "rows": rows, "w": cols * g.entry_w + (cols - 1) * g.column_gap, "h": rows * g.entry_h}


@dataclass
class SlotPlan:
    layout: str
    placement: str                 # below | side
    slot_mm: tuple                 # (w, h)
    guide: dict                    # dims in mm incl. columns/rows
    spacing_mm: float
    artwork_area_mm: tuple         # space left for the grid after guide + spacing
    artwork_mm: tuple              # actual drawn grid size = cols*cell x rows*cell
    cols: int
    rows: int
    cell_mm: float
    meets_min_cell: bool
    boxes: dict                    # positions in mm relative to the slot origin (top-left)
    n_colors: int


def plan_slot(layout: PageLayout, cols: int, rows: int, n_colors: int, placement: str,
              guide: GuideSpec = GuideSpec(), min_cell_mm: float = 3.0) -> SlotPlan | None:
    sw, sh = layout.slot_w, layout.slot_h
    d = _guide_dims(n_colors, sw, sh, placement, guide)
    if d is None: return None
    if placement == "below":
        area = (sw, sh - d["h"] - guide.spacing)
    else:
        area = (sw - d["w"] - guide.spacing, sh)
    if area[0] < guide.min_artwork or area[1] < guide.min_artwork: return None
    cell = min(area[0] / cols, area[1] / rows)
    aw, ah = cols * cell, rows * cell
    if placement == "below":
        art = ((sw - aw) / 2, 0.0); gd = ((sw - d["w"]) / 2, sh - d["h"])
    else:
        art = (0.0, (sh - ah) / 2); gd = (sw - d["w"], (sh - d["h"]) / 2)
    return SlotPlan(layout.name, placement, (sw, sh), d, guide.spacing, area, (aw, ah), cols, rows, cell,
                    cell + 1e-9 >= min_cell_mm, {"artwork": (*art, aw, ah), "guide": (*gd, d["w"], d["h"])}, n_colors)


def all_plans(cols, rows, n_colors, layouts=(SIDE_BY_SIDE, FULL_PAGE), guide=GuideSpec(), min_cell_mm=3.0):
    out = []
    for lo in layouts:
        for pl in ("below", "side"):
            p = plan_slot(lo, cols, rows, n_colors, pl, guide, min_cell_mm)
            if p: out.append(p)
    return out


def best_plan(cols, rows, n_colors, layouts=(SIDE_BY_SIDE, FULL_PAGE), guide=GuideSpec(), min_cell_mm=3.0) -> SlotPlan | None:
    """Classification by physical cell size only: layouts are tried in the given order (side-by-side first);
    the first layout whose best guide placement reaches min_cell wins. None if no layout reaches it."""
    for lo in layouts:
        ps = [p for p in all_plans(cols, rows, n_colors, (lo,), guide, min_cell_mm) if p.meets_min_cell]
        if ps: return max(ps, key=lambda p: p.cell_mm)
    return None


@dataclass(frozen=True)
class LayoutPrintSpec(PrintSpec):
    """PrintSpec backed by the real page geometry. Guide space is sized for the FULL palette (worst case) while
    searching, so the real layout (fewer colours -> smaller guide) can only have equal or larger cells."""
    layouts: tuple = (SIDE_BY_SIDE, FULL_PAGE)
    guide: GuideSpec = GuideSpec()
    palette_size: int = 21

    def cell_mm(self, cols: int, rows: int) -> float:
        best = 0.0
        for lo in self.layouts:
            for p in all_plans(cols, rows, self.palette_size, (lo,), self.guide, self.min_cell_mm):
                best = max(best, p.cell_mm)
        return best
