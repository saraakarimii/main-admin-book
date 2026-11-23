"""Per-image layout report: final artwork size, palette-guide size, physical cell size."""
from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from .geometry import SIDE_BY_SIDE, FULL_PAGE, all_plans, best_plan, GuideSpec


def _r(x): return round(float(x), 3)


def _plan_dict(p, layout):
    return {"layout": p.layout, "guide_placement": p.placement, "grid": {"cols": p.cols, "rows": p.rows},
            "slot_mm": [_r(v) for v in p.slot_mm],
            "guide_mm": {"width": _r(p.guide["w"]), "height": _r(p.guide["h"]), "columns": p.guide["cols"],
                         "rows": p.guide["rows"], "entries": p.n_colors},
            "spacing_artwork_to_guide_mm": p.spacing_mm,
            "artwork_area_available_mm": [_r(v) for v in p.artwork_area_mm],
            "artwork_final_mm": {"width": _r(p.artwork_mm[0]), "height": _r(p.artwork_mm[1])},
            "cell_mm": _r(p.cell_mm), "meets_min_cell": p.meets_min_cell,
            "slot_positions_on_page_mm": [[_r(v) for v in b] for b in layout.slot_boxes()],
            "boxes_in_slot_mm": {k: [_r(v) for v in b] for k, b in p.boxes.items()}}


def layout_report(items, min_cell_mm=3.0, guide=GuideSpec()):
    """items: [{name, cols, rows, n_colors_used}] -> report dict (one entry per image)."""
    pages = {"side_by_side": SIDE_BY_SIDE, "full_page": FULL_PAGE}
    out = {"page_specs": {n: {"page_mm": [lo.page_w, lo.page_h], "inner": lo.inner, "outer": lo.outer, "top": lo.top,
                              "bottom": lo.bottom, "gap": lo.gap, "usable_mm": [_r(lo.usable_w), _r(lo.usable_h)],
                              "slot_mm": [_r(lo.slot_w), _r(lo.slot_h)], "slots_per_page": lo.slots,
                              "reading_order": lo.reading_order, "spiral": "left edge"} for n, lo in pages.items()},
           "min_cell_mm": min_cell_mm, "classification_rule": "physical cell size only; side_by_side first, else full_page, else rejected",
           "images": []}
    for it in items:
        chosen = best_plan(it["cols"], it["rows"], it["n_colors_used"], guide=guide, min_cell_mm=min_cell_mm)
        alts = [_plan_dict(p, pages[p.layout]) for p in all_plans(it["cols"], it["rows"], it["n_colors_used"], guide=guide, min_cell_mm=min_cell_mm)]
        out["images"].append({"image": it["name"], "colors_used": it["n_colors_used"],
                              "classification": chosen.layout if chosen else "REJECTED_cell_below_min",
                              "chosen": _plan_dict(chosen, pages[chosen.layout]) if chosen else None,
                              "all_options": alts})
    return out


def write_layout_report(rep, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "layout_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    L = ["# Layout report", "", f"Minimum cell size: {rep['min_cell_mm']} mm. Rule: {rep['classification_rule']}.", ""]
    for n, s in rep["page_specs"].items():
        L.append(f"- **{n}**: page {s['page_mm'][0]}x{s['page_mm'][1]} mm, usable {s['usable_mm'][0]}x{s['usable_mm'][1]} mm, "
                 f"slot {s['slot_mm'][0]}x{s['slot_mm'][1]} mm, margins inner {s['inner']} / outer {s['outer']} / top-bottom {s['top']}, gap {s['gap']}")
    L += ["", "| image | layout | guide | grid | artwork (mm) | guide (mm) | cell (mm) | >= min |", "|---|---|---|---|---|---|---|---|"]
    for im in rep["images"]:
        c = im["chosen"]
        if c:
            L.append(f"| {im['image']} | {c['layout']} | {c['guide_placement']} | {c['grid']['cols']}x{c['grid']['rows']} | "
                     f"{c['artwork_final_mm']['width']} x {c['artwork_final_mm']['height']} | {c['guide_mm']['width']} x {c['guide_mm']['height']} "
                     f"({c['guide_mm']['columns']}c x {c['guide_mm']['rows']}r) | {c['cell_mm']} | {c['meets_min_cell']} |")
        else:
            L.append(f"| {im['image']} | REJECTED | - | - | - | - | - | False |")
    L += ["", "## All options per image (every layout x guide placement)"]
    for im in rep["images"]:
        L += ["", f"### {im['image']}", "| layout | guide | available area (mm) | artwork (mm) | cell (mm) | >= min |", "|---|---|---|---|---|---|"]
        for c in im["all_options"]:
            L.append(f"| {c['layout']} | {c['guide_placement']} | {c['artwork_area_available_mm'][0]} x {c['artwork_area_available_mm'][1]} | "
                     f"{c['artwork_final_mm']['width']} x {c['artwork_final_mm']['height']} | {c['cell_mm']} | {c['meets_min_cell']} |")
    (out / "layout_report.md").write_text("\n".join(L), encoding="utf-8")
    return {"json": str(out / "layout_report.json"), "md": str(out / "layout_report.md")}
