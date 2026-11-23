import re, shutil, subprocess, tempfile
from pathlib import Path
import numpy as np
from helpers import PAL, synth_source
from chromabook_engine.layout import (SIDE_BY_SIDE, FULL_PAGE, LayoutPrintSpec, MM, plan_slot, best_plan,
                                      layout_report, build_pdf)
from chromabook_engine.layout.geometry import all_plans
from chromabook_engine.optimize import optimize, OptimizerConfig

FAST = OptimizerConfig(preview_side=160, final_side=240, keep_preprocess=1, keep_combos=1, stability_top=1)


def _labels(rows, cols, k=8, seed=3):
    return np.random.default_rng(seed).integers(0, k, (rows, cols)).astype(np.int32)


def test_page_geometry_matches_the_confirmed_a4_specs():
    s, f = SIDE_BY_SIDE, FULL_PAGE
    assert (s.page_w, s.page_h) == (297, 210) and s.usable_w == 267 and s.slot_w == 129.5 and s.usable_h == 186
    assert (f.page_w, f.page_h) == (210, 297) and f.usable_w == 180 and f.usable_h == 273 and f.slot_w == 180
    b = s.slot_boxes()
    assert b[0][0] == 18 and b[1][0] == 18 + 129.5 + 8 and b[1][0] + b[1][2] == 297 - 12   # spiral/inner margin on the LEFT
    assert all(x[1] == 12 and x[3] == 186 for x in b)
    assert s.fill_order() == [1, 0]                                                        # right artwork first


def test_artwork_area_is_what_remains_after_guide_and_spacing():
    for placement in ("below", "side"):
        p = plan_slot(SIDE_BY_SIDE, 28, 42, 21, placement)
        gw, gh = p.guide["w"], p.guide["h"]
        if placement == "below":
            assert abs(p.artwork_area_mm[1] - (186 - gh - p.spacing_mm)) < 1e-9 and p.artwork_area_mm[0] == 129.5
        else:
            assert abs(p.artwork_area_mm[0] - (129.5 - gw - p.spacing_mm)) < 1e-9 and p.artwork_area_mm[1] == 186
        aw, ah = p.artwork_mm
        assert aw <= p.artwork_area_mm[0] + 1e-9 and ah <= p.artwork_area_mm[1] + 1e-9
        assert abs(aw - 28 * p.cell_mm) < 1e-9 and abs(ah - 42 * p.cell_mm) < 1e-9           # cell is derived, not assumed
        ax, ay, _, _ = p.boxes["artwork"]; gx, gy, _, _ = p.boxes["guide"]
        if placement == "below": assert ay + ah + p.spacing_mm <= gy + 1e-9
        else: assert ax + aw + p.spacing_mm <= gx + 1e-9
        assert gx >= -1e-9 and gy >= -1e-9 and gx + gw <= 129.5 + 1e-9 and gy + gh <= 186 + 1e-9


def test_no_hardcoded_panel_size():
    sbs = LayoutPrintSpec(layouts=(SIDE_BY_SIDE,))
    assert abs(sbs.cell_mm(28, 42) - 3.2) < 0.05 and 130 / 28 > 4.5   # a whole-panel 130 mm assumption would give 4.6 mm


def test_classification_by_cell_size_only_and_min_cell_enforced():
    assert best_plan(28, 42, 17).layout == "side_by_side"
    assert best_plan(40, 60, 21).layout == "full_page"            # too fine for side-by-side (<3 mm), fine on a full page
    assert best_plan(90, 135, 21) is None                          # below 3 mm everywhere -> rejected
    assert all(p.cell_mm >= 3.0 for p in [best_plan(28, 42, 17), best_plan(40, 60, 21)])


def test_fewer_colours_never_make_cells_smaller():
    a = best_plan(28, 42, 21); b = best_plan(28, 42, 8)
    assert b.cell_mm >= a.cell_mm - 1e-9 and b.guide["h"] <= a.guide["h"]


def test_layout_report_has_required_fields_for_every_image():
    rep = layout_report([{"name": "a", "cols": 28, "rows": 42, "n_colors_used": 17},
                         {"name": "b", "cols": 40, "rows": 60, "n_colors_used": 21},
                         {"name": "c", "cols": 90, "rows": 135, "n_colors_used": 21}])
    assert [i["classification"] for i in rep["images"]] == ["side_by_side", "full_page", "REJECTED_cell_below_min"]
    for im in rep["images"][:2]:
        c = im["chosen"]
        assert c["artwork_final_mm"]["width"] > 0 and c["guide_mm"]["width"] > 0 and c["cell_mm"] >= 3.0
    assert rep["page_specs"]["side_by_side"]["slot_mm"] == [129.5, 186.0]


def test_optimizer_uses_real_geometry_as_hard_constraint():
    sbs = optimize(synth_source(h=300, w=200), PAL, LayoutPrintSpec(layouts=(SIDE_BY_SIDE,), palette_size=len(PAL)), FAST)
    full = optimize(synth_source(h=300, w=200), PAL, LayoutPrintSpec(layouts=(FULL_PAGE,), palette_size=len(PAL)), FAST)
    assert all(c["cell_mm"] >= 3.0 - 1e-9 for c in sbs.candidates + full.candidates)
    assert max(sbs.stages["constraint"]["allowed_cols"]) < max(full.stages["constraint"]["allowed_cols"])
    r, c = sbs.labels.shape
    assert best_plan(c, r, sbs.n_colors_used, (SIDE_BY_SIDE,)).cell_mm >= 3.0      # real layout (actual colours) also passes


def _bbox_words(pdf):
    if not shutil.which("pdftotext"): return None
    with tempfile.TemporaryDirectory() as d:
        o = Path(d) / "o.html"; subprocess.run(["pdftotext", "-bbox", str(pdf), str(o)], check=True)
        return [tuple(float(v) for v in m) for m in re.findall(r'xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)"', o.read_text())]


def test_pdf_is_vector_a4_inside_margins_and_right_slot_first():
    from pypdf import PdfReader
    with tempfile.TemporaryDirectory() as d:
        for lay, size in ((SIDE_BY_SIDE, (297, 210)), (FULL_PAGE, (210, 297))):
            lab = _labels(42, 28) if lay is SIDE_BY_SIDE else _labels(60, 40)
            pdf = Path(d) / f"{lay.name}.pdf"
            placed = build_pdf(pdf, lay, [{"name": "x", "labels": lab}], PAL)
            r = PdfReader(str(pdf)); pg = r.pages[0]
            assert abs(float(pg.mediabox.width) - size[0] * MM) < 0.1 and abs(float(pg.mediabox.height) - size[1] * MM) < 0.1
            assert "/Image" not in str(pg.get("/Resources", {}).get("/XObject", {}))     # no raster artwork
            assert placed[0]["plan"].cell_mm >= 3.0
            words = _bbox_words(pdf)
            if words:
                # guide text (colour names) must never fall inside the drawn grid rectangle
                pl = placed[0]["plan"]; bx = lay.slot_boxes()[placed[0]["slot_index_left_to_right"]]
                ax, ay, aw, ah = pl.boxes["artwork"]
                gx0, gy0 = (bx[0] + ax) * MM, (bx[1] + ay) * MM; gx1, gy1 = gx0 + aw * MM, gy0 + ah * MM
                with tempfile.TemporaryDirectory() as d2:
                    o = Path(d2) / "o.html"; subprocess.run(["pdftotext", "-bbox", str(pdf), str(o)], check=True)
                    named = [(float(a), float(b), float(c_), float(d_)) for a, b, c_, d_, t in re.findall(
                        r'xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)<', o.read_text()) if re.search("[A-Za-z]", t)]
                assert named and all(w[2] <= gx0 + 0.5 or w[0] >= gx1 - 0.5 or w[3] <= gy0 + 0.5 or w[1] >= gy1 - 0.5 for w in named)
                x0, y0 = min(w[0] for w in words), min(w[1] for w in words)
                x1, y1 = max(w[2] for w in words), max(w[3] for w in words)
                assert x0 >= lay.inner * MM - 1 and x1 <= (lay.page_w - lay.outer) * MM + 1
                assert y0 >= lay.top * MM - 1 and y1 <= (lay.page_h - lay.bottom) * MM + 1
                if lay is SIDE_BY_SIDE:                                                  # first artwork sits in the RIGHT slot
                    assert x0 >= (lay.inner + lay.slot_w + lay.gap) * MM - 1


def test_pdf_refuses_cells_below_minimum():
    with tempfile.TemporaryDirectory() as d:
        try: build_pdf(Path(d) / "x.pdf", SIDE_BY_SIDE, [{"name": "x", "labels": _labels(60, 40)}], PAL)
        except ValueError as e: assert "below 3.0 mm" in str(e); return
    raise AssertionError("expected ValueError")


def test_pdf_numbers_are_printed_ids_not_indices():
    from pypdf import PdfReader
    with tempfile.TemporaryDirectory() as d:
        lab = np.zeros((42, 28), np.int32)        # all index 0 -> printed ID "01"
        pdf = Path(d) / "i.pdf"; build_pdf(pdf, SIDE_BY_SIDE, [{"name": "x", "labels": lab}], PAL)
        txt = PdfReader(str(pdf)).pages[0].extract_text()
        assert txt.count("01") >= 28 * 42 and "00" not in txt.split()
