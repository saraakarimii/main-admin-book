"""Acceptance Test Group A (pixel-art integrity) + B (mapping engines) on real executions."""
from helpers import *
import csv, hashlib
from chromabook_engine.pipeline import label_hash
from chromabook_engine.render import render_preview, render_numbered
from chromabook_engine.engines import COST_KINDS

SRC = synth_source()
CFG = small_cfg()
RES = Pipeline().run(SRC, PAL, CFG)


def test_every_cell_has_exactly_one_valid_palette_index():
    for k, eo in RES.engines.items():
        lab = eo.result.labels
        assert lab.shape == RES.geometry.shape
        assert lab.min() >= 0 and lab.max() < len(PAL), k
        assert np.isfinite(eo.result.selected_cost).all()


def test_preview_cells_are_uniform_and_match_palette():
    cp = 7
    for k, eo in RES.engines.items():
        lab = eo.result.labels
        prev = render_preview(lab, PAL, cp)
        assert prev.shape == (lab.shape[0] * cp, lab.shape[1] * cp, 3)
        for r in range(lab.shape[0]):
            for c in range(lab.shape[1]):
                block = prev[r*cp:(r+1)*cp, c*cp:(c+1)*cp].reshape(-1, 3)
                assert (block == PAL.rgb[lab[r, c]]).all(), (k, r, c)


def test_numbered_grid_uses_identical_labels_and_printed_ids():
    for k, eo in RES.engines.items():
        out = Pipeline.render(RES, k)
        texts = out["numbered_report"]["cell_texts"]
        lab = eo.result.labels
        for r in range(lab.shape[0]):
            for c in range(lab.shape[1]):
                assert texts[r][c] == PAL.ids[lab[r, c]]            # printed ID from palette, not index
        # preview and numbered grid come from the very same matrix
        assert out["color_preview"].shape[:2] == (lab.shape[0] * CFG.render.preview_cell_px,
                                                  lab.shape[1] * CFG.render.preview_cell_px)


def test_no_id_guessed_from_index():
    # a palette whose IDs are deliberately NOT index+1 must still print its own IDs
    rev = Palette.from_payload({"palette_id": "rev", "version": "1", "colors": [
        {"id": f"Z{99 - i}", "name": c.name, "hex": c.hex} for i, c in enumerate(PAL.colors)]})
    res = Pipeline().run(SRC, rev, CFG, engines=("B",))
    texts = Pipeline.render(res, "B")["numbered_report"]["cell_texts"]
    lab = res.engines["B"].result.labels
    assert texts[0][0] == f"Z{99 - int(lab[0, 0])}"


def test_numbers_centered_and_line_thickness_uniform():
    lab = np.full((4, 5), 7, np.int32)          # all ID "08"
    for cp, lw in ((44, 2), (60, 3)):
        img, rep = render_numbered(lab, PAL, cp, lw)
        assert rep["all_numbers_fit"]
        dark = img.sum(2) < 600
        # grid lines: every boundary exactly lw px thick, outer border included
        row = dark[lw + 1]                               # just below the top border, clear of text
        runs, n = [], 0
        for v in row:
            if v: n += 1
            elif n: runs.append(n); n = 0
        if n: runs.append(n)
        assert runs == [lw] * (lab.shape[1] + 1), runs
        # text bbox centred in interior of cell (1,1)
        x0, y0 = cp + lw, cp + lw
        cell = dark[y0:y0 + cp - lw, x0:x0 + cp - lw]
        ys, xs = np.where(cell)
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        assert abs(cx - (cp - lw - 1) / 2) <= 1.6 and abs(cy - (cp - lw - 1) / 2) <= 1.6, (cx, cy)


def test_cells_are_square_and_aspect_preserved():
    src = normalize_image(png_bytes(synth_rgb(h=1104, w=736, noise=3)), "t.png")
    for cols in (20, 33, 40, 57):
        cfg = small_cfg(mapping={"grid_cols": cols}, framing={"working_max_side": 0})
        prep = Pipeline().prepare(src, PAL, cfg)
        h, w = prep["frame"].rgb.shape[:2]
        rows = prep["frame"].rows
        assert abs((w / cols) - (h / rows)) / (w / cols) < 0.01, (cols, w, h, rows)
        assert abs((cols / rows) - (w / h)) / (w / h) < 1.0 / rows + 1e-9


def test_original_is_never_modified():
    data = png_bytes(synth_rgb())
    src = normalize_image(data, "a.png")
    before = hashlib.sha256(src.rgb.tobytes()).hexdigest()
    Pipeline().run(src, PAL, CFG, engines=("A", "B", "C", "D"))
    assert hashlib.sha256(src.rgb.tobytes()).hexdigest() == before
    assert src.sha256 == hashlib.sha256(data).hexdigest()
    try:
        src.rgb[0, 0, 0] = 1
    except ValueError:
        return
    raise AssertionError("normalised original must be read-only")


def test_reproducible_with_same_configuration():
    a = Pipeline().run(SRC, PAL, CFG)
    b = Pipeline().run(SRC, PAL, CFG)
    for k in "ABCD":
        assert a.engines[k].label_hash == b.engines[k].label_hash, k


def test_engines_A_to_D_run_and_differ_and_record_cost_kind():
    hashes = {k: eo.label_hash for k, eo in RES.engines.items()}
    assert len(set(hashes.values())) >= 2
    assert RES.engines["A"].result.cost_kind != RES.engines["B"].result.cost_kind
    for k, eo in RES.engines.items():
        assert eo.result.cost_kind == COST_KINDS[k]
        for m in ("DeltaE00_mean_vs_source", "DeltaE00_median_vs_source", "DeltaE00_p95_vs_source", "SSIM_vs_source",
                  "Edge_F1_1px", "singleton_ratio_pct", "transition_rate_pct", "colors_used"):
            assert m in eo.metrics and np.isfinite(eo.metrics[m]), (k, m)
    # regularised engines must not be rougher than the unregularised per-cell choice
    assert RES.engines["C"].metrics["transition_rate_pct"] <= RES.engines["B"].metrics["transition_rate_pct"] + 1e-9


def test_stability_metric_in_range():
    s = Pipeline().stability(RES, "C")
    assert 0 <= s["mean_stability_pct"] <= 100 and s["agreement_L_minus1_pct"] > 50
