import numpy as np
from helpers import PAL, synth_rgb, synth_source, png_bytes
from chromabook_engine import normalize_image
from chromabook_engine.optimize import optimize, OptimizerConfig, PrintSpec, diagnose, propose_framings
from chromabook_engine.optimize.constraints import feasible, feasible_cols
from chromabook_engine.optimize.scoring import rank_group, confidence, METRICS

FAST = OptimizerConfig(preview_side=160, final_side=240, keep_preprocess=1, keep_combos=2, stability_top=1)


def _bordered():
    inner = synth_rgb(200, 140)
    img = np.full((300, 220, 3), 255, np.uint8)
    img[50:250, 40:180] = inner
    return normalize_image(png_bytes(img), "bordered.png")


def test_constraint_is_a_hard_filter_in_every_candidate():
    rep = optimize(synth_source(h=300, w=200), PAL, PrintSpec(min_cell_mm=3.0), FAST)
    assert rep.candidates and all(c["cell_mm"] >= 3.0 - 1e-9 for c in rep.candidates)
    assert max(rep.stages["constraint"]["allowed_cols"]) <= 40        # 130 mm / 3 mm = 43 -> grid of 40 at aspect 1.5
    stricter = optimize(synth_source(h=300, w=200), PAL, PrintSpec(min_cell_mm=5.0), FAST)
    assert max(stricter.stages["constraint"]["allowed_cols"]) < max(rep.stages["constraint"]["allowed_cols"])
    assert all(c["cell_mm"] >= 5.0 - 1e-9 for c in stricter.candidates)


def test_infeasible_spec_is_refused_not_silently_relaxed():
    try:
        optimize(synth_source(h=300, w=200), PAL, PrintSpec(panel_width_mm=20, panel_height_mm=20, min_cell_mm=3.0), FAST)
    except ValueError as e:
        assert "minimum cell size" in str(e); return
    raise AssertionError("expected ValueError")


def test_feasibility_numbers():
    f = feasible(PrintSpec(), 40, 60)
    assert f["meets_min_cell"] and abs(f["cell_mm"] - 3.0) < 1e-6
    assert not feasible(PrintSpec(), 44, 66)["meets_min_cell"]
    assert 16 in feasible_cols(PrintSpec(), 1.5)


def test_diagnostics_flags():
    flat = np.clip(120 + (synth_rgb(200, 150).astype(float) - 128) * 0.15, 0, 255).astype(np.uint8)
    assert "low_contrast" in diagnose(flat, PAL)["flags"]
    assert "noisy" in diagnose(synth_rgb(200, 150, noise=40.0), PAL)["flags"]
    d = diagnose(synth_rgb(200, 150), PAL)
    assert d["size"] == [150, 200] and "palette_gap_p95_dE00" in d


def test_autocrop_trims_border_and_keeps_full_as_alternative():
    src = _bordered()
    fr = propose_framings(src.rgb)
    ids = [f["id"] for f in fr]
    assert "full" in ids and "trim_border" in ids
    trim = next(f for f in fr if f["id"] == "trim_border")
    assert trim["default"] and not next(f for f in fr if f["id"] == "full")["default"]
    x0, y0, x1, y1 = trim["crop"]
    # content occupies x 40..180 of 220, y 50..250 of 300; the crop must keep all of it
    assert x0 <= 40 / 220 + 0.02 and x1 >= 180 / 220 - 0.02 and y0 <= 50 / 300 + 0.02 and y1 >= 250 / 300 - 0.02
    assert x0 > 0.10 and y0 > 0.08                                     # and actually remove the margin


def test_no_border_means_full_is_default():
    fr = propose_framings(synth_source(h=240, w=160).rgb)
    assert next(f for f in fr if f["default"])["id"] == "full"


def test_original_unchanged_by_optimization():
    src = synth_source(h=240, w=160); before = src.rgb.copy()
    optimize(src, PAL, PrintSpec(), FAST)
    assert np.array_equal(src.rgb, before) and not src.rgb.flags.writeable


def test_ranking_ties_and_confidence_behaviour():
    mk = lambda de: {"metrics": {"DeltaE00_mean_vs_source": de, "DeltaE00_p95_vs_source": 30.0, "cell_structure_f1": 0.5,
                                 "singleton_ratio_pct": 5.0, "transition_gap_pp": 3.0}}
    tie = rank_group([mk(10.0), mk(10.05)])                 # 0.05 dE is below the 0.30 tie floor
    assert abs(tie[0]["score"] - tie[1]["score"]) < 2.0
    assert confidence(tie, 97.0)["level"] != "high"          # a near-tie can never be "high"
    clear = rank_group([mk(10.0), mk(13.0)])
    assert clear[0]["score"] - clear[1]["score"] > 10
    assert confidence([clear[0]], None)["level"] == "low"


def test_deterministic_recommendation_and_report_shape():
    src = synth_source(h=300, w=200)
    a = optimize(src, PAL, PrintSpec(), FAST); b = optimize(src, PAL, PrintSpec(), FAST)
    assert a.recommendation["label_hash"] == b.recommendation["label_hash"]
    assert a.recommendation["grid"] == b.recommendation["grid"]
    d = a.to_dict()
    for k in ("diagnostics", "framings", "stages", "candidates", "ranking", "recommendation", "resolution_curve",
              "confidence", "framing_alternatives", "note"):
        assert k in d
    assert "do not certify" in d["note"]
    assert a.recommendation["confidence"]["level"] in ("low", "medium", "high")


def test_framing_alternatives_are_marked_not_comparable():
    rep = optimize(_bordered(), PAL, PrintSpec(), FAST)
    assert rep.chosen_framing == "trim_border"
    alt = [a for a in rep.framing_alternatives if a["framing"] == "full"][0]
    assert "not comparable" in alt["note"]


def test_small_screening_resolution_is_raised_so_cells_are_never_empty():
    import warnings
    tiny = OptimizerConfig(preview_side=60, final_side=60, keep_preprocess=1, keep_combos=1, stability_top=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rep = optimize(synth_source(h=300, w=200), PAL, PrintSpec(), tiny)
    assert all(np.isfinite(v) for c in rep.candidates for v in c["metrics"].values() if isinstance(v, float))
