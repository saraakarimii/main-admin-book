"""Staged, pruned candidate search.  Every candidate is recorded (config, metrics, score, reason)."""
from __future__ import annotations
import hashlib, json, time
from dataclasses import dataclass, field, replace, asdict
import numpy as np
from ..config import PipelineConfig, FramingConfig, PreprocessConfig
from ..pipeline import Pipeline
from .autocrop import propose_framings
from .constraints import PrintSpec, rows_for, feasible, feasible_cols
from .diagnostics import diagnose
from .scoring import cell_structure, rank_group, confidence

NOTE = ("Metrics rank candidates against each other; they do not certify the printed result. "
        "Visual QA by the owner is still required.")


@dataclass(frozen=True)
class OptimizerConfig:
    preview_side: int = 320            # screening resolution
    final_side: int = 1000             # finalists are re-run here (Phase 1 finding: 900px != full res)
    engines: str = "ABCD"
    keep_preprocess: int = 2
    keep_combos: int = 2
    stability_top: int = 3
    cols_options: tuple | None = None  # None -> derived from the print constraint
    screen_cols: int | None = None     # grid used while screening preprocess / engines
    knee_dE: float = 0.35              # stop refining when the next grid step gains less than this dE00
    seed: int = 42


@dataclass
class OptimizationReport:
    source_sha256: str
    palette: dict
    print_spec: dict
    diagnostics: dict
    framings: list
    chosen_framing: str
    stages: dict
    candidates: list
    ranking: list
    recommendation: dict
    alternatives: list
    framing_alternatives: list
    resolution_curve: list
    confidence: dict
    timings: dict
    note: str = NOTE

    def to_dict(self): return asdict(self)


def _id(*p): return hashlib.sha256(json.dumps(p, sort_keys=True, default=str).encode()).hexdigest()[:10]


def preprocess_candidates(diag: dict, base: PreprocessConfig) -> list[tuple[str, PreprocessConfig]]:
    """Diagnostic-driven, bounded set (<=6). Names explain why each exists."""
    c = [("default", base), ("minimal", replace(base, illumination_strength=0.0, clahe_clip=0.0, denoise_strength=0.0,
                                               sharpen_fine=0.0, sharpen_coarse=0.0))]
    f = set(diag["flags"])
    if "low_contrast" in f: c.append(("contrast_up", replace(base, contrast=1.25, clahe_clip=2.2)))
    if "dark" in f: c.append(("brighten", replace(base, brightness=8.0)))
    if "bright" in f: c.append(("darken", replace(base, brightness=-6.0)))
    if "noisy" in f: c.append(("denoise_strong", replace(base, denoise_method="nlm", denoise_strength=10.0)))
    if "soft_or_blurry" in f: c.append(("sharpen_up", replace(base, sharpen_fine=0.35, sharpen_coarse=0.10)))
    if len(c) < 3: c.append(("gentle", replace(base, illumination_strength=0.12, clahe_clip=0.8, denoise_strength=10.0)))
    return c[:6]


def _run(pipe, src, pal, cfg, engines, cs):
    res = pipe.run(src, pal, cfg, engines=tuple(engines))
    rows = []
    for k in engines:
        eo = res.engines[k]; m = dict(eo.metrics)
        m.update(cell_structure(res.stages["reference"], eo.result.labels, res.geometry, pal.lab))
        rows.append({"engine": k, "metrics": m, "label_hash": eo.label_hash, "cell_mm": cs["cell_mm"],
                     "grid": [res.geometry.cols, res.geometry.rows], "_res": res})
    return rows


def optimize(source, palette, print_spec: PrintSpec | None = None, opt: OptimizerConfig | None = None,
             base: PipelineConfig | None = None, pipeline: Pipeline | None = None) -> OptimizationReport:
    t_all = time.time(); T = {}
    spec, opt, base = print_spec or PrintSpec(), opt or OptimizerConfig(), base or PipelineConfig()
    pipe = pipeline or Pipeline()
    cands, stages = [], {}

    # ---- stage 0: diagnostics + framing proposals --------------------------------------------
    t0 = time.time()
    diag = diagnose(source.rgb, palette)
    framings = propose_framings(source.rgb)
    chosen = next(f for f in framings if f["default"])
    T["diagnostics_and_framing"] = time.time() - t0
    aspect = (chosen["crop"][3] - chosen["crop"][1]) * source.rgb.shape[0] / (
        (chosen["crop"][2] - chosen["crop"][0]) * source.rgb.shape[1])
    cols_all = list(opt.cols_options) if opt.cols_options else feasible_cols(spec, aspect)
    cols_all = [c for c in cols_all if feasible(spec, c, rows_for(c, aspect))["meets_min_cell"]]  # HARD filter
    if not cols_all:
        raise ValueError("no grid size satisfies the minimum cell size for this panel; relax the PrintSpec")
    # every cell needs >= 4 working pixels, otherwise cell statistics are meaningless
    need = int(np.ceil(4 * max(cols_all) * max(1.0, aspect)))
    if opt.preview_side < need:
        opt = replace(opt, preview_side=need)
    if opt.final_side < opt.preview_side:
        opt = replace(opt, final_side=opt.preview_side)
    screen_cols = opt.screen_cols or cols_all[len(cols_all) // 2]
    stages["constraint"] = {"allowed_cols": cols_all, "screen_cols": screen_cols, "aspect_h_over_w": round(aspect, 4),
                            "rejected_cols_below_min_cell": [c for c in range(16, 121, 6) if c not in cols_all]}

    def frame_cfg(fr, side):
        return replace(base.framing, crop_enabled=fr["id"] != "full", crop=tuple(fr["crop"]), working_max_side=side)

    def cfg_for(fr, pre, cols, side):
        return replace(base, framing=frame_cfg(fr, side), preprocess=pre, mapping=replace(base.mapping, grid_cols=cols), seed=opt.seed)

    def record(stage, name, rows, cfg):
        for r in rows:
            r["stage"], r["name"], r["config"] = stage, name, cfg.to_dict()
            r["id"] = _id(stage, name, r["engine"], r["grid"], cfg.to_dict()["framing"])
            cands.append(r)

    def cs_of(cols): return feasible(spec, cols, rows_for(cols, aspect))

    # ---- stage 1: preprocessing screen (engine B, preview res, fixed grid) ---------------------
    t0 = time.time(); pre_rows = []
    for name, pre in preprocess_candidates(diag, base.preprocess):
        cfg = cfg_for(chosen, pre, screen_cols, opt.preview_side)
        rows = _run(pipe, source, palette, cfg, "B", cs_of(screen_cols))
        for r in rows: r["pre_name"], r["pre"] = name, pre
        record("1_preprocess", name, rows, cfg); pre_rows += rows
    ranked_pre = rank_group(list(pre_rows))
    keep_pre = ranked_pre[:opt.keep_preprocess]
    stages["preprocess"] = {"tried": [r["pre_name"] for r in pre_rows], "kept": [r["pre_name"] for r in keep_pre],
                            "scores": {r["pre_name"]: r["score"] for r in ranked_pre}}
    T["preprocess_screen"] = time.time() - t0

    # ---- stage 2: engines for the kept preprocessing (preview res) ------------------------------
    t0 = time.time(); eng_rows = []
    for kp in keep_pre:
        cfg = cfg_for(chosen, kp["pre"], screen_cols, opt.preview_side)
        rows = _run(pipe, source, palette, cfg, opt.engines, cs_of(screen_cols))
        for r in rows: r["pre_name"], r["pre"] = kp["pre_name"], kp["pre"]
        record("2_engines", kp["pre_name"], rows, cfg); eng_rows += rows
    ranked_eng = rank_group(list(eng_rows))
    combos = []
    for r in ranked_eng:
        if (r["pre_name"], r["engine"]) not in [(c["pre_name"], c["engine"]) for c in combos]: combos.append(r)
        if len(combos) >= opt.keep_combos: break
    stages["engines"] = {"kept": [(c["pre_name"], c["engine"]) for c in combos],
                         "scores": {f"{r['pre_name']}/{r['engine']}": r["score"] for r in ranked_eng}}
    T["engine_screen"] = time.time() - t0

    # ---- stage 3: resolution sweep with the best combo; knee of the dE00-vs-cols curve ----------
    t0 = time.time(); curve = []
    best = combos[0]
    for cols in cols_all:
        cfg = cfg_for(chosen, best["pre"], cols, opt.preview_side)
        r = _run(pipe, source, palette, cfg, best["engine"], cs_of(cols))[0]
        r["pre_name"], r["pre"] = best["pre_name"], best["pre"]; record("3_resolution", best["pre_name"], [r], cfg)
        curve.append({"cols": cols, "rows": r["grid"][1], "cell_mm": cs_of(cols)["cell_mm"],
                      "dE00_mean": round(r["metrics"]["DeltaE00_mean_vs_source"], 3),
                      "singleton_pct": round(r["metrics"]["singleton_ratio_pct"], 2)})
    chosen_cols = curve[-1]["cols"]
    for i in range(len(curve) - 1):
        if curve[i]["dE00_mean"] - curve[i + 1]["dE00_mean"] < opt.knee_dE:
            chosen_cols = curve[i]["cols"]; break
    stages["resolution"] = {"rule": f"smallest grid whose next step gains < {opt.knee_dE} dE00 (diminishing returns; larger cells are easier to colour)",
                            "chosen_cols": chosen_cols}
    T["resolution_sweep"] = time.time() - t0

    # ---- stage 4: finalists at final resolution, same grid, all comparable -----------------------
    t0 = time.time(); fin_rows = []
    for cb in combos:
        cfg = cfg_for(chosen, cb["pre"], chosen_cols, opt.final_side)
        r = _run(pipe, source, palette, cfg, cb["engine"], cs_of(chosen_cols))[0]
        r["pre_name"], r["pre"] = cb["pre_name"], cb["pre"]; r["cfg"] = cfg
        record("4_final", cb["pre_name"], [r], cfg); fin_rows.append(r)
    # also include the other engines for the best preprocessing so the owner sees the real alternative
    ranked_fin = rank_group(list(fin_rows))
    for r in ranked_fin[:opt.stability_top]:
        r["stability"] = pipe.stability(r["_res"], r["engine"])
    # stability is informational + feeds confidence; it does not change the ordering of metric scores
    top = ranked_fin[0]
    conf = confidence(ranked_fin, top.get("stability", {}).get("mean_stability_pct"))
    T["finalists"] = time.time() - t0

    # ---- framing alternatives: winning recipe applied to the other framings (NOT ranked) --------
    alts_fr = []
    for fr in framings:
        if fr["id"] == chosen["id"]: continue
        a_aspect = (fr["crop"][3] - fr["crop"][1]) * source.rgb.shape[0] / ((fr["crop"][2] - fr["crop"][0]) * source.rgb.shape[1])
        rr = rows_for(chosen_cols, a_aspect)
        f = feasible(spec, chosen_cols, rr)
        entry = {"framing": fr["id"], "crop": fr["crop"], "reason": fr["reason"], "grid": [chosen_cols, rr],
                 "cell_mm": f["cell_mm"], "meets_min_cell": f["meets_min_cell"],
                 "note": "different content: metrics are not comparable with the recommended framing"}
        if f["meets_min_cell"]:
            cfg = cfg_for(fr, top["pre"], chosen_cols, opt.final_side)
            r = pipe.run(source, palette, cfg, engines=(top["engine"],))
            entry["label_hash"] = r.engines[top["engine"]].label_hash
            entry["dE00_mean"] = round(r.engines[top["engine"]].metrics["DeltaE00_mean_vs_source"], 3)
        alts_fr.append(entry)

    def slim(r):
        return {"id": r["id"], "stage": r["stage"], "name": r["name"], "engine": r["engine"], "grid": r["grid"],
                "cell_mm": r["cell_mm"], "label_hash": r["label_hash"], "score": r.get("score"),
                "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r["metrics"].items()},
                "stability": r.get("stability"), "config": r["config"]}

    top_labels = top["_res"].engines[top["engine"]].result.labels.copy()
    ranking = [slim(r) for r in ranked_fin]
    for c in cands: c.pop("_res", None)
    recommendation = {**ranking[0], "confidence": conf, "framing": chosen["id"], "preprocess": top["pre_name"],
                      "why": [f"hard constraint: cell {cs_of(chosen_cols)['cell_mm']} mm >= {spec.min_cell_mm} mm",
                              f"grid: {stages['resolution']['rule']}",
                              f"best of {len(ranked_fin)} finalists on normalised metrics"] + conf["reasons"]}
    T["total"] = time.time() - t_all
    for r in fin_rows: r.pop("_res", None); r.pop("cfg", None)
    out_cands = [slim(c) for c in cands]
    rep = OptimizationReport(source.sha256, palette.snapshot(), asdict(spec), diag, framings, chosen["id"], stages,
                              out_cands, ranking, recommendation, ranking[1:], alts_fr, curve, conf, {k: round(v, 2) for k, v in T.items()})
    rep.labels = top_labels                      # not serialised; used by the layout report and PDF writer
    rep.n_colors_used = int(len(np.unique(top_labels)))
    return rep


def write_report(report: OptimizationReport, out_dir) -> dict:
    """optimization_report.json (full, machine-readable) + optimization_report.md (owner-readable)."""
    from pathlib import Path
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    d = report.to_dict()
    (out / "optimization_report.json").write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")
    rec = d["recommendation"]; m = rec["metrics"]
    lines = [f"# Recommendation (confidence: {rec['confidence']['level']}, {rec['confidence']['value']}/100)", "",
             f"- Framing: **{rec['framing']}**   Preprocess: **{rec['preprocess']}**   Engine: **{rec['engine']}**",
             f"- Grid: **{rec['grid'][0]} x {rec['grid'][1]}**, cell **{rec['cell_mm']} mm**",
             f"- dE00 mean {m['DeltaE00_mean_vs_source']}, p95 {m['DeltaE00_p95_vs_source']}, singletons {m['singleton_ratio_pct']} %, "
             f"cell-structure F1 {m['cell_structure_f1']}", "", "## Why", *[f"- {w}" for w in rec["why"]], "",
             "## Resolution curve (best recipe)", "| cols | rows | cell mm | dE00 mean | singleton % |", "|---|---|---|---|---|",
             *[f"| {c['cols']} | {c['rows']} | {c['cell_mm']} | {c['dE00_mean']} | {c['singleton_pct']} |" for c in d["resolution_curve"]], "",
             "## Framing proposals", *[f"- `{f['id']}` {f['crop']} - {f['reason']}{' (default)' if f['default'] else ''}" for f in d["framings"]], "",
             "## Diagnostics flags", f"- {', '.join(d['diagnostics']['flags']) or 'none'}", "", f"_{d['note']}_"]
    (out / "optimization_report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(out / "optimization_report.json"), "md": str(out / "optimization_report.md")}
