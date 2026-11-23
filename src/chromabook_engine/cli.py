"""Command line: chromabook run IMAGE --palette palette.json --out DIR [--cols 40] [--engines ABCD]"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from . import Pipeline, Palette, PipelineConfig, load_image
from .config import MappingConfig, FramingConfig
from .export import export_all


def main(argv=None):
    ap = argparse.ArgumentParser(prog="chromabook")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process one image with the selected engines and export assets")
    r.add_argument("image"); r.add_argument("--palette", required=True); r.add_argument("--out", required=True)
    r.add_argument("--cols", type=int, default=40); r.add_argument("--engines", default="ABCD")
    r.add_argument("--config", help="JSON file with a full PipelineConfig dict (overrides --cols)")
    r.add_argument("--stability", action="store_true")
    o = sub.add_parser("optimize", help="Phase 2: diagnose, propose framings, search candidates, recommend")
    o.add_argument("image"); o.add_argument("--palette", required=True); o.add_argument("--out", required=True)
    o.add_argument("--panel-mm", nargs=2, type=float, default=[130.0, 180.0], metavar=("W", "H"))
    o.add_argument("--min-cell-mm", type=float, default=3.0)
    o.add_argument("--final-side", type=int, default=1000)
    a = ap.parse_args(argv)
    if a.cmd == "optimize":
        from .optimize import optimize, OptimizerConfig, PrintSpec, write_report
        pal = Palette.from_file(a.palette); src = load_image(a.image)
        rep = optimize(src, pal, PrintSpec(a.panel_mm[0], a.panel_mm[1], a.min_cell_mm), OptimizerConfig(final_side=a.final_side))
        paths = write_report(rep, Path(a.out) / Path(a.image).stem)
        r = rep.recommendation
        print(f"recommend: framing={r['framing']} pre={r['preprocess']} engine={r['engine']} grid={r['grid']} "
              f"cell={r['cell_mm']}mm confidence={r['confidence']['level']}({r['confidence']['value']})")
        print("report:", paths["md"]); return 0
    pal = Palette.from_file(a.palette)
    cfg = PipelineConfig.from_dict(json.loads(Path(a.config).read_text())) if a.config else \
        PipelineConfig(mapping=MappingConfig(grid_cols=a.cols))
    src = load_image(a.image)
    pipe = Pipeline()
    res = pipe.run(src, pal, cfg, engines=tuple(a.engines.upper()))
    z = export_all(res, Path(a.out) / Path(a.image).stem)
    for k, eo in res.engines.items():
        m = eo.metrics
        line = (f"{k}: dE00 mean {m['DeltaE00_mean_vs_source']:.2f} p95 {m['DeltaE00_p95_vs_source']:.2f} "
                f"SSIM {m['SSIM_vs_source']:.3f} singleton {m['singleton_ratio_pct']:.1f}% "
                f"trans {m['transition_rate_pct']:.1f}% colors {m['colors_used']}")
        if a.stability:
            line += f" stability {pipe.stability(res, k)['mean_stability_pct']:.1f}%"
        print(line)
    print("exported:", z)
    return 0


if __name__ == "__main__":
    sys.exit(main())
