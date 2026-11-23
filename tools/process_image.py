"""Process one source image into a palette label matrix and review assets."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chromabook_engine import load_image, Palette  # noqa: E402
from chromabook_engine.optimize import optimize, OptimizerConfig, write_report  # noqa: E402
from chromabook_engine.layout import LayoutPrintSpec, SIDE_BY_SIDE  # noqa: E402
from chromabook_engine.render import render_preview, render_numbered  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out_dir")
    ap.add_argument("--preview-side", type=int, default=180)
    ap.add_argument("--final-side", type=int, default=480)
    ap.add_argument("--engines", default="BCD")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    image_path, out = Path(args.image), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    palette = Palette.from_file(ROOT / "data/main-21.palette.json")
    if len(palette) != 21:
        raise RuntimeError("The ChromaBook production palette must remain exactly 21 colours")
    if args.fast:
        opt = OptimizerConfig(preview_side=144, final_side=240, engines="D", keep_preprocess=1,
                              keep_combos=1, stability_top=1, cols_options=(24, 30), screen_cols=24)
    else:
        opt = OptimizerConfig(preview_side=args.preview_side, final_side=args.final_side,
                              engines=args.engines, keep_preprocess=1, keep_combos=2,
                              stability_top=2, cols_options=(24, 30, 36), screen_cols=24)
    spec = LayoutPrintSpec(layouts=(SIDE_BY_SIDE,), palette_size=len(palette))
    t = time.perf_counter()
    src = load_image(image_path)
    rep = optimize(src, palette, spec, opt)
    write_report(rep, out)
    np.save(out / "labels.npy", rep.labels)
    Image.fromarray(render_preview(rep.labels, palette, cell_px=12), "RGB").save(out / "color_preview.png")
    numbered, render_rep = render_numbered(rep.labels, palette, cell_px=36, line_width=1,
                                           number_scale=0.34, number_gray=10)
    Image.fromarray(numbered, "RGB").save(out / "numbered_preview.png")
    (out / "render_preflight.json").write_text(json.dumps(render_rep, indent=2), encoding="utf-8")
    summary = {"name": image_path.stem, "source": str(image_path),
               "grid": [int(rep.labels.shape[1]), int(rep.labels.shape[0])],
               "colors_used": rep.n_colors_used, "engine": rep.recommendation["engine"],
               "confidence": rep.confidence["level"], "elapsed_s": round(time.perf_counter()-t, 2),
               "timings": rep.timings, "recommendation": rep.recommendation}
    (out / "asset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done: {image_path.name} | grid={rep.labels.shape[1]}x{rep.labels.shape[0]} | "
          f"colors={rep.n_colors_used} | engine={rep.recommendation['engine']} | "
          f"confidence={rep.confidence['level']} | {summary['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
