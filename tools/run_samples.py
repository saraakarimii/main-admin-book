"""Run the engine on sample images with the main-21 palette; write examples + summary."""
import json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chromabook_engine import Pipeline, Palette, PipelineConfig, load_image
from chromabook_engine.config import FramingConfig, MappingConfig
from chromabook_engine.export import export_all

PAL = Palette.from_file(Path(__file__).resolve().parents[1] / "data" / "main-21.palette.json")
UP = Path("/mnt/user-data/uploads")
SAMPLES = {  # name -> (file, framing override)
    "tulips":  ("photo_5773998151659360459_y.jpg", {}),
    "unicorn": ("photo_5769491962166712591_y.jpg", {}),
    "cat":     ("photo_5769491962166712592_y.jpg", {"crop_enabled": True, "crop": (0.03, 0.02, 0.97, 0.90)}),
    "mermaid": ("photo_5769491962166712593_y.jpg", {}),
}
out_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("examples")
summary = {}
for name, (fn, fr) in SAMPLES.items():
    src = load_image(UP / fn)
    cfg = PipelineConfig(framing=FramingConfig(**fr), mapping=MappingConfig(grid_cols=36))
    pipe = Pipeline()
    t0 = time.time(); res = pipe.run(src, PAL, cfg); dt = time.time() - t0
    export_all(res, out_root / name, make_zip=False)
    row = {"grid": f"{res.geometry.cols}x{res.geometry.rows}", "seconds_total": round(dt, 1), "engines": {}}
    for k, eo in res.engines.items():
        m = eo.metrics
        st = pipe.stability(res, k)
        row["engines"][k] = {"dE00_mean": round(m["DeltaE00_mean_vs_source"], 2), "dE00_p95": round(m["DeltaE00_p95_vs_source"], 2),
                             "SSIM": round(m["SSIM_vs_source"], 3), "EdgeF1": round(m["Edge_F1_1px"], 3),
                             "singleton_pct": round(m["singleton_ratio_pct"], 1), "transition_pct": round(m["transition_rate_pct"], 1),
                             "colors_used": m["colors_used"], "stability_pct": round(st["mean_stability_pct"], 1),
                             "engine_s": round(m["runtime_seconds"], 2)}
    # resolution consequence check: 900px working image vs full resolution, same engine B
    cfg_full = replace(cfg, framing=replace(cfg.framing, working_max_side=0))
    r_full = Pipeline().run(src, PAL, cfg_full, engines=("B",))
    if r_full.engines["B"].result.labels.shape == res.engines["B"].result.labels.shape:
        row["label_agreement_900px_vs_full_res_B_pct"] = round(100 * float(np.mean(
            r_full.engines["B"].result.labels == res.engines["B"].result.labels)), 1)
    row["original_px"] = src.meta["original_size"]
    summary[name] = row
    print(name, json.dumps(row))
(out_root / "summary.json").write_text(json.dumps(summary, indent=2))
