"""Run the exact Advanced Mapping Lab v2 algorithm on one original image."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chromabook_engine.advanced_mapping import run_image, validate_params, DEFAULT_PARAMS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out_dir")
    ap.add_argument("--palette", default=str(ROOT / "data/main-21.palette.json"))
    ap.add_argument("--config", help="JSON containing params and selected_engine")
    ap.add_argument("--engine", default="D", choices=["A","B","C","D"])
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    params = validate_params(config.get("params", DEFAULT_PARAMS))
    engine = config.get("selected_engine", args.engine)
    result = run_image(args.image, args.palette, params, engine, Path(args.out_dir),
                       include_stability=bool(config.get("run_stability", False)))
    print(json.dumps({
        "selected_engine": engine,
        "total_runtime_seconds": result["total_runtime_seconds"],
        "grid": list(result["engines"][engine]["labels"].shape[::-1]),
        "metrics": result["engines"][engine]["metrics"],
        "all_engine_metrics": {k:v["metrics"] for k,v in result["engines"].items()},
    }, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
