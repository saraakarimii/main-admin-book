"""Build the three ChromaBook sample collections from inputs/collections.json.

Run from the repository root:
    python tools/build_collections.py
Each image is processed in its own Python process. The parent intentionally delays importing
NumPy/OpenCV until all workers have exited, avoiding native thread-pool fork hazards.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description="Build ChromaBook collection PDFs")
    ap.add_argument("--preview-side", type=int, default=180)
    ap.add_argument("--final-side", type=int, default=480)
    ap.add_argument("--engines", default="BCD", help="Mapping engines to compare")
    ap.add_argument("--fast", action="store_true", help="Use a smaller search for a quick smoke build")
    ap.add_argument("--only", default=None, help="Build only one collection id")
    ap.add_argument("--resume", action="store_true", help="Reuse existing per-image labels if present")
    args = ap.parse_args()

    manifest = json.loads((ROOT / "inputs/collections.json").read_text(encoding="utf-8"))
    out_root = ROOT / "outputs"
    out_root.mkdir(exist_ok=True)
    processed = []
    t_all = time.perf_counter()

    # Pass 1: run image workers without importing NumPy in this parent process.
    for collection in manifest["collections"]:
        if args.only and collection["id"] != args.only:
            continue
        t_collection = time.perf_counter()
        c_out = out_root / collection["id"]
        c_out.mkdir(parents=True, exist_ok=True)
        image_rows = []
        print(f"\n=== {collection['title']} ({len(collection['images'])} images) ===", flush=True)
        for i, item in enumerate(collection["images"], start=1):
            src_path = ROOT / item["path"]
            img_out = c_out / "images" / f"{i:02d}_{src_path.stem}"
            img_out.mkdir(parents=True, exist_ok=True)
            summary_path, labels_path = img_out / "asset_summary.json", img_out / "labels.npy"
            if not (args.resume and summary_path.exists() and labels_path.exists()):
                cmd = [sys.executable, str(ROOT / "tools/process_image.py"), str(src_path), str(img_out),
                       "--preview-side", str(args.preview_side), "--final-side", str(args.final_side),
                       "--engines", args.engines]
                if args.fast: cmd.append("--fast")
                print(f"[{i}/{len(collection['images'])}] Processing {item['name']} in isolated worker ...", flush=True)
                subprocess.run(cmd, cwd=ROOT, check=True,
                               env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
            else:
                print(f"[{i}/{len(collection['images'])}] Reusing {item['name']} ...", flush=True)
            asset = json.loads(summary_path.read_text(encoding="utf-8"))
            image_rows.append({"name": item["name"], "source": item["path"],
                               "labels_path": str(labels_path), "grid": asset["grid"],
                               "colors_used": asset["colors_used"], "engine": asset["engine"],
                               "confidence": asset["confidence"], "elapsed_s": asset["elapsed_s"],
                               "warnings": item.get("warnings", [])})
            print(f"    grid={asset['grid'][0]}x{asset['grid'][1]} colors={asset['colors_used']} "
                  f"engine={asset['engine']} confidence={asset['confidence']}", flush=True)
        processed.append({"collection": collection, "out_dir": str(c_out), "images": image_rows,
                          "elapsed_s": round(time.perf_counter() - t_collection, 2)})

    # Pass 2: now all workers are finished; load the numerical stack and assemble PDFs.
    import numpy as np
    sys.path.insert(0, str(ROOT / "src"))
    from chromabook_engine import Palette
    from chromabook_engine.layout import SIDE_BY_SIDE, assemble_book

    palette = Palette.from_file(ROOT / manifest["palette"])
    if len(palette) != 21:
        raise RuntimeError(f"Expected the frozen 21-colour palette; found {len(palette)}")
    summary = {"version": "0.4.0", "palette": palette.snapshot(), "collections": [],
               "optimizer": {"preview_side": 144 if args.fast else args.preview_side,
                             "final_side": 240 if args.fast else args.final_side,
                             "engines": "D" if args.fast else args.engines, "fast": args.fast}}
    for item in processed:
        collection, c_out = item["collection"], Path(item["out_dir"])
        artworks = []
        for image in item["images"]:
            labels = np.load(image["labels_path"])
            artworks.append({"name": image["name"], "labels": labels,
                             "source_path": str(ROOT / image["source"])})
        warnings = [w for x in collection["images"] for w in x.get("warnings", [])]
        result = assemble_book(c_out, artworks, palette, collection["title"], collection["subtitle"],
                               collection["id"], ROOT / collection["cover"],
                               theme_color=tuple(collection["theme_color"]), layout=SIDE_BY_SIDE,
                               language="English", min_cell_mm=3.0, max_id_chars=3, warnings=warnings)
        collection_summary = {
            "id": collection["id"], "title": collection["title"], "subtitle": collection["subtitle"],
            "image_count": len(artworks), "images": [{k: v for k, v in row.items() if k != "labels_path"}
                                                        for row in item["images"]],
            "numbered_pdf": str(result["numbered_pdf"].relative_to(ROOT)),
            "answer_key_pdf": str(result["answer_key_pdf"].relative_to(ROOT)),
            "book_report": str((c_out / "book_report.json").relative_to(ROOT)),
            "preflight_passed": result["preflight"]["passed"],
            "content_pages": result["preflight"]["pages"],
            "blank_slots": result["preflight"]["blank_slots"],
            "elapsed_s": item["elapsed_s"],
        }
        summary["collections"].append(collection_summary)
        print(f"Built {collection['title']}: {collection_summary['content_pages']} content pages + cover; "
              f"preflight={'PASS' if collection_summary['preflight_passed'] else 'FAIL'}; "
              f"image-processing time={item['elapsed_s']}s", flush=True)

    summary["elapsed_s"] = round(time.perf_counter() - t_all, 2)
    (out_root / "build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAll selected collections built in {summary['elapsed_s']}s", flush=True)
    print(f"Summary: {out_root / 'build_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
