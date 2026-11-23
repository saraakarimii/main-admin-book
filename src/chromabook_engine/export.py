"""Asset package export (per image / per engine) following the product spec."""
from __future__ import annotations
import csv, json, zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from .metrics import delta_map_rgb, palette_usage
from .pipeline import Pipeline


def _save(path: Path, arr):
    Image.fromarray(np.asarray(arr).astype(np.uint8)).save(path)


def export_asset(result, key: str, out_dir, render_cfg=None) -> Path:
    """Write one engine's asset package. Everything derives from the same label matrix."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    eo = result.engines[key]
    labels = eo.result.labels
    rows, cols = labels.shape
    rendered = Pipeline.render(result, key, render_cfg)
    pal = result.palette

    _save(out / "original.png", result.source.rgb)
    _save(out / "cropped_source.png", result.stages["reference"])
    _save(out / "preprocessed.png", result.stages["processed"])
    _save(out / "color_preview.png", rendered["color_preview"])
    _save(out / "numbered_outline.png", rendered["numbered_outline"])
    _save(out / "edge_map.png", (result.edge_map * 255).astype(np.uint8))
    _save(out / "slic_boundaries.png", (result.slic_boundary * 255).astype(np.uint8))
    _save(out / "delta_e_map.png", delta_map_rgb(eo.delta_map))
    labels.astype(np.uint8).tofile(out / "grid_indices.bin")                    # uint8 row-major INDICES
    np.save(out / "grid_indices.npy", labels.astype(np.uint8))
    eo.result.selected_cost.astype("<f4").tofile(out / "cell_delta_e.bin")      # float32 little-endian
    with open(out / "grid_color_ids.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        for r in range(rows):
            w.writerow([pal.id_of(labels[r, c]) for c in range(cols)])
    (out / "palette.json").write_text(json.dumps(pal.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(
        {"engine": key, **result.config.to_dict()}, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(
        {"engine": key, **eo.metrics, "palette_usage": palette_usage(labels, pal)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "source": {"filename": result.source.filename, "sha256": result.source.sha256, **result.source.meta},
        "engine": key, "engine_name": eo.result.name,
        "rows": rows, "cols": cols, "cell_count": rows * cols,
        "storage": {"grid_indices.bin": "uint8, row-major PALETTE INDICES (not printed IDs)",
                    "cell_delta_e.bin": "float32 little-endian, selected-label cost per cell, row-major"},
        "cell_cost_kind": eo.result.cost_kind,
        "palette": {"palette_id": pal.palette_id, "version": pal.version, "sha256": pal.sha256,
                    "index_to_printed_id": pal.ids},
        "label_sha256": eo.label_hash,
        "framing": {"bbox_original_px": list(result.frame.bbox_original_px),
                    "bbox_original_norm": list(result.frame.bbox_original_norm),
                    **result.frame.steps},
        "canny_thresholds_used": list(result.canny_thresholds),
        "numbered_render": {k: v for k, v in rendered["numbered_report"].items() if k != "cell_texts"},
        "stage_keys": result.keys, "seed": result.config.seed,
        "timings_seconds": result.timings, "versions": result.versions,
    }
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def export_all(result, out_dir, make_zip=True):
    root = Path(out_dir); root.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in result.engines:
        export_asset(result, key, root / f"Engine_{key}")
        rows.append({"engine": key, "name": result.engines[key].result.name,
                     "cell_cost_kind": result.engines[key].result.cost_kind,
                     **result.engines[key].metrics})
    with open(root / "engine_comparison_metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if make_zip:
        zp = root.with_suffix(".zip")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for fp in sorted(root.rglob("*")):
                if fp.is_file():
                    z.write(fp, fp.relative_to(root.parent))
        return zp
    return root
