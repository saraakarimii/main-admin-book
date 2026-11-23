# Phase 5.1 — Interactive Hybrid Image Optimization

This implementation is based on the actual controls and algorithms in `ChromaBook_Advanced_Mapping_Lab_v2.ipynb`.

## Workflow

1. Upload an image.
2. Run Automatic using the Notebook v2 default parameter set.
3. Compare the four independent engines A/B/C/D using their previews and metrics.
4. Select the asset in Hybrid Mapping Controls.
5. Change any Notebook v2 parameter and reprocess. The original uploaded file is always used as the input; processed previews are never fed back as source.
6. Each processing run stores its own parameter JSON, selected engine, all four engine previews, all four metric sets and output labels.
7. Approve the desired result and use that run's selected-engine labels in book assembly.

## Parameter contract

The API `GET /api/mapping/defaults` returns the exact defaults and widget bounds for the Notebook v2 controls. `POST /api/assets/{asset_id}/process` accepts:

```json
{
  "params": {
    "max_side": 900,
    "crop_enabled": false,
    "crop_x0": 0,
    "crop_y0": 0,
    "crop_x1": 100,
    "crop_y1": 100,
    "illumination_strength": 0.25,
    "illumination_sigma": 0.1,
    "clahe_clip": 1.5,
    "clahe_tiles": 8,
    "denoise_method": "bilateral",
    "denoise_strength": 18,
    "sharpen_fine": 0.15,
    "sharpen_coarse": 0.05,
    "auto_canny": true,
    "canny_low": 55,
    "canny_high": 145,
    "edge_close": 1,
    "grid_cols": 40,
    "max_samples": 96,
    "trim_fraction": 0.03,
    "smoothness_lambda": 1.8,
    "boundary_beta": 4.0,
    "icm_iterations": 5,
    "slic_segments": 320,
    "slic_compactness": 10,
    "region_balance": 0.25,
    "numbered_cell_px": 44,
    "numbered_line_width": 2
  },
  "selected_engine": "D",
  "run_stability": false
}
```

All keys above are the Notebook's `get_params()` keys. No extra image-processing sliders are introduced. The `run_stability` flag corresponds to the Notebook's separate optional “Stability Test ±1 L*” action, not a new mapping parameter.

## Notebook algorithms retained

- Alpha compositing onto white and EXIF orientation correction.
- Crop/resize and illumination correction.
- CLAHE on luminance, selectable bilateral/NLM/median denoising and multiscale sharpening.
- Scharr gradient plus Canny edge extraction and morphological closing.
- Grid-cell CIEDE2000 unary costs with deterministic pixel sampling and trimmed mean.
- Engine A: legacy mean-Lab / ΔE76.
- Engine B: cell-wise CIEDE2000.
- Engine C: boundary-aware ICM.
- Engine D: SLIC region-balanced unary costs plus boundary-aware ICM.
- Pixel preview and numbered grid rendering.
- Metrics: ΔE00 mean/median/p95, SSIM, Edge F1/precision/recall, colors used, singleton ratio, tiny-component ratio and transition rate.
- Optional ±1 L* stability test for B/C/D.

## Palette lock

The Notebook's sample palette editor is intentionally not exposed. The server always loads `data/main-21.palette.json`, checks for exactly 21 colours and retains its explicit printed IDs. No palette upload, palette editing or colour addition is possible from the Hybrid UI.

## API and persistence

- `GET /api/mapping/defaults`: parameter defaults, control ranges and engine list.
- `POST /api/assets/{id}/process`: automatic or manual processing request.
- `GET /api/assets`: latest run, complete run history, stored configs and metric summaries.
- `GET /api/assets/{id}/preview/engine_A/numbered_preview.png` (A/B/C/D): inspect a specific engine output.
- Every run has a separate output folder and a persisted `params_json`, so previous results are retained.

## Important boundary

This path uses the Notebook v2 mapping implementation. It is not silently blended with the earlier Phase 2 optimizer. The Phase 5 book assembly consumes the selected engine's `labels.npy` from the latest approved processing run.
