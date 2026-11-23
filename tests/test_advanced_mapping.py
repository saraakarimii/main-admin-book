from pathlib import Path
import json
import numpy as np
import pytest
from PIL import Image

from chromabook_engine.advanced_mapping import (
    DEFAULT_PARAMS, PARAMETER_UI, validate_params, run_image, ENGINE_NAMES
)

ROOT = Path(__file__).resolve().parents[1]
PALETTE = ROOT / "data" / "main-21.palette.json"

def test_defaults_match_notebook_v2_controls_and_locked_palette():
    assert DEFAULT_PARAMS["max_side"] == 900
    assert DEFAULT_PARAMS["illumination_strength"] == 0.25
    assert DEFAULT_PARAMS["clahe_clip"] == 1.5
    assert DEFAULT_PARAMS["denoise_method"] == "bilateral"
    assert DEFAULT_PARAMS["grid_cols"] == 40
    assert DEFAULT_PARAMS["max_samples"] == 96
    assert DEFAULT_PARAMS["smoothness_lambda"] == 1.8
    assert DEFAULT_PARAMS["boundary_beta"] == 4.0
    assert DEFAULT_PARAMS["slic_segments"] == 320
    assert DEFAULT_PARAMS["numbered_cell_px"] == 44
    assert len(PARAMETER_UI) == len(DEFAULT_PARAMS)
    palette = json.loads(PALETTE.read_text(encoding="utf-8"))
    assert len(palette["colors"]) == 21

def test_parameter_validation_rejects_unknown_or_out_of_range_values():
    assert validate_params({"grid_cols": 48})["grid_cols"] == 48
    with pytest.raises(ValueError):
        validate_params({"grid_cols": 121})
    with pytest.raises(ValueError):
        validate_params({"invented_filter": 1})
    with pytest.raises(ValueError):
        validate_params({"denoise_method": "gaussian"})

def test_notebook_v2_runs_all_four_engines_and_saves_selected_output(tmp_path):
    image_path = tmp_path / "mapping_sample.png"
    arr = np.zeros((96, 128, 3), dtype=np.uint8)
    arr[:, :64] = (210, 55, 65)
    arr[:, 64:] = (45, 90, 185)
    Image.fromarray(arr).save(image_path)
    out = tmp_path / "run"
    run = run_image(image_path, PALETTE, DEFAULT_PARAMS, "C", out, include_stability=True)
    assert set(run["engines"]) == set(ENGINE_NAMES) == {"A", "B", "C", "D"}
    assert np.load(out / "labels.npy").shape == run["engines"]["C"]["labels"].shape
    assert (out / "engine_comparison_metrics.json").is_file()
    assert (out / "engine_A" / "numbered_preview.png").is_file()
    assert (out / "engine_D" / "color_preview.png").is_file()
    assert json.loads((out / "config.json").read_text())["selected_engine"] == "C"
    stability = json.loads((out / "stability_metrics.json").read_text())
    assert [x["Engine"].split(" — ")[0] for x in stability] == ["B", "C", "D"]
