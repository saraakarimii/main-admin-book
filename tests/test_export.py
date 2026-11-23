from helpers import *
import csv, json, tempfile
from chromabook_engine.export import export_asset, export_all

SRC = synth_source()
RES = Pipeline().run(SRC, PAL, small_cfg())


def test_asset_package_contents_and_binary_formats():
    with tempfile.TemporaryDirectory() as d:
        out = export_asset(RES, "C", d)
        for f in ("original.png", "cropped_source.png", "preprocessed.png", "color_preview.png",
                  "numbered_outline.png", "grid_indices.bin", "grid_indices.npy", "cell_delta_e.bin",
                  "palette.json", "metadata.json", "metrics.json", "config.json"):
            assert (out / f).exists(), f
        lab = RES.engines["C"].result.labels
        rows, cols = lab.shape
        raw = np.fromfile(out / "grid_indices.bin", dtype=np.uint8)
        assert raw.size == rows * cols and (raw.reshape(rows, cols) == lab).all()      # row-major uint8
        assert (np.load(out / "grid_indices.npy") == lab).all()
        cost = np.fromfile(out / "cell_delta_e.bin", dtype="<f4")
        assert cost.size == rows * cols
        assert np.allclose(cost.reshape(rows, cols), RES.engines["C"].result.selected_cost)
        meta = json.loads((out / "metadata.json").read_text())
        assert meta["cell_cost_kind"].startswith("ciede2000")
        assert meta["palette"]["index_to_printed_id"] == PAL.ids and meta["palette"]["sha256"] == PAL.sha256
        assert meta["label_sha256"] == RES.engines["C"].label_hash
        ids = list(csv.reader(open(out / "grid_color_ids.csv", encoding="utf-8-sig")))
        assert ids[0][0] == PAL.ids[lab[0, 0]]
        snap = json.loads((out / "palette.json").read_text())
        assert [c["id"] for c in snap["colors"]] == PAL.ids


def test_engine_A_cost_kind_is_not_mislabelled():
    with tempfile.TemporaryDirectory() as d:
        out = export_asset(RES, "A", d)
        meta = json.loads((out / "metadata.json").read_text())
        assert "deltaE76" in meta["cell_cost_kind"]


def test_export_all_creates_zip_with_comparison_csv():
    with tempfile.TemporaryDirectory() as d:
        z = export_all(RES, Path(d) / "asset")
        assert z.exists() and z.suffix == ".zip"
        assert (Path(d) / "asset" / "engine_comparison_metrics.csv").exists()
