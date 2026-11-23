"""Parity: new modules vs the ORIGINAL notebook code, executed from the .ipynb itself.
Uses legacy sampling and no square-snap so the geometry matches the notebook exactly."""
from helpers import *
import json, sys, tempfile, types

NB_CANDIDATES = [os.environ.get("CHROMABOOK_NOTEBOOK", ""),
                 str(ROOT / "ChromaBook_Advanced_Mapping_Lab_v2.ipynb"),
                 "/mnt/user-data/uploads/ChromaBook_Advanced_Mapping_Lab_v2__1_.ipynb"]


def load_notebook_namespace():
    path = next((p for p in NB_CANDIDATES if p and os.path.exists(p)), None)
    if not path:
        skip("reference notebook not found (set CHROMABOOK_NOTEBOOK)")
    nb = json.load(open(path, encoding="utf-8"))
    code = "".join(next(c for c in nb["cells"] if c["cell_type"] == "code")["source"])
    code = code[:code.index("# 11) VISUALIZATION")]
    code = code.replace("_missing = [pkg for module, pkg in _required.items() if importlib.util.find_spec(module) is None]",
                        "_missing = []")
    code = code.replace('Path("/content/chromabook_advanced_exports")', f'Path({tempfile.mkdtemp()!r})')
    for name in ("ipywidgets", "IPython", "IPython.display"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["IPython.display"].display = lambda *a, **k: None
    sys.modules["IPython.display"].clear_output = lambda *a, **k: None
    ns = {"__name__": "nbref"}
    exec(compile(code, "notebook", "exec"), ns)
    return ns


def test_engines_match_notebook_exactly():
    ns = load_notebook_namespace()
    ns["PALETTE"] = ns["attach_palette_lab"](ns["normalize_palette"](
        [{"id": c.id, "name": c.name, "hex": c.hex} for c in PAL.colors]))
    img = Image.fromarray(synth_rgb(h=150, w=100, noise=10))
    p = dict(max_side=900, crop_enabled=False, crop_x0=0, crop_y0=0, crop_x1=100, crop_y1=100,
             illumination_strength=0.25, illumination_sigma=0.10, clahe_clip=1.5, clahe_tiles=8,
             denoise_method="bilateral", denoise_strength=18.0, sharpen_fine=0.15, sharpen_coarse=0.05,
             auto_canny=True, canny_low=55, canny_high=145, edge_close=1, grid_cols=12, max_samples=32,
             trim_fraction=0.03, smoothness_lambda=1.8, boundary_beta=4.0, icm_iterations=3, slic_segments=60,
             slic_compactness=10.0, region_balance=0.25, numbered_cell_px=44, numbered_line_width=2)
    nb_run = ns["run_all_engines"].__globals__  # noqa  (namespace holder)
    ns["pil_original"] = img
    ref = ns["run_all_engines"](p)

    cfg = PipelineConfig(framing=FramingConfig(snap_square_cells=False, working_max_side=900),
                         mapping=MappingConfig(grid_cols=12, max_samples=32, sampling="legacy",
                                               slic_segments=60, icm_iterations=3))
    src = normalize_image(png_bytes(np.asarray(img)), "p.png")
    new = Pipeline().run(src, PAL, cfg)

    assert np.array_equal(ref["pipeline"]["processed"], new.stages["processed"]), "preprocessing differs"
    assert np.allclose(ref["pipeline"]["edge_map"], new.edge_map, atol=1e-6), "edge map differs"
    assert np.allclose(ref["pipeline"]["unary"], new.unary, atol=1e-4), "unary costs differ"
    assert np.array_equal(ref["pipeline"]["slic_labels"], new.slic_labels), "SLIC differs"
    for k in "ABCD":
        assert np.array_equal(ref["engines"][k]["labels"], new.engines[k].result.labels), f"engine {k} labels differ"
        assert abs(ref["engines"][k]["metrics"]["DeltaE00_mean_vs_source"]
                   - new.engines[k].metrics["DeltaE00_mean_vs_source"]) < 1e-4, f"engine {k} dE00 differs"
