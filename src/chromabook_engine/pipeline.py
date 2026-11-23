"""Staged pipeline with independent cache keys per stage.

frame -> preprocess -> edges / unary / slic -> engine mapping (+metrics)
Rendering settings are NOT part of any mapping key: changing number size or line width
re-renders only, it never re-runs preprocessing or mapping.
"""
from __future__ import annotations
import hashlib, json, platform, time
from dataclasses import dataclass, field
import numpy as np
import cv2, scipy, skimage, PIL
from . import ALGORITHM_VERSION, __version__
from .color import rgb_to_lab
from .config import PipelineConfig
from .edges import extract_edges
from .engines import ENGINE_NAMES, run_engine
from .grid import make_geometry
from .imaging import NormalizedImage, frame_image
from .metrics import evaluate_grid, delta_map_rgb, palette_usage
from .preprocess import preprocess
from .render import render_preview, render_preview_analysis, render_numbered
from .slic_map import make_slic
from .unary import build_cell_unary


def _hash(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:20]


def label_hash(labels) -> str:
    a = np.ascontiguousarray(labels.astype(np.uint8))
    return hashlib.sha256(bytes(str(a.shape), "ascii") + a.tobytes()).hexdigest()


class StageCache:
    def __init__(self):
        self._d, self.computed, self.hits = {}, {}, {}

    def get_or_compute(self, stage, key, fn):
        k = (stage, key)
        if k in self._d:
            self.hits[stage] = self.hits.get(stage, 0) + 1
            return self._d[k]
        val = fn()
        self._d[k] = val
        self.computed[stage] = self.computed.get(stage, 0) + 1
        return val


@dataclass
class EngineOutput:
    result: object            # EngineResult
    metrics: dict
    delta_map: np.ndarray
    label_hash: str


@dataclass
class RunResult:
    source: NormalizedImage
    palette: object
    config: PipelineConfig
    frame: object
    stages: dict
    edge_map: np.ndarray
    canny: np.ndarray
    gradient: np.ndarray
    canny_thresholds: tuple
    geometry: object
    slic_labels: np.ndarray
    slic_boundary: np.ndarray
    lab: np.ndarray
    unary: np.ndarray
    engines: dict
    keys: dict
    timings: dict = field(default_factory=dict)
    versions: dict = field(default_factory=dict)


def _versions():
    return {"chromabook_engine": __version__, "algorithm": ALGORITHM_VERSION,
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "opencv": cv2.__version__, "scikit-image": skimage.__version__, "pillow": PIL.__version__}


class Pipeline:
    def __init__(self, cache: StageCache | None = None):
        self.cache = cache or StageCache()

    # ------------------------------------------------------------------ stages
    def prepare(self, source: NormalizedImage, palette, cfg: PipelineConfig) -> dict:
        c, m = self.cache, cfg.mapping
        np.random.seed(cfg.seed)
        t = {}
        k_frame = _hash(source.sha256, cfg.to_dict()["framing"], m.grid_cols)
        t0 = time.time()
        frame = c.get_or_compute("frame", k_frame, lambda: frame_image(source.rgb, cfg.framing, m.grid_cols))
        t["frame"] = time.time() - t0
        k_pre = _hash(k_frame, cfg.to_dict()["preprocess"])
        t0 = time.time()
        stages = c.get_or_compute("preprocess", k_pre, lambda: preprocess(frame.rgb, cfg.preprocess))
        t["preprocess"] = time.time() - t0
        processed = stages["processed"]
        lab = c.get_or_compute("lab", k_pre, lambda: rgb_to_lab(processed))
        geo = make_geometry(processed.shape[0], processed.shape[1], frame.cols, frame.rows)
        k_edge = _hash(k_pre, cfg.to_dict()["edges"])
        t0 = time.time()
        edge = c.get_or_compute("edges", k_edge, lambda: extract_edges(processed, cfg.edges))
        t["edges"] = time.time() - t0
        k_unary = _hash(k_pre, palette.sha256, m.grid_cols, m.max_samples, m.trim_fraction, m.sampling)
        t0 = time.time()
        unary = c.get_or_compute("unary", k_unary, lambda: build_cell_unary(
            lab, geo, palette.lab, m.max_samples, m.trim_fraction, m.sampling))
        t["unary"] = time.time() - t0
        k_slic = _hash(k_pre, m.slic_segments, m.slic_compactness)
        t0 = time.time()
        slic = c.get_or_compute("slic", k_slic, lambda: make_slic(processed, m.slic_segments, m.slic_compactness))
        t["slic"] = time.time() - t0
        return {"frame": frame, "stages": stages, "lab": lab, "geo": geo, "edge": edge, "unary": unary,
                "slic": slic, "keys": {"frame": k_frame, "pre": k_pre, "edge": k_edge,
                                       "unary": k_unary, "slic": k_slic}, "timings": t}

    def _engine_key(self, prep, key, cfg):
        m = cfg.mapping
        extra = {"A": (), "B": (),
                 "C": (m.smoothness_lambda, m.boundary_beta, m.icm_iterations),
                 "D": (m.smoothness_lambda, m.boundary_beta, m.icm_iterations, m.region_balance,
                       m.max_samples, m.trim_fraction, m.slic_segments, m.slic_compactness)}[key]
        k = prep["keys"]
        return _hash(key, k["unary"], k["edge"], k["slic"], extra)

    def map_engine(self, prep, palette, cfg, key) -> EngineOutput:
        def compute():
            res = run_engine(key, lab=prep["lab"], geo=prep["geo"], pal_lab=palette.lab, unary=prep["unary"],
                             edge_map=prep["edge"][0], slic_labels=prep["slic"][0],
                             slic_boundary=prep["slic"][1], mapping=cfg.mapping)
            mapped = render_preview_analysis(res.labels, prep["geo"], palette)
            metrics, de = evaluate_grid(prep["stages"]["reference"], prep["stages"]["processed"],
                                        mapped, res.labels, len(palette))
            metrics["runtime_seconds"] = res.runtime_seconds
            return EngineOutput(res, metrics, de, label_hash(res.labels))
        return self.cache.get_or_compute("engine:" + key, self._engine_key(prep, key, cfg), compute)

    # --------------------------------------------------------------------- run
    def run(self, source: NormalizedImage, palette, cfg: PipelineConfig | None = None,
            engines=("A", "B", "C", "D")) -> RunResult:
        cfg = cfg or PipelineConfig()
        t0 = time.time()
        prep = self.prepare(source, palette, cfg)
        outs = {k: self.map_engine(prep, palette, cfg, k) for k in engines}
        timings = {**prep["timings"], "total": time.time() - t0}
        return RunResult(source=source, palette=palette, config=cfg, frame=prep["frame"],
                         stages=prep["stages"], edge_map=prep["edge"][0], canny=prep["edge"][1],
                         gradient=prep["edge"][2], canny_thresholds=prep["edge"][3],
                         geometry=prep["geo"], slic_labels=prep["slic"][0], slic_boundary=prep["slic"][1],
                         lab=prep["lab"], unary=prep["unary"], engines=outs, keys=prep["keys"],
                         timings=timings, versions=_versions())

    # ------------------------------------------------------------- rendering
    @staticmethod
    def render(result: RunResult, key: str, render_cfg=None) -> dict:
        """Pure re-render from the stored label matrix; never touches mapping."""
        rc = render_cfg or result.config.render
        labels = result.engines[key].result.labels
        numbered, report = render_numbered(labels, result.palette, rc.numbered_cell_px, rc.line_width,
                                           rc.number_scale, rc.number_gray, rc.font_path)
        return {"color_preview": render_preview(labels, result.palette, rc.preview_cell_px),
                "numbered_outline": numbered, "numbered_report": report}

    # ------------------------------------------------------------- stability
    def stability(self, result: RunResult, key: str) -> dict:
        """+/-1 L* brightness perturbation; fraction of cells whose label is unchanged."""
        cfg, m = result.config, result.config.mapping
        base = result.engines[key].result.labels
        agree = []
        for dl in (-1.0, 1.0):
            pert = result.lab.copy()
            pert[:, :, 0] = np.clip(pert[:, :, 0] + dl, 0, 100)
            unary = build_cell_unary(pert, result.geometry, result.palette.lab, m.max_samples,
                                     m.trim_fraction, m.sampling)
            out = run_engine(key, lab=pert, geo=result.geometry, pal_lab=result.palette.lab, unary=unary,
                             edge_map=result.edge_map, slic_labels=result.slic_labels,
                             slic_boundary=result.slic_boundary, mapping=m)
            agree.append(float(np.mean(out.labels == base)) * 100.0)
        return {"engine": ENGINE_NAMES[key], "agreement_L_minus1_pct": agree[0],
                "agreement_L_plus1_pct": agree[1], "mean_stability_pct": float(np.mean(agree))}
