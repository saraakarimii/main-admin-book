"""Frozen, serialisable configuration. Defaults mirror the reference notebook's sliders."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, field, asdict, fields


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class FramingConfig:
    crop_enabled: bool = False
    crop: tuple = (0.0, 0.0, 1.0, 1.0)      # normalised (x0, y0, x1, y1) of the ORIGINAL image
    content_zoom: float = 1.0               # >=1: shrink the processed window (a real processing op)
    focus: tuple = (0.5, 0.5)               # zoom anchor inside the cropped region (0..1)
    padding: float = 0.0                    # fraction of window size added as white border
    snap_square_cells: bool = True          # trim <= half a cell so cells are exactly square
    working_max_side: int = 900             # 0 = keep full resolution (notebook default: 900)


@dataclass(frozen=True)
class PreprocessConfig:
    brightness: float = 0.0                 # L* offset
    contrast: float = 1.0                   # multiplier around L*=50
    illumination_strength: float = 0.25
    illumination_sigma: float = 0.10
    clahe_clip: float = 1.5
    clahe_tiles: int = 8
    denoise_method: str = "bilateral"       # none | bilateral | nlm | median
    denoise_strength: float = 18.0
    sharpen_fine: float = 0.15
    sharpen_coarse: float = 0.05


@dataclass(frozen=True)
class EdgeConfig:
    auto_canny: bool = True
    canny_low: int = 55
    canny_high: int = 145
    close_iter: int = 1
    scharr_weight: float = 0.75
    sensitivity: float = 1.0                # gain on the final boundary map (clipped to 0..1)


@dataclass(frozen=True)
class MappingConfig:
    grid_cols: int = 40
    max_samples: int = 96
    trim_fraction: float = 0.03
    sampling: str = "stratified"            # stratified | legacy (notebook-identical)
    smoothness_lambda: float = 1.8
    boundary_beta: float = 4.0
    icm_iterations: int = 5
    slic_segments: int = 320
    slic_compactness: float = 10.0
    region_balance: float = 0.25


@dataclass(frozen=True)
class RenderConfig:
    preview_cell_px: int = 16
    numbered_cell_px: int = 44
    line_width: int = 2
    number_scale: float = 0.34              # font height as fraction of cell (upper bound)
    number_gray: int = 10                   # 0 = black text
    font_path: str | None = None


@dataclass(frozen=True)
class PipelineConfig:
    framing: FramingConfig = field(default_factory=FramingConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    edges: EdgeConfig = field(default_factory=EdgeConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        def mk(klass, sub):
            kw = {}
            for f in fields(klass):
                if f.name in sub:
                    v = sub[f.name]
                    kw[f.name] = tuple(v) if isinstance(v, list) else v
            return klass(**kw)
        return cls(framing=mk(FramingConfig, d.get("framing", {})),
                   preprocess=mk(PreprocessConfig, d.get("preprocess", {})),
                   edges=mk(EdgeConfig, d.get("edges", {})),
                   mapping=mk(MappingConfig, d.get("mapping", {})),
                   render=mk(RenderConfig, d.get("render", {})),
                   seed=int(d.get("seed", 42)))

    def key(self, *sections) -> str:
        d = self.to_dict()
        return _h({s: d[s] for s in sections})
