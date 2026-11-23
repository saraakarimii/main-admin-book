import io, os, sys, unittest
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chromabook_engine import (Palette, Pipeline, PipelineConfig, normalize_image)  # noqa: E402
from chromabook_engine.config import (FramingConfig, MappingConfig, PreprocessConfig,  # noqa: E402
                                      EdgeConfig, RenderConfig)

PAL = Palette.from_file(ROOT / "data" / "main-21.palette.json")


def skip(msg):
    raise unittest.SkipTest(msg)


def synth_rgb(h=240, w=160, seed=7, noise=14.0):
    """Deterministic textured test image: gradient sky, red disc, blue block, green ground."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.float32)
    img[...] = np.stack([200 - 60 * y / h, 220 - 30 * y / h, 250 - 20 * x / w], -1)
    disc = (x - w * 0.35) ** 2 + (y - h * 0.35) ** 2 < (w * 0.22) ** 2
    img[disc] = (215, 45, 55)
    img[int(h * 0.62):int(h * 0.82), int(w * 0.55):int(w * 0.9)] = (50, 80, 190)
    img[int(h * 0.85):] = (70, 160, 80)
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def png_bytes(arr, mode="RGB"):
    b = io.BytesIO(); Image.fromarray(arr, mode).save(b, "PNG"); return b.getvalue()


def synth_source(**kw):
    return normalize_image(png_bytes(synth_rgb(**kw)), "synthetic.png")


def small_cfg(**over):
    m = dict(grid_cols=12, max_samples=32, slic_segments=60, icm_iterations=3)
    m.update(over.pop("mapping", {}))
    fr = dict(working_max_side=0); fr.update(over.pop("framing", {}))
    return PipelineConfig(framing=FramingConfig(**fr), mapping=MappingConfig(**m), **over)
