"""ChromaBook Studio - core pixel-art color-by-number engine (Phase 1)."""
__version__ = "0.4.0"
ALGORITHM_VERSION = "cb-engine-1.0"  # bump when any mapping/metric algorithm changes

from .palette import Palette, PaletteError            # noqa: F401
from .config import (PipelineConfig, FramingConfig, PreprocessConfig,  # noqa: F401
                     EdgeConfig, MappingConfig, RenderConfig)
from .imaging import load_image, normalize_image, ImageIntegrityError  # noqa: F401
from .pipeline import Pipeline, StageCache, RunResult                  # noqa: F401
