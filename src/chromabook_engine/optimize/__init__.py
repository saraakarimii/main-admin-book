"""Phase 2: diagnostics, framing proposals, candidate search, ranking and confidence."""
from .constraints import PrintSpec, feasible_cols, cell_size_mm          # noqa: F401
from .diagnostics import diagnose                                         # noqa: F401
from .autocrop import propose_framings                                    # noqa: F401
from .optimizer import OptimizerConfig, optimize, OptimizationReport, write_report      # noqa: F401
