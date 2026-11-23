"""Phase 5 foundation: real A4 geometry, palette-guide allocation, physical cell size, layout report, vector PDF."""
from .geometry import (PageLayout, SIDE_BY_SIDE, FULL_PAGE, GuideSpec, SlotPlan, plan_slot,   # noqa: F401
                       best_plan, LayoutPrintSpec, MM)
from .report import layout_report, write_layout_report                                        # noqa: F401
from .pdf import build_pdf                                                                       # noqa: F401

from .book import assemble_book, preflight_book  # noqa: F401
