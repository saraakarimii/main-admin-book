"""Optimise one image per book type, write layout report + vector PDFs.
usage: python tools/run_book.py IMAGE OUT_DIR"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chromabook_engine import load_image, Palette
from chromabook_engine.optimize import optimize, OptimizerConfig, write_report
from chromabook_engine.layout import (LayoutPrintSpec, SIDE_BY_SIDE, FULL_PAGE, layout_report, write_layout_report, build_pdf)

img, out = Path(sys.argv[1]), Path(sys.argv[2])
pal = Palette.from_file(Path(__file__).resolve().parents[1] / "data" / "main-21.palette.json")
src = load_image(img)
for lay in (SIDE_BY_SIDE, FULL_PAGE):
    d = out / lay.name
    rep = optimize(src, pal, LayoutPrintSpec(layouts=(lay,), palette_size=len(pal)), OptimizerConfig(final_side=900))
    write_report(rep, d)
    rows, cols = rep.labels.shape
    lr = layout_report([{"name": img.stem, "cols": cols, "rows": rows, "n_colors_used": rep.n_colors_used}])
    lr["page_specs"] = {lay.name: lr["page_specs"][lay.name]}
    write_layout_report(lr, d)
    arts = [{"name": img.stem, "labels": rep.labels}]
    build_pdf(d / f"{lay.name}_numbered.pdf", lay, arts, pal)
    build_pdf(d / f"{lay.name}_answer_key.pdf", lay, arts, pal, colored=True)
    c = lr["images"][0]["chosen"]
    print(lay.name, "grid", c["grid"], "cell", c["cell_mm"], "mm", "guide", c["guide_mm"], "art", c["artwork_final_mm"], rep.recommendation["engine"], rep.confidence["level"])
