"""Vector print PDF (ReportLab). Numbered grid, palette guide and swatches are drawn as vector paths and text:
no raster images are embedded. RGB print-ready PDF; NOT PDF/X (needs an ICC pipeline - deferred)."""
from __future__ import annotations
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color
from .geometry import MM, SIDE_BY_SIDE, FULL_PAGE, GuideSpec, all_plans


def _lum(rgb):
    # WCAG relative luminance, with sRGB linearisation.
    vals = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb]
    return 0.2126 * vals[0] + 0.7152 * vals[1] + 0.0722 * vals[2]


def _contrast_text_color(rgb):
    """Choose readable black/white label text for an sRGB triplet in [0, 1]."""
    lum = _lum(rgb)
    white = 1.05 / (lum + 0.05)
    black = (lum + 0.05) / 0.05
    return Color(1, 1, 1) if white >= black else Color(0, 0, 0)


def draw_artwork(c, page_h_mm, slot_origin_mm, plan, labels, palette, guide: GuideSpec, colored=False):
    ox, oy = slot_origin_mm
    X = lambda x: (ox + x) * MM
    Y = lambda y: (page_h_mm - (oy + y)) * MM         # y measured from the top of the page, in mm
    ax, ay, aw, ah = plan.boxes["artwork"]
    cell = plan.cell_mm
    rows, cols = labels.shape
    if colored:
        for r in range(rows):
            for k in range(cols):
                rgb = palette.rgb[int(labels[r, k])] / 255.0
                c.setFillColor(Color(*rgb)); c.setStrokeColor(Color(*rgb))
                c.rect(X(ax + k * cell), Y(ay + (r + 1) * cell), cell * MM, cell * MM, stroke=0, fill=1)
    # grid lines: each boundary drawn once as one path
    c.setStrokeColor(Color(0.35, 0.35, 0.35)); c.setLineWidth(0.12 * MM)
    p = c.beginPath()
    for r in range(1, rows):
        p.moveTo(X(ax), Y(ay + r * cell)); p.lineTo(X(ax + aw), Y(ay + r * cell))
    for k in range(1, cols):
        p.moveTo(X(ax + k * cell), Y(ay)); p.lineTo(X(ax + k * cell), Y(ay + ah))
    c.drawPath(p, stroke=1, fill=0)
    c.setStrokeColor(Color(0, 0, 0)); c.setLineWidth(0.3 * MM); c.rect(X(ax), Y(ay + ah), aw * MM, ah * MM, stroke=1, fill=0)
    # numbers: printed colour IDs (strings from the palette), never array indices
    fs = 0.52 * cell * MM
    c.setFont("Helvetica", fs)
    for r in range(rows):
        for k in range(cols):
            idx = int(labels[r, k])
            txt = palette.id_of(idx)
            if colored:
                c.setFillColor(_contrast_text_color(palette.rgb[idx] / 255.0))
            else:
                c.setFillColor(Color(0.04, 0.04, 0.04))
            c.drawCentredString(X(ax + (k + 0.5) * cell), Y(ay + (r + 0.5) * cell) - 0.36 * fs, txt)
    # palette guide: colours used, ordered by printed ID
    gx, gy, gw, gh = plan.boxes["guide"]
    used = sorted({int(i) for i in labels.ravel()}, key=lambda i: palette.id_of(i))
    cols_g, rows_g = plan.guide["cols"], plan.guide["rows"]
    for n, idx in enumerate(used):
        col, row = divmod(n, rows_g) if plan.placement == "side" else (n % cols_g, n // cols_g)
        ex = gx + col * (guide.entry_w + guide.column_gap); ey = gy + row * guide.entry_h
        rgb = palette.rgb[idx] / 255.0
        c.setFillColor(Color(*rgb)); c.setStrokeColor(Color(0, 0, 0)); c.setLineWidth(0.2 * MM)
        c.rect(X(ex), Y(ey + guide.swatch + (guide.entry_h - guide.swatch) / 2), guide.swatch * MM, guide.swatch * MM, stroke=1, fill=1)
        c.setFillColor(_contrast_text_color(rgb)); c.setFont("Helvetica-Bold", 7)
        c.drawString(X(ex + guide.swatch + 1.2), Y(ey + guide.entry_h / 2) - 2.4, palette.id_of(idx))
        c.setFont("Helvetica", 6.5)
        c.drawString(X(ex + guide.swatch + 8.2), Y(ey + guide.entry_h / 2) - 2.3, palette.colors[idx].name)


def build_pdf(path, layout, artworks, palette, guide: GuideSpec = GuideSpec(), min_cell_mm=3.0, colored=False):
    """artworks: [{'name', 'labels'}]. Slots are filled in the layout's reading order (right slot first for side-by-side).
    Returns one record per placed artwork (page, slot, plan)."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=(layout.page_w * MM, layout.page_h * MM))
    c.setTitle("ChromaBook Studio"); c.setAuthor("ChromaBook Studio")
    order = layout.fill_order(); boxes = layout.slot_boxes(); placed = []
    for i, art in enumerate(artworks):
        page, k = divmod(i, layout.slots)
        if k == 0 and i > 0: c.showPage()
        slot = order[k]
        rows, cols = art["labels"].shape
        n = len({int(v) for v in art["labels"].ravel()})
        plans = [p for p in all_plans(cols, rows, n, (layout,), guide, min_cell_mm) if p.meets_min_cell]
        if not plans:
            raise ValueError(f"{art['name']}: cell size below {min_cell_mm} mm in layout {layout.name}")
        plan = max(plans, key=lambda p: p.cell_mm)
        draw_artwork(c, layout.page_h, (boxes[slot][0], boxes[slot][1]), plan, art["labels"], palette, guide, colored)
        placed.append({"name": art["name"], "page": page + 1, "slot_index_left_to_right": slot, "plan": plan})
    c.showPage(); c.save()
    return placed
