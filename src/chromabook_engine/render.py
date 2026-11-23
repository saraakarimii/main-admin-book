"""Rendering. Every output here is a pure function of (labels, palette, render settings):
the colour preview and the numbered grid can never disagree about a cell."""
from __future__ import annotations
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .grid import Geometry

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)


def _find_font(path=None):
    cands = [path] if path else []
    cands += list(_FONT_CANDIDATES)
    try:
        import matplotlib
        cands.append(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"))
    except Exception:
        pass
    for p in cands:
        if p and os.path.exists(p):
            return p
    return None


def check_labels(labels, palette):
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("labels must be 2-D (rows x cols)")
    if labels.min() < 0 or labels.max() >= len(palette):
        raise ValueError("label outside palette index range")
    return labels


def render_preview_analysis(labels, geo: Geometry, palette) -> np.ndarray:
    """Colour preview at the working-image resolution (used for metrics against the source)."""
    labels = check_labels(labels, palette)
    h, w = int(geo.ye[-1]), int(geo.xe[-1])
    out = np.full((h, w, 3), 255, np.uint8)
    for r in range(geo.rows):
        for c in range(geo.cols):
            out[geo.ye[r]:geo.ye[r + 1], geo.xe[c]:geo.xe[c + 1]] = palette.rgb[int(labels[r, c])]
    return out


def render_preview(labels, palette, cell_px=16) -> np.ndarray:
    """Pixel-art colour preview: uniform square cells, nearest-neighbour (no smoothing)."""
    labels = check_labels(labels, palette)
    small = palette.rgb[labels]
    cp = int(max(1, cell_px))
    return np.repeat(np.repeat(small, cp, axis=0), cp, axis=1)


def render_numbered(labels, palette, cell_px=44, line_width=2, number_scale=0.34,
                    number_gray=10, font_path=None):
    """Printable white grid with the printed colour ID centred in each cell.

    Grid lines are drawn once per boundary (uniform thickness everywhere, including the
    outer border). One font size is used for the whole page, chosen so the longest ID used
    fits inside a cell.
    """
    labels = check_labels(labels, palette)
    rows, cols = labels.shape
    cp = int(np.clip(cell_px, 12, 200))
    lw = int(np.clip(line_width, 1, max(1, cp // 4)))
    W, H = cols * cp + lw, rows * cp + lw
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    for i in range(cols + 1):
        d.rectangle([i * cp, 0, i * cp + lw - 1, H - 1], fill=(15, 15, 15))
    for j in range(rows + 1):
        d.rectangle([0, j * cp, W - 1, j * cp + lw - 1], fill=(15, 15, 15))

    used = sorted({palette.id_of(i) for i in np.unique(labels)}, key=len)
    longest = used[-1] if used else "00"
    inner = cp - lw
    fpath = _find_font(font_path)
    size = max(6, int(cp * float(number_scale)))
    font = None
    while size >= 6:
        font = ImageFont.truetype(fpath, size) if fpath else ImageFont.load_default()
        b = d.textbbox((0, 0), longest, font=font)
        if (b[2] - b[0]) <= 0.82 * inner and (b[3] - b[1]) <= 0.70 * inner:
            break
        if not fpath:
            break
        size -= 1
    max_wr = max_hr = 0.0
    texts = []
    gray = int(np.clip(number_gray, 0, 120))
    for r in range(rows):
        row_txt = []
        for c in range(cols):
            text = palette.id_of(labels[r, c])
            row_txt.append(text)
            b = d.textbbox((0, 0), text, font=font)
            tw, th = b[2] - b[0], b[3] - b[1]
            cx = c * cp + (cp + lw) / 2.0
            cy = r * cp + (cp + lw) / 2.0
            d.text((cx - tw / 2.0 - b[0], cy - th / 2.0 - b[1]), text, fill=(gray,) * 3, font=font)
            max_wr, max_hr = max(max_wr, tw / inner), max(max_hr, th / inner)
        texts.append(row_txt)
    report = {"cell_px": cp, "line_width_px": lw, "font_size_px": size, "font_path": fpath,
              "image_size": [W, H], "max_text_width_ratio": round(max_wr, 4),
              "max_text_height_ratio": round(max_hr, 4),
              "all_numbers_fit": bool(max_wr <= 0.95 and max_hr <= 0.95),
              "cell_texts": texts}
    return np.asarray(img), report
