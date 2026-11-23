"""Image loading / normalisation and framing (crop, content zoom, padding, square-cell snap).

The original is never modified: loading returns a new array, framing returns new arrays.
"""
from __future__ import annotations
import hashlib, io
from dataclasses import dataclass, field
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageCms


class ImageIntegrityError(ValueError):
    pass


@dataclass
class NormalizedImage:
    rgb: np.ndarray                 # uint8 HxWx3 sRGB, composited on white, EXIF-corrected
    sha256: str                     # SHA256 of the ORIGINAL FILE BYTES
    filename: str
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.rgb.setflags(write=False)  # guard against accidental in-place edits


def normalize_image(data: bytes, filename: str = "image") -> NormalizedImage:
    sha = hashlib.sha256(data).hexdigest()
    try:
        probe = Image.open(io.BytesIO(data)); probe.verify()
        im = Image.open(io.BytesIO(data)); im.load()
    except Exception as e:  # corrupt / unsupported
        raise ImageIntegrityError(f"{filename}: cannot decode image ({e})") from e
    meta = {"format": im.format, "mode": im.mode, "original_size": list(im.size),
            "has_icc_profile": bool(im.info.get("icc_profile")),
            "exif_orientation_applied": False, "icc_converted_to_srgb": False,
            "had_alpha": im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info}
    if im.mode in ("I;16", "I;16L", "I;16B", "I"):
        arr = np.asarray(im, dtype=np.float32)
        arr = arr / max(1.0, float(arr.max())) * 255.0
        im = Image.fromarray(arr.astype(np.uint8), "L")
    oriented = ImageOps.exif_transpose(im)
    if oriented.size != im.size or oriented is not im:
        meta["exif_orientation_applied"] = oriented.size != im.size or \
            (im.getexif().get(274, 1) not in (0, 1))
    im = oriented
    icc = im.info.get("icc_profile")
    if icc and im.mode in ("RGB", "RGBA", "CMYK", "L"):
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            dst = ImageCms.createProfile("sRGB")
            mode = "RGBA" if im.mode == "RGBA" else "RGB"
            im = ImageCms.profileToProfile(im.convert(mode) if im.mode != "CMYK" else im,
                                           src, dst, outputMode=mode)
            meta["icc_converted_to_srgb"] = True
        except Exception:
            meta["icc_converted_to_srgb"] = False
    rgba = im.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = np.asarray(Image.alpha_composite(bg, rgba).convert("RGB"), dtype=np.uint8).copy()
    meta["normalized_size"] = [rgb.shape[1], rgb.shape[0]]
    return NormalizedImage(rgb=rgb, sha256=sha, filename=Path(filename).name, meta=meta)


def load_image(path) -> NormalizedImage:
    p = Path(path)
    return normalize_image(p.read_bytes(), p.name)


# ----------------------------------------------------------------------------
# Framing
# ----------------------------------------------------------------------------
@dataclass
class FrameResult:
    rgb: np.ndarray            # framed + resized working image (uint8)
    rows: int
    cols: int
    steps: dict                # every geometric decision, for the report
    bbox_original_px: tuple    # (x0,y0,x1,y1) in original pixels (ignores padding)
    bbox_original_norm: tuple


def _resize_like_notebook(arr: np.ndarray, max_side: int) -> np.ndarray:
    hh, ww = arr.shape[:2]
    scale = min(1.0, max_side / max(hh, ww)) if max_side and max_side > 0 else 1.0
    if scale < 1:
        return cv2.resize(arr, (max(16, round(ww * scale)), max(16, round(hh * scale))),
                          interpolation=cv2.INTER_AREA)
    return arr.copy()


def frame_image(rgb: np.ndarray, cfg, cols: int) -> FrameResult:
    """Crop -> content zoom -> padding -> square-cell snap -> working-resolution resize.

    Crop, content zoom and display zoom are different things; this function implements the
    first two as real changes to the pixels that reach the mapping engines.
    """
    H, W = rgb.shape[:2]
    steps = {}
    # 1. crop (normalised coordinates of the original)
    if cfg.crop_enabled:
        x0, y0, x1, y1 = [float(v) for v in cfg.crop]
        x0, y0 = float(np.clip(x0, 0, 0.98)), float(np.clip(y0, 0, 0.98))
        x1, y1 = float(np.clip(x1, x0 + 0.02, 1.0)), float(np.clip(y1, y0 + 0.02, 1.0))
    else:
        x0, y0, x1, y1 = 0.0, 0.0, 1.0, 1.0
    px0, py0 = int(round(x0 * W)), int(round(y0 * H))
    px1, py1 = max(px0 + 1, int(round(x1 * W))), max(py0 + 1, int(round(y1 * H)))
    steps["crop_px"] = [px0, py0, px1, py1]
    # 2. content zoom: shrink the window around the focus point
    z = max(1.0, float(cfg.content_zoom))
    cw, ch = px1 - px0, py1 - py0
    if z > 1.0:
        zw, zh = max(8, int(round(cw / z))), max(8, int(round(ch / z)))
        fx, fy = float(np.clip(cfg.focus[0], 0, 1)), float(np.clip(cfg.focus[1], 0, 1))
        zx0 = px0 + int(round(np.clip(fx * cw - zw / 2, 0, cw - zw)))
        zy0 = py0 + int(round(np.clip(fy * ch - zh / 2, 0, ch - zh)))
        px0, py0, px1, py1 = zx0, zy0, zx0 + zw, zy0 + zh
    steps["content_zoom"] = z
    steps["zoom_window_px"] = [px0, py0, px1, py1]
    arr = rgb[py0:py1, px0:px1].copy()
    bbox_px = (px0, py0, px1, py1)
    # 3. padding (white)
    if cfg.padding > 0:
        pad_x, pad_y = int(round(arr.shape[1] * cfg.padding)), int(round(arr.shape[0] * cfg.padding))
        arr = np.pad(arr, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), constant_values=255)
        steps["padding_px"] = [pad_x, pad_y]
    # 4. snap to square cells (trim at most half a cell, centred)
    h, w = arr.shape[:2]
    cols = int(np.clip(cols, 4, 180))
    if cfg.snap_square_cells:
        cell_w = w / cols
        rows = max(4, int(round(h / cell_w)))
        cell = min(w / cols, h / rows)
        nw, nh = int(round(cols * cell)), int(round(rows * cell))
        ox, oy = (w - nw) // 2, (h - nh) // 2
        arr = arr[oy:oy + nh, ox:ox + nw]
        steps["snap_trim_px"] = {"x": [ox, w - nw - ox], "y": [oy, h - nh - oy]}
        steps["cell_px_before_resize"] = cell
        if cfg.padding == 0:
            bbox_px = (bbox_px[0] + ox, bbox_px[1] + oy, bbox_px[0] + ox + nw, bbox_px[1] + oy + nh)
    else:
        rows = max(4, int(round(cols * h / w)))
    # 5. working resolution (analysis resolution; separate from the stored original)
    arr = _resize_like_notebook(arr, cfg.working_max_side)
    steps["working_size"] = [arr.shape[1], arr.shape[0]]
    norm = (bbox_px[0] / W, bbox_px[1] / H, bbox_px[2] / W, bbox_px[3] / H)
    return FrameResult(arr, rows, cols, steps, tuple(int(v) for v in bbox_px),
                       tuple(round(v, 6) for v in norm))
