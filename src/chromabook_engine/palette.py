"""Palette handling.

Machine palette *index* (0..K-1, position in the array) and printed colour *ID*
(string such as "01") are strictly separate. IDs are never derived from indices.
"""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from skimage.color import rgb2lab


class PaletteError(ValueError):
    pass


def hex_to_rgb(value) -> tuple[int, int, int]:
    s = str(value).strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        raise PaletteError(f"Invalid HEX: {value!r}")
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError as e:
        raise PaletteError(f"Invalid HEX: {value!r}") from e


def _entries(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("colors", "palette", "items", "data"):
            val = payload.get(key)
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                for sub in ("colors", "items", "entries"):
                    if isinstance(val.get(sub), list):
                        return val[sub]
    raise PaletteError("Unrecognised palette JSON: need an array or a colors/palette key.")


@dataclass(frozen=True)
class Color:
    id: str          # printed identifier (string, e.g. "01")
    name: str
    hex: str
    rgb: tuple


class Palette:
    """Immutable palette snapshot with identity, version and content hash."""

    def __init__(self, colors, palette_id="unnamed", version="0.0.0", name=""):
        colors = list(colors)
        if len(colors) < 2:
            raise PaletteError("A palette needs at least two colours.")
        if len(colors) > 256:
            raise PaletteError("At most 256 colours are supported (uint8 grid storage).")
        ids = [c.id for c in colors]
        if len(set(ids)) != len(ids):
            raise PaletteError("Printed colour IDs must be unique.")
        self.colors = tuple(colors)
        self.palette_id = str(palette_id)
        self.version = str(version)
        self.name = str(name or palette_id)
        rgb = np.array([c.rgb for c in colors], dtype=np.float32) / 255.0
        self._lab = rgb2lab(rgb.reshape(1, -1, 3)).reshape(-1, 3).astype(np.float32)
        self._rgb = np.array([c.rgb for c in colors], dtype=np.uint8)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_payload(cls, payload, palette_id=None, version=None, name=None):
        meta = payload if isinstance(payload, dict) else {}
        out = []
        for i, entry in enumerate(_entries(payload)):
            if not isinstance(entry, dict):
                raise PaletteError(f"Colour record {i + 1} must be an object.")
            nested = entry.get("color") if isinstance(entry.get("color"), dict) else {}
            item = {**nested, **entry}
            cid = item.get("id", item.get("code", item.get("number")))
            if cid is None or str(cid).strip() == "":
                raise PaletteError(
                    f"Colour record {i + 1} has no printed ID; IDs are never guessed from position.")
            hx = item.get("hex") or item.get("hexCode") or item.get("hex_code")
            rgbv = item.get("rgb")
            if hx:
                rgb = hex_to_rgb(hx)
            elif isinstance(rgbv, (list, tuple)) and len(rgbv) >= 3:
                rgb = tuple(int(np.clip(v, 0, 255)) for v in rgbv[:3])
            else:
                raise PaletteError(f"Colour record {i + 1} has neither hex nor rgb.")
            out.append(Color(str(cid).strip(), str(item.get("name", item.get("label", ""))),
                             "#{:02X}{:02X}{:02X}".format(*rgb), tuple(int(v) for v in rgb)))
        return cls(out,
                   palette_id or meta.get("palette_id") or "unnamed",
                   version or meta.get("version") or "0.0.0",
                   name or meta.get("name") or "")

    @classmethod
    def from_file(cls, path, **kw):
        return cls.from_payload(json.loads(Path(path).read_text(encoding="utf-8")), **kw)

    # -- views -------------------------------------------------------------
    def __len__(self):
        return len(self.colors)

    @property
    def ids(self) -> list[str]:
        return [c.id for c in self.colors]

    @property
    def lab(self) -> np.ndarray:
        return self._lab

    @property
    def rgb(self) -> np.ndarray:
        return self._rgb

    def id_of(self, index: int) -> str:
        return self.colors[int(index)].id

    def index_of(self, color_id: str) -> int:
        return self.ids.index(str(color_id))

    def to_payload(self) -> dict:
        return {"palette_id": self.palette_id, "name": self.name, "version": self.version,
                "colors": [{"id": c.id, "name": c.name, "hex": c.hex, "rgb": list(c.rgb)}
                           for c in self.colors]}

    @property
    def sha256(self) -> str:
        """Hash of the canonical colour list (ids, names, hex) - detects any silent edit."""
        canon = json.dumps([[c.id, c.name, c.hex] for c in self.colors], ensure_ascii=False)
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    def snapshot(self) -> dict:
        return {**self.to_payload(), "sha256": self.sha256}
