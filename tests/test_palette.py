from helpers import *
from chromabook_engine import PaletteError
import json


def test_printed_ids_are_strings_distinct_from_indices():
    assert PAL.ids[0] == "01" and PAL.ids[20] == "21"
    assert PAL.index_of("21") == 20 and PAL.id_of(0) == "01"
    assert all(isinstance(i, str) for i in PAL.ids)
    assert PAL.ids != [str(i) for i in range(len(PAL))]       # id != index, ever


def test_missing_id_is_rejected_never_guessed():
    try:
        Palette.from_payload([{"hex": "#000000"}, {"hex": "#FFFFFF"}])
    except PaletteError:
        return
    raise AssertionError("palette without printed IDs must be rejected")


def test_duplicate_ids_rejected():
    try:
        Palette.from_payload([{"id": "1", "hex": "#000"}, {"id": "1", "hex": "#fff"}])
    except PaletteError:
        return
    raise AssertionError("duplicate IDs must be rejected")


def test_palette_hash_detects_edits_and_roundtrips():
    p = json.loads((ROOT / "data" / "main-21.palette.json").read_text())
    p2 = json.loads(json.dumps(p)); p2["colors"][3]["hex"] = "#D5D7DB"
    assert Palette.from_payload(p).sha256 == PAL.sha256
    assert Palette.from_payload(p2).sha256 != PAL.sha256
    back = Palette.from_payload(PAL.to_payload())
    assert back.ids == PAL.ids and back.sha256 == PAL.sha256


def test_palette_identity_metadata():
    assert PAL.palette_id == "main-21" and PAL.version == "1.0.0" and len(PAL) == 21
