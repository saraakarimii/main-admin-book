from helpers import *
from chromabook_engine.imaging import ImageIntegrityError, normalize_image
from chromabook_engine.pipeline import label_hash
from chromabook_engine.render import render_preview
from PIL import Image
import io


def test_crop_and_content_zoom_change_the_processed_pixels():
    src = synth_source(h=300, w=200)
    base = Pipeline().prepare(src, PAL, small_cfg())["frame"]
    crop = Pipeline().prepare(src, PAL, small_cfg(framing={"crop_enabled": True, "crop": (0.1, 0.1, 0.9, 0.9)}))["frame"]
    zoom = Pipeline().prepare(src, PAL, small_cfg(framing={"content_zoom": 2.0}))["frame"]
    assert base.bbox_original_norm == (0.0, 0.0, 1.0, 1.0)
    bx0, by0, bx1, by1 = crop.bbox_original_norm
    assert 0.08 < bx0 < 0.15 and 0.85 < bx1 < 0.92
    zx0, zy0, zx1, zy1 = zoom.bbox_original_norm
    assert abs((zx1 - zx0) - 0.5) < 0.06 and abs((zy1 - zy0) - 0.5) < 0.06      # window is half-size
    # zoom really changes the source content seen by the engines
    assert not np.array_equal(base.rgb[:50, :50], zoom.rgb[:50, :50])
    # restoring the full frame restores the original bbox
    full = Pipeline().prepare(src, PAL, small_cfg(framing={"crop_enabled": False}))["frame"]
    assert full.bbox_original_norm == (0.0, 0.0, 1.0, 1.0)


def test_display_zoom_never_changes_labels():
    res = Pipeline().run(synth_source(), PAL, small_cfg(), engines=("B",))
    h1 = label_hash(res.engines["B"].result.labels)
    for cp in (4, 16, 40):
        prev = render_preview(res.engines["B"].result.labels, PAL, cp)
        assert prev.shape[1] == res.engines["B"].result.labels.shape[1] * cp
    assert label_hash(res.engines["B"].result.labels) == h1


def test_stage_caches_are_independent():
    pipe, src = Pipeline(), synth_source()
    cfg = small_cfg()
    pipe.run(src, PAL, cfg)
    base = dict(pipe.cache.computed)
    # render-only changes: not part of any mapping key
    from dataclasses import replace
    cfg_r = replace(cfg, render=RenderConfig(numbered_cell_px=60, line_width=3, number_scale=0.4))
    res = pipe.run(src, PAL, cfg_r)
    assert pipe.cache.computed == base, "render settings must not recompute any stage"
    Pipeline.render(res, "D", cfg_r.render)
    # mapping-only change recomputes C and D, not preprocessing / unary / A / B
    cfg_l = replace(cfg, mapping=replace(cfg.mapping, smoothness_lambda=3.0))
    pipe.run(src, PAL, cfg_l)
    after = pipe.cache.computed
    assert after["engine:C"] == base["engine:C"] + 1 and after["engine:D"] == base["engine:D"] + 1
    assert after["engine:A"] == base["engine:A"] and after["engine:B"] == base["engine:B"]
    assert after["preprocess"] == base["preprocess"] and after["unary"] == base["unary"]
    # preprocessing change recomputes everything downstream
    cfg_p = replace(cfg, preprocess=replace(cfg.preprocess, denoise_strength=30.0))
    pipe.run(src, PAL, cfg_p)
    assert pipe.cache.computed["preprocess"] == base["preprocess"] + 1


def test_palette_change_creates_new_mapping_and_leaves_old_result_intact():
    pipe, src, cfg = Pipeline(), synth_source(), small_cfg()
    r1 = pipe.run(src, PAL, cfg, engines=("B",))
    h1 = r1.engines["B"].label_hash
    sub = Palette.from_payload({"palette_id": "sub", "version": "1", "colors": PAL.to_payload()["colors"][:12]})
    r2 = pipe.run(src, sub, cfg, engines=("B",))
    assert r2.engines["B"].result.labels.max() < 12 and r2.palette.sha256 != r1.palette.sha256
    assert r1.engines["B"].label_hash == h1 and r1.palette is PAL


def test_image_normalisation():
    try:
        normalize_image(b"definitely not an image", "bad.png")
    except ImageIntegrityError:
        pass
    else:
        raise AssertionError("corrupt data must be rejected")
    rgba = np.zeros((20, 30, 4), np.uint8); rgba[:, :, :3] = 255; rgba[:, :, 3] = 0
    n = normalize_image(png_bytes(rgba, "RGBA"), "t.png")
    assert (n.rgb == 255).all() and n.meta["had_alpha"]                           # composited on white
    im = Image.fromarray(synth_rgb(h=40, w=20)); exif = im.getexif(); exif[274] = 6
    b = io.BytesIO(); im.save(b, "JPEG", exif=exif.tobytes())
    n = normalize_image(b.getvalue(), "t.jpg")
    assert n.rgb.shape[:2] == (20, 40)                                            # EXIF rotation applied
    assert n.meta["original_size"] == [20, 40]
