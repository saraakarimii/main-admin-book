"""Notebook-faithful Advanced Mapping Lab v2 engine.

The functions below are extracted from ChromaBook_Advanced_Mapping_Lab_v2.
Palette is injected from the project's immutable main-21.palette.json.
"""
from __future__ import annotations
import io, os, json, time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont
from scipy import ndimage
from skimage.color import rgb2lab, lab2rgb, deltaE_ciede2000
from skimage.metrics import structural_similarity
from skimage.segmentation import slic, find_boundaries

PALETTE = []
pil_original = None
source_filename = None
CURRENT_PIPELINE = None

def hex_to_rgb(value):
    s = str(value).strip().replace("#", "")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        raise ValueError(f"HEX نامعتبر: {value}")
    return tuple(int(s[i:i+2], 16) for i in (0, 2, 4))

def _palette_entries(payload):
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
        if any(k in payload for k in ("hex", "rgb", "color")):
            return [payload]
    raise ValueError("ساختار palette.json شناخته نشد؛ آرایه یا کلید colors/palette لازم است.")

def normalize_palette(payload):
    entries = _palette_entries(payload)
    out = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"رکورد رنگ {i+1} باید object باشد.")
        nested = entry.get("color") if isinstance(entry.get("color"), dict) else {}
        item = {**nested, **entry}
        hx = item.get("hex") or item.get("hexCode") or item.get("hex_code")
        rgbv = item.get("rgb") or item.get("RGB")
        if hx:
            rgb = hex_to_rgb(hx)
        elif isinstance(rgbv, (list, tuple)) and len(rgbv) >= 3:
            rgb = tuple(int(np.clip(v, 0, 255)) for v in rgbv[:3])
            hx = "#{:02X}{:02X}{:02X}".format(*rgb)
        elif isinstance(rgbv, dict):
            rgb = tuple(int(np.clip(rgbv.get(k, 0), 0, 255)) for k in ("r","g","b"))
            hx = "#{:02X}{:02X}{:02X}".format(*rgb)
        elif all(k in item for k in ("r","g","b")):
            rgb = tuple(int(np.clip(item[k], 0, 255)) for k in ("r","g","b"))
            hx = "#{:02X}{:02X}{:02X}".format(*rgb)
        else:
            raise ValueError(f"رنگ {i+1}: فیلد hex یا rgb وجود ندارد.")
        cid = item.get("id", item.get("code", item.get("number")))
        if cid is None or str(cid).strip() == "":
            raise ValueError(
                f"رنگ شماره {i+1}: شناسه چاپی الزامی است؛ "
                "شناسه را از ترتیب آرایه حدس نمی‌زنیم."
            )
        name = item.get("name", item.get("label", f"Color {i+1}"))
        out.append({"id":str(cid), "name":str(name), "hex":str(hx).upper(),
                    "rgb":tuple(int(v) for v in rgb)})
    if len(out) < 2:
        raise ValueError("پالت باید حداقل دو رنگ داشته باشد.")
    if len({x["id"] for x in out}) != len(out):
        raise ValueError("شناسه چاپی رنگ‌ها در پالت تکراری است.")
    if len(out) > 256:
        raise ValueError("برای خروجی uint8 حداکثر ۲۵۶ رنگ پشتیبانی می‌شود.")
    return out

def attach_palette_lab(palette):
    rgb = np.array([p["rgb"] for p in palette], dtype=np.float32) / 255.0
    labs = rgb2lab(rgb.reshape(1, -1, 3)).reshape(-1, 3).astype(np.float32)
    return [{**p, "lab":labs[i]} for i,p in enumerate(palette)]

def to_rgb_array(pil_img):
    rgba = pil_img.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255,255,255,255))
    return np.asarray(Image.alpha_composite(bg, rgba).convert("RGB"), dtype=np.uint8)

def crop_resize(rgb, p):
    h,w = rgb.shape[:2]
    if p["crop_enabled"]:
        x0 = int(np.clip(p["crop_x0"], 0, 98))
        y0 = int(np.clip(p["crop_y0"], 0, 98))
        x1 = int(np.clip(p["crop_x1"], x0+1, 100))
        y1 = int(np.clip(p["crop_y1"], y0+1, 100))
    else:
        x0,y0,x1,y1 = 0,0,100,100
    xa,ya = round(x0*w/100), round(y0*h/100)
    xb,yb = round(x1*w/100), round(y1*h/100)
    cropped = rgb[ya:max(ya+1,yb), xa:max(xa+1,xb)].copy()
    max_side = int(p["max_side"])
    hh,ww = cropped.shape[:2]
    scale = min(1.0, max_side/max(hh,ww)) if max_side > 0 else 1.0
    if scale < 1:
        cropped = cv2.resize(cropped, (max(16,round(ww*scale)), max(16,round(hh*scale))),
                              interpolation=cv2.INTER_AREA)
    return cropped

def illumination_correction(rgb, strength=0.0, sigma_fraction=0.10):
    if strength <= 0:
        return rgb.copy(), np.zeros(rgb.shape[:2], np.float32)
    lab = rgb2lab(rgb.astype(np.float32)/255.0).astype(np.float32)
    L = lab[:,:,0]
    sigma = max(3.0, min(rgb.shape[:2]) * float(sigma_fraction))
    background = cv2.GaussianBlur(L, (0,0), sigmaX=sigma, sigmaY=sigma)
    target = float(np.median(background))
    corrected = L - float(strength) * (background - target)
    lab[:,:,0] = np.clip(corrected, 0, 100)
    out = np.clip(lab2rgb(lab)*255.0, 0, 255).astype(np.uint8)
    shade = background - target
    lo,hi = np.percentile(shade,[2,98])
    shade_vis = np.clip((shade-lo)/(hi-lo+1e-6)*255,0,255).astype(np.uint8)
    return out, shade_vis

def clahe_luminance(rgb, clip_limit=0.0, tile_grid=8):
    if clip_limit <= 0:
        return rgb.copy()
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                            tileGridSize=(int(tile_grid),int(tile_grid)))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def denoise_rgb(rgb, method="bilateral", strength=0.0):
    if method == "none" or strength <= 0:
        return rgb.copy()
    if method == "bilateral":
        d = 5 if strength < 35 else 7
        return cv2.bilateralFilter(rgb, d, float(strength), max(1.0,float(strength)/8))
    if method == "nlm":
        h = float(strength)
        return cv2.fastNlMeansDenoisingColored(rgb, None, h, h, 7, 21)
    if method == "median":
        k = 3 if strength < 35 else 5
        return cv2.medianBlur(rgb, k)
    return rgb.copy()

def multiscale_sharpen(rgb, amount_fine=0.0, amount_coarse=0.0):
    if amount_fine <= 0 and amount_coarse <= 0:
        return rgb.copy()
    lab = rgb2lab(rgb.astype(np.float32)/255.0).astype(np.float32)
    L = lab[:,:,0]
    fine = L - cv2.GaussianBlur(L,(0,0),0.8)
    coarse = L - cv2.GaussianBlur(L,(0,0),2.0)
    lab[:,:,0] = np.clip(L + amount_fine*fine + amount_coarse*coarse, 0, 100)
    return np.clip(lab2rgb(lab)*255.0,0,255).astype(np.uint8)

def preprocess_pipeline(pil_img, p):
    raw = crop_resize(to_rgb_array(pil_img), p)
    illum, shade = illumination_correction(raw, p["illumination_strength"], p["illumination_sigma"])
    local = clahe_luminance(illum, p["clahe_clip"], p["clahe_tiles"])
    den = denoise_rgb(local, p["denoise_method"], p["denoise_strength"])
    sharp = multiscale_sharpen(den, p["sharpen_fine"], p["sharpen_coarse"])
    return {"reference":raw, "illumination_corrected":illum, "clahe":local,
            "denoised":den, "processed":sharp, "shade_map":shade}

def extract_edges(rgb, low=55, high=145, auto_canny=True, close_iter=1):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx,gy)
    scale = max(1e-6, float(np.percentile(mag, 99.5)))
    grad = np.clip(mag/scale,0,1)

    if auto_canny:
        med = float(np.median(gray))
        lo = int(max(0,(1.0-0.33)*med))
        hi = int(min(255,max(lo+1,(1.0+0.33)*med)))
    else:
        lo,hi = int(low),int(high)
        if hi <= lo: hi = min(255,lo+1)

    canny = cv2.Canny(gray,lo,hi,L2gradient=True)
    if close_iter > 0:
        kernel = np.ones((3,3),np.uint8)
        canny = cv2.morphologyEx(canny,cv2.MORPH_CLOSE,kernel,iterations=int(close_iter))
    canny_f = canny.astype(np.float32)/255.0
    combined = np.maximum(grad*0.75,canny_f)
    combined = cv2.GaussianBlur(combined,(0,0),0.65)
    return np.clip(combined,0,1).astype(np.float32), canny, grad

def grid_geometry(h,w,cols):
    cols = int(np.clip(cols,4,180))
    rows = max(4,int(round(cols*h/w)))
    xe = np.linspace(0,w,cols+1).round().astype(int)
    ye = np.linspace(0,h,rows+1).round().astype(int)
    return rows,cols,xe,ye

def lab_palette():
    return np.stack([p["lab"] for p in PALETTE]).astype(np.float32)

def delta_e_matrix(pixel_labs, palette_labs):
    # skimage broadcasts (N,1,3) against (1,K,3), returning N x K.
    return deltaE_ciede2000(pixel_labs[:,None,:],palette_labs[None,:,:]).astype(np.float32)

def sampled_pixels(arr, max_samples):
    flat = arr.reshape(-1,arr.shape[-1])
    if len(flat) <= max_samples:
        return flat
    idx = np.linspace(0,len(flat)-1,int(max_samples)).astype(np.int32)
    return flat[idx]

def trimmed_column_mean(costs, trim_fraction=0.0):
    frac = float(np.clip(trim_fraction,0,0.40))
    if frac <= 0 or costs.shape[0] < 5:
        return np.mean(costs,axis=0)
    ordered = np.sort(costs,axis=0)
    cut = min(int(len(ordered)*frac),max(0,(len(ordered)-1)//2))
    return np.mean(ordered[cut:len(ordered)-cut],axis=0) if cut else np.mean(costs,axis=0)

def build_cell_unary(processed_lab, cols, max_samples=96, trim_fraction=0.03):
    h,w = processed_lab.shape[:2]
    rows,cols,xe,ye = grid_geometry(h,w,cols)
    pal = lab_palette()
    unary = np.zeros((rows,cols,len(PALETTE)),np.float32)
    for r in range(rows):
        for c in range(cols):
            cell = processed_lab[ye[r]:ye[r+1],xe[c]:xe[c+1]]
            pix = sampled_pixels(cell,max_samples).astype(np.float32)
            costs = delta_e_matrix(pix,pal)
            unary[r,c] = trimmed_column_mean(costs,trim_fraction)
    return unary,(rows,cols,xe,ye)

def build_region_balanced_unary(processed_lab, slic_labels, geometry,
                                max_samples=72, trim_fraction=0.03):
    rows,cols,xe,ye = geometry
    pal = lab_palette()
    out = np.zeros((rows,cols,len(PALETTE)),np.float32)
    for r in range(rows):
        for c in range(cols):
            y0,y1,x0,x1 = ye[r],ye[r+1],xe[c],xe[c+1]
            cell_lab = processed_lab[y0:y1,x0:x1]
            cell_seg = slic_labels[y0:y1,x0:x1]
            ids,counts = np.unique(cell_seg,return_counts=True)
            region_costs,weights = [],[]
            for sid,count in zip(ids,counts):
                region = cell_lab[cell_seg==sid]
                pix = sampled_pixels(region,max(12,min(max_samples,int(max_samples*count/max(1,cell_seg.size)*2))))
                cm = trimmed_column_mean(delta_e_matrix(pix.astype(np.float32),pal),trim_fraction)
                region_costs.append(cm)
                weights.append(np.sqrt(float(count)))
            weights = np.asarray(weights,np.float32)
            weights /= max(1e-8,weights.sum())
            out[r,c] = np.sum(np.asarray(region_costs)*weights[:,None],axis=0)
    return out

def legacy_mean_lab_grid(processed_lab, geometry, return_cost=False):
    rows,cols,xe,ye = geometry
    pal = lab_palette()
    labels = np.zeros((rows,cols),np.int32)
    selected_cost = np.zeros((rows,cols),np.float32)
    for r in range(rows):
        for c in range(cols):
            cell = processed_lab[ye[r]:ye[r+1],xe[c]:xe[c+1]]
            mean_lab = np.mean(cell.reshape(-1,3),axis=0)
            d = np.linalg.norm(pal-mean_lab[None,:],axis=1) # legacy ΔE76
            labels[r,c] = int(np.argmin(d))
            selected_cost[r,c] = float(d[labels[r,c]])
    return (labels,selected_cost) if return_cost else labels

def boundary_pairwise_weights(boundary_map, geometry, beta=4.0):
    rows,cols,xe,ye = geometry
    h,w = boundary_map.shape
    wh = np.ones((rows,max(0,cols-1)),np.float32)
    wv = np.ones((max(0,rows-1),cols),np.float32)
    radius = 2
    for r in range(rows):
        y0,y1 = ye[r],ye[r+1]
        for c in range(cols-1):
            x = int(xe[c+1])
            xa,xb = max(0,x-radius),min(w,x+radius+1)
            score = float(np.mean(boundary_map[y0:y1,xa:xb])) if y1>y0 and xb>xa else 0.0
            wh[r,c] = np.exp(-float(beta)*np.clip(score,0,1))
    for r in range(rows-1):
        y = int(ye[r+1])
        for c in range(cols):
            x0,x1 = xe[c],xe[c+1]
            ya,yb = max(0,y-radius),min(h,y+radius+1)
            score = float(np.mean(boundary_map[ya:yb,x0:x1])) if yb>ya and x1>x0 else 0.0
            wv[r,c] = np.exp(-float(beta)*np.clip(score,0,1))
    return wh,wv

def icm_optimize(unary, wh, wv, lam=1.8, iterations=5):
    rows,cols,k = unary.shape
    labels = np.argmin(unary,axis=2).astype(np.int32)
    label_ids = np.arange(k)
    for _ in range(int(iterations)):
        changes = 0
        # Raster scan coordinate descent; deterministic and easy to reproduce.
        for r in range(rows):
            for c in range(cols):
                costs = unary[r,c].copy()
                if c > 0:
                    costs += lam*wh[r,c-1]*(label_ids != labels[r,c-1])
                if c < cols-1:
                    costs += lam*wh[r,c]*(label_ids != labels[r,c+1])
                if r > 0:
                    costs += lam*wv[r-1,c]*(label_ids != labels[r-1,c])
                if r < rows-1:
                    costs += lam*wv[r,c]*(label_ids != labels[r+1,c])
                new = int(np.argmin(costs))
                if new != labels[r,c]:
                    labels[r,c] = new
                    changes += 1
        if changes == 0:
            break
    return labels

def make_slic(processed_rgb, n_segments=320, compactness=10.0):
    labels = slic(
        processed_rgb.astype(np.float32)/255.0,
        n_segments=int(n_segments),
        compactness=float(compactness),
        sigma=0,
        start_label=0,
        convert2lab=True,
        enforce_connectivity=True,
        channel_axis=-1,
    ).astype(np.int32)
    b = find_boundaries(labels,mode="thick").astype(np.uint8)
    b = cv2.dilate(b,np.ones((3,3),np.uint8),iterations=1)
    b = cv2.GaussianBlur(b.astype(np.float32),(0,0),0.8)
    if b.max() > 0: b /= b.max()
    return labels,b

def render_pixel_preview(labels, geometry):
    rows,cols,xe,ye = geometry
    h,w = int(ye[-1]),int(xe[-1])
    out = np.full((h,w,3),255,np.uint8)
    for r in range(rows):
        for c in range(cols):
            out[ye[r]:ye[r+1],xe[c]:xe[c+1]] = PALETTE[int(labels[r,c])]["rgb"]
    return out

def render_numbered_grid(labels, cell_px=44, line_width=2):
    rows,cols = labels.shape
    cp = int(np.clip(cell_px,20,100))
    canvas = Image.new("RGB",(cols*cp+1,rows*cp+1),"white")
    draw = ImageDraw.Draw(canvas)
    font = None
    for fp in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp,max(9,int(cp*0.34)))
                break
            except Exception:
                pass
    if font is None: font = ImageFont.load_default()
    for r in range(rows):
        for c in range(cols):
            x0,y0 = c*cp,r*cp
            x1,y1 = x0+cp,y0+cp
            draw.rectangle([x0,y0,x1,y1],fill="white",outline=(15,15,15),width=int(line_width))
            text = str(PALETTE[int(labels[r,c])]["id"])
            box = draw.textbbox((0,0),text,font=font)
            tw,th = box[2]-box[0],box[3]-box[1]
            draw.text((x0+(cp-tw)/2-box[0],y0+(cp-th)/2-box[1]),
                      text,fill=(10,10,10),font=font)
    return np.asarray(canvas)

def grid_metrics(labels):
    total = int(labels.size)
    singleton = 0
    tiny = 0
    structure = np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)
    for color_idx in np.unique(labels):
        cc,n = ndimage.label(labels==color_idx,structure=structure)
        if n:
            sizes = np.bincount(cc.ravel())[1:]
            singleton += int(np.sum(sizes==1))
            tiny += int(np.sum(sizes<=2))
    htrans = int(np.sum(labels[:,1:] != labels[:,:-1]))
    vtrans = int(np.sum(labels[1:,:] != labels[:-1,:]))
    adj = labels.shape[0]*max(0,labels.shape[1]-1)+max(0,labels.shape[0]-1)*labels.shape[1]
    return {
        "singleton_ratio_pct":100.0*singleton/max(1,total),
        "tiny_component_ratio_pct":100.0*tiny/max(1,total),
        "transition_rate_pct":100.0*(htrans+vtrans)/max(1,adj),
    }

def edge_f1(reference_rgb, mapped_rgb):
    a = cv2.Canny(cv2.cvtColor(reference_rgb,cv2.COLOR_RGB2GRAY),55,150)
    b = cv2.Canny(cv2.cvtColor(mapped_rgb,cv2.COLOR_RGB2GRAY),55,150)
    kernel = np.ones((3,3),np.uint8)
    ad = cv2.dilate(a,kernel,iterations=1)>0
    bd = cv2.dilate(b,kernel,iterations=1)>0
    ab,bb = a>0,b>0
    precision = float(np.sum(bb & ad)/max(1,np.sum(bb)))
    recall = float(np.sum(ab & bd)/max(1,np.sum(ab)))
    f1 = 2*precision*recall/max(1e-12,precision+recall)
    return f1,precision,recall

def evaluate_grid(reference_rgb, processed_rgb, mapped_rgb, labels):
    ref_lab = rgb2lab(reference_rgb.astype(np.float32)/255.0)
    proc_lab = rgb2lab(processed_rgb.astype(np.float32)/255.0)
    map_lab = rgb2lab(mapped_rgb.astype(np.float32)/255.0)
    de_ref = deltaE_ciede2000(ref_lab,map_lab)
    de_proc = deltaE_ciede2000(proc_lab,map_lab)
    try:
        ssim = float(structural_similarity(reference_rgb,mapped_rgb,
                                           channel_axis=-1,data_range=255,win_size=7))
    except Exception:
        ssim = float("nan")
    f1,prec,rec = edge_f1(reference_rgb,mapped_rgb)
    used = len(np.unique(labels))
    m = {
        "DeltaE00_mean_vs_source":float(np.mean(de_ref)),
        "DeltaE00_median_vs_source":float(np.median(de_ref)),
        "DeltaE00_p95_vs_source":float(np.percentile(de_ref,95)),
        "DeltaE00_mean_vs_processed":float(np.mean(de_proc)),
        "SSIM_vs_source":ssim,
        "Edge_F1_1px":f1,
        "Edge_precision":prec,
        "Edge_recall":rec,
        "colors_used":int(used),
        **grid_metrics(labels),
    }
    return m,de_ref

def delta_map_rgb(de):
    vmax = max(1.0,float(np.percentile(de,99)))
    x = np.clip(de/vmax*255,0,255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(x,cv2.COLORMAP_INFERNO),cv2.COLOR_BGR2RGB)

def palette_usage(labels):
    u,c = np.unique(labels,return_counts=True)
    return pd.DataFrame([
        {"Color ID":PALETTE[int(i)]["id"],"Name":PALETTE[int(i)]["name"],
         "HEX":PALETTE[int(i)]["hex"],"Cells":int(n),
         "Share %":100.0*float(n)/labels.size}
        for i,n in zip(u,c)
    ]).sort_values("Cells",ascending=False).reset_index(drop=True)

ENGINE_NAMES = {
    "A":"A — Legacy Mean-Lab / ΔE76",
    "B":"B — Cell-wise CIEDE2000",
    "C":"C — Edge-Aware ICM",
    "D":"D — SLIC Region-Aware ICM",
}

def map_one_engine(key, processed_lab, geometry, unary, edge_map,
                   slic_labels=None, slic_boundary=None, p=None):
    rows,cols,xe,ye = geometry
    if key == "A":
        labels,selected_cost = legacy_mean_lab_grid(processed_lab,geometry,return_cost=True)
    elif key == "B":
        labels = np.argmin(unary,axis=2).astype(np.int32)
        selected_cost = np.take_along_axis(unary,labels[:,:,None],axis=2)[:,:,0]
    elif key == "C":
        wh,wv = boundary_pairwise_weights(edge_map,geometry,p["boundary_beta"])
        labels = icm_optimize(unary,wh,wv,p["smoothness_lambda"],p["icm_iterations"])
        selected_cost = np.take_along_axis(unary,labels[:,:,None],axis=2)[:,:,0]
    elif key == "D":
        if slic_labels is None:
            raise RuntimeError("SLIC labels برای موتور D ساخته نشده‌اند.")
        region_unary = build_region_balanced_unary(
            processed_lab,slic_labels,geometry,p["max_samples"],p["trim_fraction"])
        blend = float(p["region_balance"])
        d_unary = (1.0-blend)*unary + blend*region_unary
        combined = np.maximum(edge_map,0.95*slic_boundary)
        wh,wv = boundary_pairwise_weights(combined,geometry,p["boundary_beta"])
        labels = icm_optimize(d_unary,wh,wv,p["smoothness_lambda"],p["icm_iterations"])
        selected_cost = np.take_along_axis(d_unary,labels[:,:,None],axis=2)[:,:,0]
    else:
        raise ValueError(key)
    mapped = render_pixel_preview(labels,geometry)
    numbered = render_numbered_grid(labels,p["numbered_cell_px"],p["numbered_line_width"])
    metrics,de = evaluate_grid(CURRENT_PIPELINE["reference"],CURRENT_PIPELINE["processed"],mapped,labels)
    return {
        "key":key,"name":ENGINE_NAMES[key],"labels":labels,"pixel":mapped,
        "numbered":numbered,"metrics":metrics,"delta_map":de,
        "selected_cell_cost":selected_cost,
    }

def run_all_engines(p):
    global CURRENT_PIPELINE
    if pil_original is None:
        raise RuntimeError("ابتدا تصویر را بارگذاری کن.")
    t0 = time.time()
    pipe = preprocess_pipeline(pil_original,p)
    CURRENT_PIPELINE = pipe
    processed = pipe["processed"]
    processed_lab = rgb2lab(processed.astype(np.float32)/255.0).astype(np.float32)
    edge_map,canny,gradient = extract_edges(
        processed,p["canny_low"],p["canny_high"],p["auto_canny"],p["edge_close"])
    unary,geometry = build_cell_unary(
        processed_lab,p["grid_cols"],p["max_samples"],p["trim_fraction"])
    slic_labels,slic_boundary = make_slic(processed,p["slic_segments"],p["slic_compactness"])
    results = {}
    for key in ("A","B","C","D"):
        start = time.time()
        results[key] = map_one_engine(key,processed_lab,geometry,unary,edge_map,
                                      slic_labels,slic_boundary,p)
        results[key]["runtime_seconds"] = time.time()-start
        results[key]["metrics"]["runtime_seconds"] = results[key]["runtime_seconds"]
    pipe["edge_map"] = edge_map
    pipe["canny"] = canny
    pipe["gradient"] = gradient
    pipe["slic_labels"] = slic_labels
    pipe["slic_boundary"] = slic_boundary
    pipe["geometry"] = geometry
    pipe["unary"] = unary
    pipe["processed_lab"] = processed_lab
    return {"pipeline":pipe,"engines":results,"params":dict(p),
            "total_runtime_seconds":time.time()-t0}

def run_stability_test(engine_key, run_data):
    pipe = run_data["pipeline"]
    p = run_data["params"]
    base = run_data["engines"][engine_key]["labels"]
    lab0 = pipe["processed_lab"]
    geometry = pipe["geometry"]
    edge_map = pipe["edge_map"]
    slic_labels = pipe["slic_labels"]
    slic_boundary = pipe["slic_boundary"]
    agreements = []
    for delta_L in (-1.0,1.0):
        pert = lab0.copy()
        pert[:,:,0] = np.clip(pert[:,:,0]+delta_L,0,100)
        unary,_ = build_cell_unary(pert,p["grid_cols"],p["max_samples"],p["trim_fraction"])
        out = map_one_engine(engine_key,pert,geometry,unary,edge_map,
                             slic_labels,slic_boundary,p)
        agreements.append(float(np.mean(out["labels"]==base))*100.0)
    return {
        "Engine":ENGINE_NAMES[engine_key],
        "Agreement after L* -1":agreements[0],
        "Agreement after L* +1":agreements[1],
        "Mean stability %":float(np.mean(agreements)),
    }

# Exact defaults and control bounds from Notebook v2 widget declarations.
DEFAULT_PARAMS = {
    "max_side": 900,
    "crop_enabled": False, "crop_x0": 0, "crop_y0": 0, "crop_x1": 100, "crop_y1": 100,
    "illumination_strength": 0.25, "illumination_sigma": 0.10,
    "clahe_clip": 1.5, "clahe_tiles": 8,
    "denoise_method": "bilateral", "denoise_strength": 18.0,
    "sharpen_fine": 0.15, "sharpen_coarse": 0.05,
    "auto_canny": True, "canny_low": 55, "canny_high": 145, "edge_close": 1,
    "grid_cols": 40, "max_samples": 96, "trim_fraction": 0.03,
    "smoothness_lambda": 1.8, "boundary_beta": 4.0, "icm_iterations": 5,
    "slic_segments": 320, "slic_compactness": 10.0, "region_balance": 0.25,
    "numbered_cell_px": 44, "numbered_line_width": 2,
}

PARAMETER_BOUNDS = {
    "max_side": (256, 1800), "crop_x0": (0,98), "crop_y0": (0,98),
    "crop_x1": (2,100), "crop_y1": (2,100),
    "illumination_strength": (0,1), "illumination_sigma": (0.02,0.25),
    "clahe_clip": (0,5), "clahe_tiles": (4,16),
    "denoise_strength": (0,60), "sharpen_fine": (0,1.5), "sharpen_coarse": (0,1),
    "canny_low": (0,254), "canny_high": (1,255), "edge_close": (0,3),
    "grid_cols": (8,120), "max_samples": (16,256), "trim_fraction": (0,0.30),
    "smoothness_lambda": (0,8), "boundary_beta": (0,10), "icm_iterations": (1,12),
    "slic_segments": (80,900), "slic_compactness": (1,30), "region_balance": (0,0.8),
    "numbered_cell_px": (20,90), "numbered_line_width": (1,5),
}
BOOL_PARAMS = {"crop_enabled","auto_canny"}
INT_PARAMS = {"max_side","crop_x0","crop_y0","crop_x1","crop_y1","clahe_tiles",
              "canny_low","canny_high","edge_close","grid_cols","max_samples",
              "icm_iterations","slic_segments","numbered_cell_px","numbered_line_width"}
ALLOWED_DENOISE = {"none","bilateral","nlm","median"}

def validate_params(params=None):
    p = {**DEFAULT_PARAMS, **(params or {})}
    unknown = set(p) - set(DEFAULT_PARAMS)
    if unknown:
        raise ValueError(f"Unknown Notebook v2 parameters: {sorted(unknown)}")
    for key in BOOL_PARAMS:
        if not isinstance(p[key], bool):
            raise ValueError(f"{key} must be boolean")
    if p["denoise_method"] not in ALLOWED_DENOISE:
        raise ValueError("denoise_method must be none, bilateral, nlm or median")
    for key,(lo,hi) in PARAMETER_BOUNDS.items():
        val = float(p[key])
        if not np.isfinite(val) or val < lo or val > hi:
            raise ValueError(f"{key} must be between {lo} and {hi}")
        if key in INT_PARAMS and int(val) != val:
            raise ValueError(f"{key} must be an integer")
        p[key] = int(val) if key in INT_PARAMS else float(val)
    if p["crop_enabled"] and (p["crop_x1"] <= p["crop_x0"] or p["crop_y1"] <= p["crop_y0"]):
        raise ValueError("Crop right/bottom must be greater than crop left/top")
    return p

def run_image(image_path, palette_path, params=None, selected_engine="D", output_dir=None, include_stability=False):
    global PALETTE, pil_original, source_filename
    selected_engine = str(selected_engine).upper()
    if selected_engine not in ENGINE_NAMES:
        raise ValueError("selected_engine must be A, B, C or D")
    payload = json.loads(Path(palette_path).read_text(encoding="utf-8-sig"))
    PALETTE = attach_palette_lab(normalize_palette(payload))
    if len(PALETTE) != 21:
        raise ValueError(f"ChromaBook palette is locked to exactly 21 colours; found {len(PALETTE)}")
    ids = [x["id"] for x in PALETTE]
    if len(ids) != len(set(ids)):
        raise ValueError("Palette IDs must be unique")
    with Image.open(image_path) as im:
        pil_original = ImageOps.exif_transpose(im).convert("RGBA")
    source_filename = Path(image_path).name
    p = validate_params(params)
    run_data = run_all_engines(p)
    if output_dir is not None:
        save_run_outputs(run_data, Path(output_dir), selected_engine, include_stability=include_stability)
    return run_data

def save_run_outputs(run_data, out_dir, selected_engine="D", include_stability=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = run_data["engines"][selected_engine]
    np.save(out_dir/"labels.npy", chosen["labels"].astype(np.uint8))
    Image.fromarray(chosen["pixel"].astype(np.uint8)).save(out_dir/"color_preview.png")
    Image.fromarray(chosen["numbered"].astype(np.uint8)).save(out_dir/"numbered_preview.png")
    pipe = run_data["pipeline"]
    for fname,arr in [
        ("00_source_cropped.png",pipe["reference"]),
        ("01_illumination_corrected.png",pipe["illumination_corrected"]),
        ("02_clahe.png",pipe["clahe"]),
        ("03_preprocessed.png",pipe["processed"]),
        ("04_edge_strength.png",(pipe["edge_map"]*255).astype(np.uint8)),
        ("05_canny.png",pipe["canny"]),
        ("06_slic_boundaries.png",(pipe["slic_boundary"]*255).astype(np.uint8)),
    ]:
        Image.fromarray(np.asarray(arr).astype(np.uint8)).save(out_dir/fname)
    comparison = {}
    for key,res in run_data["engines"].items():
        folder = out_dir/f"engine_{key}"
        folder.mkdir(exist_ok=True)
        Image.fromarray(res["pixel"].astype(np.uint8)).save(folder/"color_preview.png")
        Image.fromarray(res["numbered"].astype(np.uint8)).save(folder/"numbered_preview.png")
        np.save(folder/"labels.npy",res["labels"].astype(np.uint8))
        (folder/"metrics.json").write_text(json.dumps(res["metrics"],ensure_ascii=False,indent=2),encoding="utf-8")
        comparison[key] = res["metrics"]
    metrics_payload = {
        "selected_engine":selected_engine,
        "total_runtime_seconds":run_data["total_runtime_seconds"],
        "engines":comparison,
        "selected_metrics":chosen["metrics"],
        "palette_colors":len(PALETTE),
    }
    (out_dir/"engine_comparison_metrics.json").write_text(json.dumps(metrics_payload,ensure_ascii=False,indent=2),encoding="utf-8")
    if include_stability:
        stability = [run_stability_test(k, run_data) for k in ("B","C","D")]
        (out_dir/"stability_metrics.json").write_text(
            json.dumps(stability, ensure_ascii=False, indent=2), encoding="utf-8")
    params_payload = {"schema":"chromabook-advanced-mapping-v2","params":run_data["params"],
                      "selected_engine":selected_engine, "run_stability":bool(include_stability),
                      "palette_ids":[x["id"] for x in PALETTE]}
    (out_dir/"config.json").write_text(json.dumps(params_payload,ensure_ascii=False,indent=2),encoding="utf-8")
    summary = {
        "name":source_filename, "grid":[int(chosen["labels"].shape[1]),int(chosen["labels"].shape[0])],
        "colors_used":int(chosen["metrics"]["colors_used"]), "engine":selected_engine,
        "confidence":"not_computed", "elapsed_s":round(run_data["total_runtime_seconds"],2),
        "timings":{"total":run_data["total_runtime_seconds"]},
        "recommendation":{"engine":selected_engine,"confidence":{"level":"not_computed"},
                          "metrics":chosen["metrics"]},
    }
    (out_dir/"asset_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    render_preflight = {
        "passed": bool(chosen["numbered"].ndim == 3 and chosen["numbered"].shape[2] == 3),
        "grid_rows":int(chosen["labels"].shape[0]),"grid_cols":int(chosen["labels"].shape[1]),
        "palette_colors":len(PALETTE),"colors_used":int(chosen["metrics"]["colors_used"]),
        "numbered_cell_px":run_data["params"]["numbered_cell_px"],
        "numbered_line_width":run_data["params"]["numbered_line_width"],
    }
    (out_dir/"render_preflight.json").write_text(json.dumps(render_preflight,indent=2),encoding="utf-8")


PARAMETER_UI = [
    {'group': 'Preprocess', 'key': 'max_side', 'label': 'Max analysis side', 'kind': 'range', 'min': 256, 'max': 1800, 'step': 32},
    {'group': 'Preprocess', 'key': 'crop_enabled', 'label': 'Enable crop', 'kind': 'bool', 'min': None, 'max': None, 'step': None},
    {'group': 'Preprocess', 'key': 'crop_x0', 'label': 'Crop left %', 'kind': 'range', 'min': 0, 'max': 98, 'step': 1},
    {'group': 'Preprocess', 'key': 'crop_y0', 'label': 'Crop top %', 'kind': 'range', 'min': 0, 'max': 98, 'step': 1},
    {'group': 'Preprocess', 'key': 'crop_x1', 'label': 'Crop right %', 'kind': 'range', 'min': 2, 'max': 100, 'step': 1},
    {'group': 'Preprocess', 'key': 'crop_y1', 'label': 'Crop bottom %', 'kind': 'range', 'min': 2, 'max': 100, 'step': 1},
    {'group': 'Preprocess', 'key': 'illumination_strength', 'label': 'Illumination correction', 'kind': 'range', 'min': 0, 'max': 1, 'step': 0.025},
    {'group': 'Preprocess', 'key': 'illumination_sigma', 'label': 'Illumination scale', 'kind': 'range', 'min': 0.02, 'max': 0.25, 'step': 0.01},
    {'group': 'Preprocess', 'key': 'clahe_clip', 'label': 'CLAHE clip limit', 'kind': 'range', 'min': 0, 'max': 5, 'step': 0.1},
    {'group': 'Preprocess', 'key': 'clahe_tiles', 'label': 'CLAHE tile grid', 'kind': 'range', 'min': 4, 'max': 16, 'step': 1},
    {'group': 'Preprocess', 'key': 'denoise_method', 'label': 'Denoise method', 'kind': 'select', 'min': None, 'max': None, 'step': None},
    {'group': 'Preprocess', 'key': 'denoise_strength', 'label': 'Denoise strength', 'kind': 'range', 'min': 0, 'max': 60, 'step': 1},
    {'group': 'Preprocess', 'key': 'sharpen_fine', 'label': 'Fine sharpen', 'kind': 'range', 'min': 0, 'max': 1.5, 'step': 0.025},
    {'group': 'Preprocess', 'key': 'sharpen_coarse', 'label': 'Coarse sharpen', 'kind': 'range', 'min': 0, 'max': 1, 'step': 0.025},
    {'group': 'Edges', 'key': 'auto_canny', 'label': 'Auto Canny thresholds', 'kind': 'bool', 'min': None, 'max': None, 'step': None},
    {'group': 'Edges', 'key': 'canny_low', 'label': 'Canny low', 'kind': 'range', 'min': 0, 'max': 254, 'step': 1},
    {'group': 'Edges', 'key': 'canny_high', 'label': 'Canny high', 'kind': 'range', 'min': 1, 'max': 255, 'step': 1},
    {'group': 'Edges', 'key': 'edge_close', 'label': 'Close iterations', 'kind': 'range', 'min': 0, 'max': 3, 'step': 1},
    {'group': 'Mapping', 'key': 'grid_cols', 'label': 'Grid columns', 'kind': 'range', 'min': 8, 'max': 120, 'step': 1},
    {'group': 'Mapping', 'key': 'max_samples', 'label': 'Samples / cell', 'kind': 'range', 'min': 16, 'max': 256, 'step': 8},
    {'group': 'Mapping', 'key': 'trim_fraction', 'label': 'Trim outliers', 'kind': 'range', 'min': 0, 'max': 0.3, 'step': 0.01},
    {'group': 'Mapping', 'key': 'smoothness_lambda', 'label': 'Smoothness λ', 'kind': 'range', 'min': 0, 'max': 8, 'step': 0.1},
    {'group': 'Mapping', 'key': 'boundary_beta', 'label': 'Boundary sensitivity β', 'kind': 'range', 'min': 0, 'max': 10, 'step': 0.25},
    {'group': 'Mapping', 'key': 'icm_iterations', 'label': 'ICM iterations', 'kind': 'range', 'min': 1, 'max': 12, 'step': 1},
    {'group': 'Mapping', 'key': 'slic_segments', 'label': 'SLIC segments', 'kind': 'range', 'min': 80, 'max': 900, 'step': 20},
    {'group': 'Mapping', 'key': 'slic_compactness', 'label': 'SLIC compactness', 'kind': 'range', 'min': 1, 'max': 30, 'step': 0.5},
    {'group': 'Mapping', 'key': 'region_balance', 'label': 'Region balance', 'kind': 'range', 'min': 0, 'max': 0.8, 'step': 0.025},
    {'group': 'Mapping', 'key': 'numbered_cell_px', 'label': 'Number cell px', 'kind': 'range', 'min': 20, 'max': 90, 'step': 2},
    {'group': 'Mapping', 'key': 'numbered_line_width', 'label': 'Grid line width', 'kind': 'range', 'min': 1, 'max': 5, 'step': 1},
]
