import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
import { PALETTE_PATH } from './store.js';

export interface PaletteColor {
  id: string;
  name: string;
  hex: string;
  rgb: [number, number, number];
  lab: [number, number, number];
}

export interface MappingParams {
  max_side: number;
  crop_enabled: boolean;
  crop_x0: number;
  crop_y0: number;
  crop_x1: number;
  crop_y1: number;
  illumination_strength: number;
  illumination_sigma: number;
  clahe_clip: number;
  clahe_tiles: number;
  denoise_method: 'none' | 'bilateral' | 'nlm' | 'median';
  denoise_strength: number;
  sharpen_fine: number;
  sharpen_coarse: number;
  auto_canny: boolean;
  canny_low: number;
  canny_high: number;
  edge_close: number;
  grid_cols: number;
  max_samples: number;
  trim_fraction: number;
  smoothness_lambda: number;
  boundary_beta: number;
  icm_iterations: number;
  slic_segments: number;
  slic_compactness: number;
  region_balance: number;
  numbered_cell_px: number;
  numbered_line_width: number;
}

export const DEFAULT_PARAMS: MappingParams = {
  max_side: 900,
  crop_enabled: false,
  crop_x0: 0,
  crop_y0: 0,
  crop_x1: 100,
  crop_y1: 100,
  illumination_strength: 0.25,
  illumination_sigma: 0.10,
  clahe_clip: 1.5,
  clahe_tiles: 8,
  denoise_method: 'bilateral',
  denoise_strength: 18.0,
  sharpen_fine: 0.15,
  sharpen_coarse: 0.05,
  auto_canny: true,
  canny_low: 55,
  canny_high: 145,
  edge_close: 1,
  grid_cols: 40,
  max_samples: 96,
  trim_fraction: 0.03,
  smoothness_lambda: 1.8,
  boundary_beta: 4.0,
  icm_iterations: 5,
  slic_segments: 320,
  slic_compactness: 10.0,
  region_balance: 0.25,
  numbered_cell_px: 44,
  numbered_line_width: 2,
};

export const PARAMETER_BOUNDS: Record<string, [number, number]> = {
  max_side: [256, 1800],
  crop_x0: [0, 98],
  crop_y0: [0, 98],
  crop_x1: [2, 100],
  crop_y1: [2, 100],
  illumination_strength: [0, 1],
  illumination_sigma: [0.02, 0.25],
  clahe_clip: [0, 5],
  clahe_tiles: [4, 16],
  denoise_strength: [0, 60],
  sharpen_fine: [0, 1.5],
  sharpen_coarse: [0, 1],
  canny_low: [0, 254],
  canny_high: [1, 255],
  edge_close: [0, 3],
  grid_cols: [8, 120],
  max_samples: [16, 256],
  trim_fraction: [0, 0.30],
  smoothness_lambda: [0, 8],
  boundary_beta: [0, 10],
  icm_iterations: [1, 12],
  slic_segments: [80, 900],
  slic_compactness: [1, 30],
  region_balance: [0, 0.8],
  numbered_cell_px: [20, 90],
  numbered_line_width: [1, 5],
};

export const BOOL_PARAMS = new Set(['crop_enabled', 'auto_canny']);
export const INT_PARAMS = new Set([
  'max_side', 'crop_x0', 'crop_y0', 'crop_x1', 'crop_y1', 'clahe_tiles',
  'canny_low', 'canny_high', 'edge_close', 'grid_cols', 'max_samples',
  'icm_iterations', 'slic_segments', 'numbered_cell_px', 'numbered_line_width',
]);
export const ALLOWED_DENOISE = new Set(['none', 'bilateral', 'nlm', 'median']);

export const ENGINE_NAMES: Record<string, string> = {
  A: 'A — Legacy Mean-Lab / ΔE76',
  B: 'B — Cell-wise CIEDE2000',
  C: 'C — Edge-Aware ICM',
  D: 'D — SLIC Region-Aware ICM',
};

export const PARAMETER_UI = [
  { group: 'Preprocess', key: 'max_side', label: 'Max analysis side', kind: 'range', min: 256, max: 1800, step: 32 },
  { group: 'Preprocess', key: 'crop_enabled', label: 'Enable crop', kind: 'bool', min: null, max: null, step: null },
  { group: 'Preprocess', key: 'crop_x0', label: 'Crop left %', kind: 'range', min: 0, max: 98, step: 1 },
  { group: 'Preprocess', key: 'crop_y0', label: 'Crop top %', kind: 'range', min: 0, max: 98, step: 1 },
  { group: 'Preprocess', key: 'crop_x1', label: 'Crop right %', kind: 'range', min: 2, max: 100, step: 1 },
  { group: 'Preprocess', key: 'crop_y1', label: 'Crop bottom %', kind: 'range', min: 2, max: 100, step: 1 },
  { group: 'Preprocess', key: 'illumination_strength', label: 'Illumination correction', kind: 'range', min: 0, max: 1, step: 0.025 },
  { group: 'Preprocess', key: 'illumination_sigma', label: 'Illumination scale', kind: 'range', min: 0.02, max: 0.25, step: 0.01 },
  { group: 'Preprocess', key: 'clahe_clip', label: 'CLAHE clip limit', kind: 'range', min: 0, max: 5, step: 0.1 },
  { group: 'Preprocess', key: 'clahe_tiles', label: 'CLAHE tile grid', kind: 'range', min: 4, max: 16, step: 1 },
  { group: 'Preprocess', key: 'denoise_method', label: 'Denoise method', kind: 'select', min: null, max: null, step: null },
  { group: 'Preprocess', key: 'denoise_strength', label: 'Denoise strength', kind: 'range', min: 0, max: 60, step: 1 },
  { group: 'Preprocess', key: 'sharpen_fine', label: 'Fine sharpen', kind: 'range', min: 0, max: 1.5, step: 0.025 },
  { group: 'Preprocess', key: 'sharpen_coarse', label: 'Coarse sharpen', kind: 'range', min: 0, max: 1, step: 0.025 },
  { group: 'Edges', key: 'auto_canny', label: 'Auto Canny thresholds', kind: 'bool', min: null, max: null, step: null },
  { group: 'Edges', key: 'canny_low', label: 'Canny low', kind: 'range', min: 0, max: 254, step: 1 },
  { group: 'Edges', key: 'canny_high', label: 'Canny high', kind: 'range', min: 1, max: 255, step: 1 },
  { group: 'Edges', key: 'edge_close', label: 'Close iterations', kind: 'range', min: 0, max: 3, step: 1 },
  { group: 'Mapping', key: 'grid_cols', label: 'Grid columns', kind: 'range', min: 8, max: 120, step: 1 },
  { group: 'Mapping', key: 'max_samples', label: 'Samples / cell', kind: 'range', min: 16, max: 256, step: 8 },
  { group: 'Mapping', key: 'trim_fraction', label: 'Trim outliers', kind: 'range', min: 0, max: 0.3, step: 0.01 },
  { group: 'Mapping', key: 'smoothness_lambda', label: 'Smoothness λ', kind: 'range', min: 0, max: 8, step: 0.1 },
  { group: 'Mapping', key: 'boundary_beta', label: 'Boundary sensitivity β', kind: 'range', min: 0, max: 10, step: 0.25 },
  { group: 'Mapping', key: 'icm_iterations', label: 'ICM iterations', kind: 'range', min: 1, max: 12, step: 1 },
  { group: 'Mapping', key: 'slic_segments', label: 'SLIC segments', kind: 'range', min: 80, max: 900, step: 20 },
  { group: 'Mapping', key: 'slic_compactness', label: 'SLIC compactness', kind: 'range', min: 1, max: 30, step: 0.5 },
  { group: 'Mapping', key: 'region_balance', label: 'Region balance', kind: 'range', min: 0, max: 0.8, step: 0.025 },
  { group: 'Mapping', key: 'numbered_cell_px', label: 'Number cell px', kind: 'range', min: 20, max: 90, step: 2 },
  { group: 'Mapping', key: 'numbered_line_width', label: 'Grid line width', kind: 'range', min: 1, max: 5, step: 1 },
];

export function validateParams(params?: Record<string, unknown> | null): MappingParams {
  const merged: Record<string, unknown> = { ...DEFAULT_PARAMS, ...(params || {}) };
  const allowed = new Set(Object.keys(DEFAULT_PARAMS));
  const unknown = Object.keys(merged).filter((k) => !allowed.has(k));
  if (unknown.length > 0) {
    throw new Error(`Unknown Notebook v2 parameters: ${unknown.sort().join(', ')}`);
  }
  for (const key of BOOL_PARAMS) {
    if (typeof merged[key] !== 'boolean') {
      throw new Error(`${key} must be boolean`);
    }
  }
  if (typeof merged.denoise_method !== 'string' || !ALLOWED_DENOISE.has(merged.denoise_method)) {
    throw new Error('denoise_method must be none, bilateral, nlm or median');
  }
  for (const [key, [lo, hi]] of Object.entries(PARAMETER_BOUNDS)) {
    const val = Number(merged[key]);
    if (!Number.isFinite(val) || val < lo || val > hi) {
      throw new Error(`${key} must be between ${lo} and ${hi}`);
    }
    if (INT_PARAMS.has(key) && Math.trunc(val) !== val) {
      throw new Error(`${key} must be an integer`);
    }
    merged[key] = INT_PARAMS.has(key) ? Math.trunc(val) : val;
  }
  const p = merged as unknown as MappingParams;
  if (p.crop_enabled && (p.crop_x1 <= p.crop_x0 || p.crop_y1 <= p.crop_y0)) {
    throw new Error('Crop right/bottom must be greater than crop left/top');
  }
  return p;
}

// --- Color Math (sRGB <-> CIELAB D65 & CIEDE2000) ---

function srgbToLinear(c: number): number {
  const v = c / 255;
  return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
}

function linearToSrgb(v: number): number {
  const clamped = Math.max(0, Math.min(1, v));
  const s = clamped <= 0.0031308 ? 12.92 * clamped : 1.055 * Math.pow(clamped, 1 / 2.4) - 0.055;
  return Math.round(Math.max(0, Math.min(255, s * 255)));
}

const D65_XN = 0.95047;
const D65_YN = 1.0;
const D65_ZN = 1.08883;

function labF(t: number): number {
  return t > 0.008856451679035631 ? Math.cbrt(t) : 7.787037037037037 * t + 16 / 116;
}

function labFInv(t: number): number {
  return t > 0.20689655172413793 ? t * t * t : (t - 16 / 116) / 7.787037037037037;
}

export function rgbToLabTuple(r: number, g: number, b: number): [number, number, number] {
  const lr = srgbToLinear(r);
  const lg = srgbToLinear(g);
  const lb = srgbToLinear(b);
  const x = (0.4124564 * lr + 0.3575761 * lg + 0.1804375 * lb) / D65_XN;
  const y = (0.2126729 * lr + 0.7151522 * lg + 0.0721750 * lb) / D65_YN;
  const z = (0.0193339 * lr + 0.1191920 * lg + 0.9503041 * lb) / D65_ZN;
  const fx = labF(x);
  const fy = labF(y);
  const fz = labF(z);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

export function labToRgbTuple(L: number, a: number, b: number): [number, number, number] {
  const fy = (L + 16) / 116;
  const fx = a / 500 + fy;
  const fz = fy - b / 200;
  const x = D65_XN * labFInv(fx);
  const y = D65_YN * labFInv(fy);
  const z = D65_ZN * labFInv(fz);
  const lr = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
  const lg = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
  const lb = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
  return [linearToSrgb(lr), linearToSrgb(lg), linearToSrgb(lb)];
}

export function rgbBufferToLab(rgb: Uint8Array, nPixels: number): Float32Array {
  const out = new Float32Array(nPixels * 3);
  for (let i = 0; i < nPixels; i++) {
    const o = i * 3;
    const [L, a, b] = rgbToLabTuple(rgb[o], rgb[o + 1], rgb[o + 2]);
    out[o] = L;
    out[o + 1] = a;
    out[o + 2] = b;
  }
  return out;
}

export function labBufferToRgb(lab: Float32Array, nPixels: number): Uint8Array {
  const out = new Uint8Array(nPixels * 3);
  for (let i = 0; i < nPixels; i++) {
    const o = i * 3;
    const [r, g, b] = labToRgbTuple(lab[o], lab[o + 1], lab[o + 2]);
    out[o] = r;
    out[o + 1] = g;
    out[o + 2] = b;
  }
  return out;
}

const DEG2RAD = Math.PI / 180;
const RAD2DEG = 180 / Math.PI;
const POW25_7 = Math.pow(25, 7);

export function ciede2000(
  L1: number, a1: number, b1: number,
  L2: number, a2: number, b2: number,
): number {
  const C1 = Math.hypot(a1, b1);
  const C2 = Math.hypot(a2, b2);
  const Cbar = 0.5 * (C1 + C2);
  const Cbar7 = Math.pow(Cbar, 7);
  const G = 0.5 * (1 - Math.sqrt(Cbar7 / (Cbar7 + POW25_7)));
  const a1p = (1 + G) * a1;
  const a2p = (1 + G) * a2;
  const C1p = Math.hypot(a1p, b1);
  const C2p = Math.hypot(a2p, b2);

  let h1p = Math.atan2(b1, a1p) * RAD2DEG;
  if (h1p < 0) h1p += 360;
  let h2p = Math.atan2(b2, a2p) * RAD2DEG;
  if (h2p < 0) h2p += 360;

  const dLp = L2 - L1;
  const dCp = C2p - C1p;
  let dhp = 0;
  if (C1p * C2p !== 0) {
    const diff = h2p - h1p;
    if (Math.abs(diff) <= 180) dhp = diff;
    else if (diff > 180) dhp = diff - 360;
    else dhp = diff + 360;
  }
  const dHp = 2 * Math.sqrt(C1p * C2p) * Math.sin(0.5 * dhp * DEG2RAD);

  const Lpbar = 0.5 * (L1 + L2);
  const Cpbar = 0.5 * (C1p + C2p);
  let hpbar = h1p + h2p;
  if (C1p * C2p !== 0) {
    if (Math.abs(h1p - h2p) <= 180) hpbar = 0.5 * (h1p + h2p);
    else if (h1p + h2p < 360) hpbar = 0.5 * (h1p + h2p + 360);
    else hpbar = 0.5 * (h1p + h2p - 360);
  }

  const T =
    1 -
    0.17 * Math.cos((hpbar - 30) * DEG2RAD) +
    0.24 * Math.cos(2 * hpbar * DEG2RAD) +
    0.32 * Math.cos((3 * hpbar + 6) * DEG2RAD) -
    0.20 * Math.cos((4 * hpbar - 63) * DEG2RAD);

  const dTheta = 30 * Math.exp(-Math.pow((hpbar - 275) / 25, 2));
  const Cpbar7 = Math.pow(Cpbar, 7);
  const RC = 2 * Math.sqrt(Cpbar7 / (Cpbar7 + POW25_7));
  const L50sq = Math.pow(Lpbar - 50, 2);
  const SL = 1 + (0.015 * L50sq) / Math.sqrt(20 + L50sq);
  const SC = 1 + 0.045 * Cpbar;
  const SH = 1 + 0.015 * Cpbar * T;
  const RT = -Math.sin(2 * dTheta * DEG2RAD) * RC;

  const tL = dLp / SL;
  const tC = dCp / SC;
  const tH = dHp / SH;
  return Math.sqrt(tL * tL + tC * tC + tH * tH + RT * tC * tH);
}

function hexToRgb(hex: string): [number, number, number] {
  const clean = hex.trim().replace('#', '');
  const full = clean.length === 3 ? clean.split('').map((c) => c + c).join('') : clean;
  if (full.length !== 6) throw new Error(`Invalid hex color: ${hex}`);
  return [
    Number.parseInt(full.slice(0, 2), 16),
    Number.parseInt(full.slice(2, 4), 16),
    Number.parseInt(full.slice(4, 6), 16),
  ];
}

export function loadPalette(palettePath = PALETTE_PATH): PaletteColor[] {
  const raw = JSON.parse(fs.readFileSync(palettePath, 'utf-8')) as {
    colors?: Array<{ id?: string; name?: string; hex?: string }>;
  };
  const entries = raw.colors || [];
  if (entries.length !== 21) {
    throw new Error(`ChromaBook palette is locked to exactly 21 colours; found ${entries.length}`);
  }
  const seen = new Set<string>();
  return entries.map((entry, i) => {
    const id = String(entry.id || '').trim();
    if (!id || seen.has(id)) throw new Error(`Invalid or duplicate palette ID at index ${i}`);
    seen.add(id);
    const rgb = hexToRgb(String(entry.hex || ''));
    return {
      id,
      name: String(entry.name || `Color ${i + 1}`),
      hex: String(entry.hex || '').toUpperCase(),
      rgb,
      lab: rgbToLabTuple(rgb[0], rgb[1], rgb[2]),
    };
  });
}

// --- Image Processing Primitives ---

interface RgbImage {
  data: Uint8Array;
  width: number;
  height: number;
}

function gaussianKernel1D(sigma: number): Float32Array {
  const radius = Math.max(1, Math.min(48, Math.ceil(sigma * 3)));
  const k = new Float32Array(radius * 2 + 1);
  let sum = 0;
  const s2 = 2 * sigma * sigma;
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-(i * i) / s2);
    k[i + radius] = v;
    sum += v;
  }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  return k;
}

function gaussianBlurFloat(src: Float32Array, w: number, h: number, sigma: number): Float32Array {
  if (sigma <= 0.05) return new Float32Array(src);
  const k = gaussianKernel1D(sigma);
  const r = (k.length - 1) >> 1;
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);

  for (let y = 0; y < h; y++) {
    const rowOff = y * w;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) {
        const sx = Math.max(0, Math.min(w - 1, x + i));
        acc += src[rowOff + sx] * k[i + r];
      }
      tmp[rowOff + x] = acc;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) {
        const sy = Math.max(0, Math.min(h - 1, y + i));
        acc += tmp[sy * w + x] * k[i + r];
      }
      out[y * w + x] = acc;
    }
  }
  return out;
}

async function loadCropResize(imagePath: string, p: MappingParams): Promise<RgbImage> {
  const base = sharp(imagePath).rotate().flatten({ background: '#ffffff' });
  const meta = await base.metadata();
  const origW = meta.width || 1;
  const origH = meta.height || 1;

  let pipeline = base;
  if (p.crop_enabled) {
    const x0 = Math.max(0, Math.min(98, p.crop_x0));
    const y0 = Math.max(0, Math.min(98, p.crop_y0));
    const x1 = Math.max(x0 + 1, Math.min(100, p.crop_x1));
    const y1 = Math.max(y0 + 1, Math.min(100, p.crop_y1));
    const left = Math.round((x0 * origW) / 100);
    const top = Math.round((y0 * origH) / 100);
    const right = Math.max(left + 1, Math.round((x1 * origW) / 100));
    const bottom = Math.max(top + 1, Math.round((y1 * origH) / 100));
    pipeline = pipeline.extract({ left, top, width: right - left, height: bottom - top });
  }

  const maxSide = Math.trunc(p.max_side);
  if (maxSide > 0) {
    pipeline = pipeline.resize({
      width: maxSide,
      height: maxSide,
      fit: 'inside',
      withoutEnlargement: true,
    });
  }

  const { data, info } = await pipeline.removeAlpha().raw().toBuffer({ resolveWithObject: true });
  return { data: new Uint8Array(data), width: info.width, height: info.height };
}

function illuminationCorrection(src: RgbImage, strength: number, sigmaFraction: number): RgbImage {
  if (strength <= 0) return { data: new Uint8Array(src.data), width: src.width, height: src.height };
  const n = src.width * src.height;
  const lab = rgbBufferToLab(src.data, n);
  const L = new Float32Array(n);
  for (let i = 0; i < n; i++) L[i] = lab[i * 3];

  const sigma = Math.max(3, Math.min(src.width, src.height) * sigmaFraction);
  const bg = gaussianBlurFloat(L, src.width, src.height, sigma);
  const sortedBg = Float32Array.from(bg).sort();
  const target = sortedBg[sortedBg.length >> 1];

  for (let i = 0; i < n; i++) {
    lab[i * 3] = Math.max(0, Math.min(100, L[i] - strength * (bg[i] - target)));
  }
  return { data: labBufferToRgb(lab, n), width: src.width, height: src.height };
}

function claheLuminance(src: RgbImage, clipLimit: number, tileGrid: number): RgbImage {
  if (clipLimit <= 0) return { data: new Uint8Array(src.data), width: src.width, height: src.height };
  const { width: w, height: h } = src;
  const n = w * h;
  const lab = rgbBufferToLab(src.data, n);
  const L8 = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    L8[i] = Math.max(0, Math.min(255, Math.round((lab[i * 3] / 100) * 255)));
  }

  const tilesX = Math.max(2, Math.trunc(tileGrid));
  const tilesY = Math.max(2, Math.trunc(tileGrid));
  const tileW = w / tilesX;
  const tileH = h / tilesY;
  const luts: Uint8Array[] = [];

  for (let ty = 0; ty < tilesY; ty++) {
    const y0 = Math.floor(ty * tileH);
    const y1 = ty === tilesY - 1 ? h : Math.floor((ty + 1) * tileH);
    for (let tx = 0; tx < tilesX; tx++) {
      const x0 = Math.floor(tx * tileW);
      const x1 = tx === tilesX - 1 ? w : Math.floor((tx + 1) * tileW);
      const hist = new Int32Array(256);
      let count = 0;
      for (let y = y0; y < y1; y++) {
        const off = y * w;
        for (let x = x0; x < x1; x++) {
          hist[L8[off + x]]++;
          count++;
        }
      }
      const limit = Math.max(1, Math.floor((clipLimit * count) / 256));
      let excess = 0;
      for (let i = 0; i < 256; i++) {
        if (hist[i] > limit) {
          excess += hist[i] - limit;
          hist[i] = limit;
        }
      }
      const redist = Math.floor(excess / 256);
      const rem = excess - redist * 256;
      for (let i = 0; i < 256; i++) hist[i] += redist + (i < rem ? 1 : 0);

      const lut = new Uint8Array(256);
      let cdf = 0;
      const denom = Math.max(1, count);
      for (let i = 0; i < 256; i++) {
        cdf += hist[i];
        lut[i] = Math.max(0, Math.min(255, Math.round((cdf / denom) * 255)));
      }
      luts.push(lut);
    }
  }

  for (let y = 0; y < h; y++) {
    const fy = (y + 0.5) / tileH - 0.5;
    const ty0 = Math.max(0, Math.min(tilesY - 1, Math.floor(fy)));
    const ty1 = Math.max(0, Math.min(tilesY - 1, ty0 + 1));
    const wy = Math.max(0, Math.min(1, fy - ty0));
    for (let x = 0; x < w; x++) {
      const fx = (x + 0.5) / tileW - 0.5;
      const tx0 = Math.max(0, Math.min(tilesX - 1, Math.floor(fx)));
      const tx1 = Math.max(0, Math.min(tilesX - 1, tx0 + 1));
      const wx = Math.max(0, Math.min(1, fx - tx0));
      const idx = y * w + x;
      const v = L8[idx];
      const v00 = luts[ty0 * tilesX + tx0][v];
      const v10 = luts[ty0 * tilesX + tx1][v];
      const v01 = luts[ty1 * tilesX + tx0][v];
      const v11 = luts[ty1 * tilesX + tx1][v];
      const interp = (1 - wy) * ((1 - wx) * v00 + wx * v10) + wy * ((1 - wx) * v01 + wx * v11);
      lab[idx * 3] = (interp / 255) * 100;
    }
  }

  return { data: labBufferToRgb(lab, n), width: w, height: h };
}

function denoiseRgb(src: RgbImage, method: string, strength: number): RgbImage {
  const { width: w, height: h, data } = src;
  if (method === 'none' || strength <= 0) {
    return { data: new Uint8Array(data), width: w, height: h };
  }
  const out = new Uint8Array(data.length);
  if (method === 'median') {
    const r = strength < 35 ? 1 : 2;
    const win = (2 * r + 1) * (2 * r + 1);
    const buf = new Uint8Array(win);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        for (let c = 0; c < 3; c++) {
          let k = 0;
          for (let dy = -r; dy <= r; dy++) {
            const sy = Math.max(0, Math.min(h - 1, y + dy));
            for (let dx = -r; dx <= r; dx++) {
              const sx = Math.max(0, Math.min(w - 1, x + dx));
              buf[k++] = data[(sy * w + sx) * 3 + c];
            }
          }
          buf.sort();
          out[(y * w + x) * 3 + c] = buf[win >> 1];
        }
      }
    }
    return { data: out, width: w, height: h };
  }

  // Bilateral / NLM approximation (fast edge-preserving spatial + color Gaussian filter)
  const radius = method === 'nlm' ? 3 : strength < 35 ? 2 : 3;
  const sigmaColor = Math.max(4, strength * (method === 'nlm' ? 1.4 : 1.1));
  const sigmaSpace = Math.max(1, strength / 8);
  const inv2SC2 = -0.5 / (sigmaColor * sigmaColor);
  const inv2SS2 = -0.5 / (sigmaSpace * sigmaSpace);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const ci = (y * w + x) * 3;
      const r0 = data[ci];
      const g0 = data[ci + 1];
      const b0 = data[ci + 2];
      let sr = 0, sg = 0, sb = 0, wsum = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        const sy = Math.max(0, Math.min(h - 1, y + dy));
        for (let dx = -radius; dx <= radius; dx++) {
          const sx = Math.max(0, Math.min(w - 1, x + dx));
          const si = (sy * w + sx) * 3;
          const dr = data[si] - r0;
          const dg = data[si + 1] - g0;
          const db = data[si + 2] - b0;
          const weight = Math.exp((dx * dx + dy * dy) * inv2SS2 + (dr * dr + dg * dg + db * db) * inv2SC2);
          sr += data[si] * weight;
          sg += data[si + 1] * weight;
          sb += data[si + 2] * weight;
          wsum += weight;
        }
      }
      out[ci] = Math.round(sr / wsum);
      out[ci + 1] = Math.round(sg / wsum);
      out[ci + 2] = Math.round(sb / wsum);
    }
  }
  return { data: out, width: w, height: h };
}

function multiscaleSharpen(src: RgbImage, amountFine: number, amountCoarse: number): RgbImage {
  if (amountFine <= 0 && amountCoarse <= 0) {
    return { data: new Uint8Array(src.data), width: src.width, height: src.height };
  }
  const n = src.width * src.height;
  const lab = rgbBufferToLab(src.data, n);
  const L = new Float32Array(n);
  for (let i = 0; i < n; i++) L[i] = lab[i * 3];
  const blurFine = gaussianBlurFloat(L, src.width, src.height, 0.8);
  const blurCoarse = gaussianBlurFloat(L, src.width, src.height, 2.0);
  for (let i = 0; i < n; i++) {
    const fine = L[i] - blurFine[i];
    const coarse = L[i] - blurCoarse[i];
    lab[i * 3] = Math.max(0, Math.min(100, L[i] + amountFine * fine + amountCoarse * coarse));
  }
  return { data: labBufferToRgb(lab, n), width: src.width, height: src.height };
}

function rgbToGray(src: RgbImage): Float32Array {
  const n = src.width * src.height;
  const g = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const o = i * 3;
    g[i] = 0.299 * src.data[o] + 0.587 * src.data[o + 1] + 0.114 * src.data[o + 2];
  }
  return g;
}

function cannyEdges(gray: Float32Array, w: number, h: number, low: number, high: number): Uint8Array {
  const blurred = gaussianBlurFloat(gray, w, h, 1.0);
  const mag = new Float32Array(w * h);
  const dir = new Uint8Array(w * h);

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx =
        -blurred[(y - 1) * w + (x - 1)] + blurred[(y - 1) * w + (x + 1)]
        - 2 * blurred[y * w + (x - 1)] + 2 * blurred[y * w + (x + 1)]
        - blurred[(y + 1) * w + (x - 1)] + blurred[(y + 1) * w + (x + 1)];
      const gy =
        -blurred[(y - 1) * w + (x - 1)] - 2 * blurred[(y - 1) * w + x] - blurred[(y - 1) * w + (x + 1)]
        + blurred[(y + 1) * w + (x - 1)] + 2 * blurred[(y + 1) * w + x] + blurred[(y + 1) * w + (x + 1)];
      mag[i] = Math.hypot(gx, gy);
      const angle = ((Math.atan2(gy, gx) * 180) / Math.PI + 180) % 180;
      if (angle < 22.5 || angle >= 157.5) dir[i] = 0;
      else if (angle < 67.5) dir[i] = 1;
      else if (angle < 112.5) dir[i] = 2;
      else dir[i] = 3;
    }
  }

  const nms = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const m = mag[i];
      let m1 = 0, m2 = 0;
      if (dir[i] === 0) {
        m1 = mag[i - 1]; m2 = mag[i + 1];
      } else if (dir[i] === 1) {
        m1 = mag[(y - 1) * w + (x + 1)]; m2 = mag[(y + 1) * w + (x - 1)];
      } else if (dir[i] === 2) {
        m1 = mag[(y - 1) * w + x]; m2 = mag[(y + 1) * w + x];
      } else {
        m1 = mag[(y - 1) * w + (x - 1)]; m2 = mag[(y + 1) * w + (x + 1)];
      }
      if (m >= m1 && m >= m2) nms[i] = m;
    }
  }

  const out = new Uint8Array(w * h);
  const stack: number[] = [];
  for (let i = 0; i < w * h; i++) {
    if (nms[i] >= high) {
      out[i] = 255;
      stack.push(i);
    } else if (nms[i] >= low) {
      out[i] = 128;
    }
  }
  while (stack.length > 0) {
    const idx = stack.pop()!;
    const x = idx % w;
    const y = (idx - x) / w;
    for (let dy = -1; dy <= 1; dy++) {
      const sy = y + dy;
      if (sy < 0 || sy >= h) continue;
      for (let dx = -1; dx <= 1; dx++) {
        const sx = x + dx;
        if (sx < 0 || sx >= w) continue;
        const ni = sy * w + sx;
        if (out[ni] === 128) {
          out[ni] = 255;
          stack.push(ni);
        }
      }
    }
  }
  for (let i = 0; i < out.length; i++) {
    if (out[i] !== 255) out[i] = 0;
  }
  return out;
}

function dilate3x3(src: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let mx = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const sy = Math.max(0, Math.min(h - 1, y + dy));
        for (let dx = -1; dx <= 1; dx++) {
          const sx = Math.max(0, Math.min(w - 1, x + dx));
          if (src[sy * w + sx] > mx) mx = src[sy * w + sx];
        }
      }
      out[y * w + x] = mx;
    }
  }
  return out;
}

function erode3x3(src: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let mn = 255;
      for (let dy = -1; dy <= 1; dy++) {
        const sy = Math.max(0, Math.min(h - 1, y + dy));
        for (let dx = -1; dx <= 1; dx++) {
          const sx = Math.max(0, Math.min(w - 1, x + dx));
          if (src[sy * w + sx] < mn) mn = src[sy * w + sx];
        }
      }
      out[y * w + x] = mn;
    }
  }
  return out;
}

function extractEdges(rgb: RgbImage, low: number, high: number, autoCanny: boolean, closeIter: number): Float32Array {
  const { width: w, height: h } = rgb;
  const gray = rgbToGray(rgb);
  const mag = new Float32Array(w * h);

  // Scharr gradient
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const gx =
        -3 * gray[(y - 1) * w + (x - 1)] + 3 * gray[(y - 1) * w + (x + 1)]
        - 10 * gray[y * w + (x - 1)] + 10 * gray[y * w + (x + 1)]
        - 3 * gray[(y + 1) * w + (x - 1)] + 3 * gray[(y + 1) * w + (x + 1)];
      const gy =
        -3 * gray[(y - 1) * w + (x - 1)] - 10 * gray[(y - 1) * w + x] - 3 * gray[(y - 1) * w + (x + 1)]
        + 3 * gray[(y + 1) * w + (x - 1)] + 10 * gray[(y + 1) * w + x] + 3 * gray[(y + 1) * w + (x + 1)];
      mag[y * w + x] = Math.hypot(gx, gy);
    }
  }
  const sortedMag = Float32Array.from(mag).sort();
  const scale = Math.max(1e-6, sortedMag[Math.min(sortedMag.length - 1, Math.floor(sortedMag.length * 0.995))]);

  let lo = low;
  let hi = high;
  if (autoCanny) {
    const sortedGray = Float32Array.from(gray).sort();
    const med = sortedGray[sortedGray.length >> 1];
    lo = Math.max(0, Math.trunc(0.67 * med));
    hi = Math.min(255, Math.max(lo + 1, Math.trunc(1.33 * med)));
  } else if (hi <= lo) {
    hi = Math.min(255, lo + 1);
  }

  let canny = cannyEdges(gray, w, h, lo, hi);
  for (let i = 0; i < closeIter; i++) {
    canny = erode3x3(dilate3x3(canny, w, h), w, h);
  }

  const combined = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const grad = Math.min(1, mag[i] / scale);
    const cf = canny[i] / 255;
    combined[i] = Math.max(grad * 0.75, cf);
  }
  const smoothed = gaussianBlurFloat(combined, w, h, 0.65);
  for (let i = 0; i < smoothed.length; i++) {
    smoothed[i] = Math.max(0, Math.min(1, smoothed[i]));
  }
  return smoothed;
}

// --- Grid, SLIC, Unary & ICM Engines ---

interface GridGeometry {
  rows: number;
  cols: number;
  xe: number[];
  ye: number[];
}

function gridGeometry(h: number, w: number, colsInput: number): GridGeometry {
  const cols = Math.max(4, Math.min(180, Math.trunc(colsInput)));
  const rows = Math.max(4, Math.round((cols * h) / w));
  const xe: number[] = [];
  const ye: number[] = [];
  for (let i = 0; i <= cols; i++) xe.push(Math.round((i * w) / cols));
  for (let i = 0; i <= rows; i++) ye.push(Math.round((i * h) / rows));
  return { rows, cols, xe, ye };
}

function trimmedColumnMean(costs: Float32Array, nSamples: number, kColors: number, trimFraction: number): Float32Array {
  const out = new Float32Array(kColors);
  const frac = Math.max(0, Math.min(0.4, trimFraction));
  const cut = frac <= 0 || nSamples < 5 ? 0 : Math.min(Math.floor(nSamples * frac), Math.floor((nSamples - 1) / 2));
  const col = new Float32Array(nSamples);
  const count = nSamples - 2 * cut;
  for (let c = 0; c < kColors; c++) {
    for (let s = 0; s < nSamples; s++) col[s] = costs[s * kColors + c];
    if (cut > 0) col.sort();
    let sum = 0;
    for (let s = cut; s < nSamples - cut; s++) sum += col[s];
    out[c] = sum / Math.max(1, count);
  }
  return out;
}

function buildCellUnary(
  lab: Float32Array, w: number, h: number, palette: PaletteColor[],
  cols: number, maxSamples: number, trimFraction: number,
): { unary: Float32Array; geometry: GridGeometry } {
  const geom = gridGeometry(h, w, cols);
  const k = palette.length;
  const unary = new Float32Array(geom.rows * geom.cols * k);

  for (let r = 0; r < geom.rows; r++) {
    const y0 = geom.ye[r], y1 = geom.ye[r + 1];
    for (let c = 0; c < geom.cols; c++) {
      const x0 = geom.xe[c], x1 = geom.xe[c + 1];
      const cellPixels = Math.max(1, (y1 - y0) * (x1 - x0));
      const nSamples = Math.min(cellPixels, maxSamples);
      const costs = new Float32Array(nSamples * k);
      for (let s = 0; s < nSamples; s++) {
        const idx = cellPixels <= nSamples ? s : Math.floor((s * (cellPixels - 1)) / Math.max(1, nSamples - 1));
        const cy = y0 + Math.floor(idx / Math.max(1, x1 - x0));
        const cx = x0 + (idx % Math.max(1, x1 - x0));
        const pOff = (cy * w + cx) * 3;
        const L1 = lab[pOff], a1 = lab[pOff + 1], b1 = lab[pOff + 2];
        for (let p = 0; p < k; p++) {
          const [L2, a2, b2] = palette[p].lab;
          costs[s * k + p] = ciede2000(L1, a1, b1, L2, a2, b2);
        }
      }
      const meanCosts = trimmedColumnMean(costs, nSamples, k, trimFraction);
      unary.set(meanCosts, (r * geom.cols + c) * k);
    }
  }
  return { unary, geometry: geom };
}

function makeSlic(lab: Float32Array, w: number, h: number, nSegments: number, compactness: number): {
  labels: Int32Array;
  boundary: Float32Array;
} {
  const n = w * h;
  const step = Math.max(4, Math.sqrt(n / Math.max(16, nSegments)));
  const centers: Array<{ L: number; a: number; b: number; x: number; y: number }> = [];

  for (let y = Math.floor(step / 2); y < h; y += Math.round(step)) {
    for (let x = Math.floor(step / 2); x < w; x += Math.round(step)) {
      const o = (y * w + x) * 3;
      centers.push({ L: lab[o], a: lab[o + 1], b: lab[o + 2], x, y });
    }
  }
  if (centers.length === 0) {
    const o = Math.floor(h / 2) * w * 3 + Math.floor(w / 2) * 3;
    centers.push({ L: lab[o], a: lab[o + 1], b: lab[o + 2], x: w / 2, y: h / 2 });
  }

  const labels = new Int32Array(n).fill(-1);
  const dist = new Float32Array(n);
  const kCount = centers.length;
  const invStep2 = (compactness * compactness) / (step * step);
  const searchR = Math.ceil(step);

  for (let iter = 0; iter < 5; iter++) {
    dist.fill(Number.POSITIVE_INFINITY);
    for (let ci = 0; ci < kCount; ci++) {
      const ct = centers[ci];
      const y0 = Math.max(0, Math.floor(ct.y - searchR));
      const y1 = Math.min(h - 1, Math.ceil(ct.y + searchR));
      const x0 = Math.max(0, Math.floor(ct.x - searchR));
      const x1 = Math.min(w - 1, Math.ceil(ct.x + searchR));
      for (let y = y0; y <= y1; y++) {
        const dy = y - ct.y;
        const dy2 = dy * dy;
        for (let x = x0; x <= x1; x++) {
          const i = y * w + x;
          const o = i * 3;
          const dL = lab[o] - ct.L;
          const da = lab[o + 1] - ct.a;
          const db = lab[o + 2] - ct.b;
          const dx = x - ct.x;
          const d = dL * dL + da * da + db * db + (dx * dx + dy2) * invStep2;
          if (d < dist[i]) {
            dist[i] = d;
            labels[i] = ci;
          }
        }
      }
    }
    const sumL = new Float64Array(kCount);
    const suma = new Float64Array(kCount);
    const sumb = new Float64Array(kCount);
    const sumx = new Float64Array(kCount);
    const sumy = new Float64Array(kCount);
    const cnt = new Int32Array(kCount);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const ci = labels[i];
        if (ci >= 0) {
          const o = i * 3;
          sumL[ci] += lab[o];
          suma[ci] += lab[o + 1];
          sumb[ci] += lab[o + 2];
          sumx[ci] += x;
          sumy[ci] += y;
          cnt[ci]++;
        }
      }
    }
    for (let ci = 0; ci < kCount; ci++) {
      if (cnt[ci] > 0) {
        centers[ci] = {
          L: sumL[ci] / cnt[ci],
          a: suma[ci] / cnt[ci],
          b: sumb[ci] / cnt[ci],
          x: sumx[ci] / cnt[ci],
          y: sumy[ci] / cnt[ci],
        };
      }
    }
  }

  for (let i = 0; i < n; i++) {
    if (labels[i] < 0) labels[i] = 0;
  }

  const rawBound = new Uint8Array(n);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const cur = labels[i];
      if ((x + 1 < w && labels[i + 1] !== cur) || (y + 1 < h && labels[i + w] !== cur)) {
        rawBound[i] = 255;
      }
    }
  }
  const dilated = dilate3x3(rawBound, w, h);
  const bf = new Float32Array(n);
  for (let i = 0; i < n; i++) bf[i] = dilated[i] / 255;
  const smoothed = gaussianBlurFloat(bf, w, h, 0.8);
  let mx = 0;
  for (let i = 0; i < n; i++) if (smoothed[i] > mx) mx = smoothed[i];
  if (mx > 0) for (let i = 0; i < n; i++) smoothed[i] /= mx;

  return { labels, boundary: smoothed };
}

function buildRegionBalancedUnary(
  lab: Float32Array, w: number, slicLabels: Int32Array, geom: GridGeometry,
  palette: PaletteColor[], maxSamples: number, trimFraction: number,
): Float32Array {
  const k = palette.length;
  const out = new Float32Array(geom.rows * geom.cols * k);
  for (let r = 0; r < geom.rows; r++) {
    const y0 = geom.ye[r], y1 = geom.ye[r + 1];
    for (let c = 0; c < geom.cols; c++) {
      const x0 = geom.xe[c], x1 = geom.xe[c + 1];
      const cellSize = Math.max(1, (y1 - y0) * (x1 - x0));
      const byRegion = new Map<number, number[]>();
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const i = y * w + x;
          const sid = slicLabels[i];
          let arr = byRegion.get(sid);
          if (!arr) {
            arr = [];
            byRegion.set(sid, arr);
          }
          arr.push(i);
        }
      }
      const cellCost = new Float32Array(k);
      let weightSum = 0;
      for (const pixIndices of byRegion.values()) {
        const count = pixIndices.length;
        const target = Math.max(12, Math.min(maxSamples, Math.floor((maxSamples * count) / cellSize * 2)));
        const nSamples = Math.min(count, target);
        const costs = new Float32Array(nSamples * k);
        for (let s = 0; s < nSamples; s++) {
          const idx = count <= nSamples ? s : Math.floor((s * (count - 1)) / Math.max(1, nSamples - 1));
          const pOff = pixIndices[idx] * 3;
          const L1 = lab[pOff], a1 = lab[pOff + 1], b1 = lab[pOff + 2];
          for (let p = 0; p < k; p++) {
            const [L2, a2, b2] = palette[p].lab;
            costs[s * k + p] = ciede2000(L1, a1, b1, L2, a2, b2);
          }
        }
        const meanCosts = trimmedColumnMean(costs, nSamples, k, trimFraction);
        const weight = Math.sqrt(count);
        weightSum += weight;
        for (let p = 0; p < k; p++) cellCost[p] += meanCosts[p] * weight;
      }
      const norm = Math.max(1e-8, weightSum);
      const base = (r * geom.cols + c) * k;
      for (let p = 0; p < k; p++) out[base + p] = cellCost[p] / norm;
    }
  }
  return out;
}

function legacyMeanLabGrid(lab: Float32Array, w: number, geom: GridGeometry, palette: PaletteColor[]): Uint8Array {
  const labels = new Uint8Array(geom.rows * geom.cols);
  for (let r = 0; r < geom.rows; r++) {
    const y0 = geom.ye[r], y1 = geom.ye[r + 1];
    for (let c = 0; c < geom.cols; c++) {
      const x0 = geom.xe[c], x1 = geom.xe[c + 1];
      let sL = 0, sa = 0, sb = 0, cnt = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const o = (y * w + x) * 3;
          sL += lab[o]; sa += lab[o + 1]; sb += lab[o + 2];
          cnt++;
        }
      }
      const mL = sL / Math.max(1, cnt);
      const ma = sa / Math.max(1, cnt);
      const mb = sb / Math.max(1, cnt);
      let bestIdx = 0;
      let bestDist = Number.POSITIVE_INFINITY;
      for (let p = 0; p < palette.length; p++) {
        const [pL, pa, pb] = palette[p].lab;
        const d = Math.hypot(pL - mL, pa - ma, pb - mb);
        if (d < bestDist) {
          bestDist = d;
          bestIdx = p;
        }
      }
      labels[r * geom.cols + c] = bestIdx;
    }
  }
  return labels;
}

function boundaryPairwiseWeights(
  boundaryMap: Float32Array, w: number, h: number, geom: GridGeometry, beta: number,
): { wh: Float32Array; wv: Float32Array } {
  const { rows, cols, xe, ye } = geom;
  const wh = new Float32Array(rows * Math.max(0, cols - 1));
  const wv = new Float32Array(Math.max(0, rows - 1) * cols);
  const radius = 2;

  for (let r = 0; r < rows; r++) {
    const y0 = ye[r], y1 = ye[r + 1];
    for (let c = 0; c < cols - 1; c++) {
      const x = xe[c + 1];
      const xa = Math.max(0, x - radius);
      const xb = Math.min(w, x + radius + 1);
      let sum = 0, cnt = 0;
      for (let y = y0; y < y1; y++) {
        for (let xx = xa; xx < xb; xx++) {
          sum += boundaryMap[y * w + xx];
          cnt++;
        }
      }
      const score = cnt > 0 ? Math.max(0, Math.min(1, sum / cnt)) : 0;
      wh[r * (cols - 1) + c] = Math.exp(-beta * score);
    }
  }

  for (let r = 0; r < rows - 1; r++) {
    const y = ye[r + 1];
    const ya = Math.max(0, y - radius);
    const yb = Math.min(h, y + radius + 1);
    for (let c = 0; c < cols; c++) {
      const x0 = xe[c], x1 = xe[c + 1];
      let sum = 0, cnt = 0;
      for (let yy = ya; yy < yb; yy++) {
        for (let x = x0; x < x1; x++) {
          sum += boundaryMap[yy * w + x];
          cnt++;
        }
      }
      const score = cnt > 0 ? Math.max(0, Math.min(1, sum / cnt)) : 0;
      wv[r * cols + c] = Math.exp(-beta * score);
    }
  }
  return { wh, wv };
}

function argminUnary(unary: Float32Array, rows: number, cols: number, k: number): Uint8Array {
  const labels = new Uint8Array(rows * cols);
  for (let i = 0; i < rows * cols; i++) {
    const off = i * k;
    let best = 0;
    let bestVal = unary[off];
    for (let p = 1; p < k; p++) {
      if (unary[off + p] < bestVal) {
        bestVal = unary[off + p];
        best = p;
      }
    }
    labels[i] = best;
  }
  return labels;
}

function icmOptimize(
  unary: Float32Array, rows: number, cols: number, k: number,
  wh: Float32Array, wv: Float32Array, lam: number, iterations: number,
): Uint8Array {
  const labels = argminUnary(unary, rows, cols, k);
  const costs = new Float32Array(k);
  for (let it = 0; it < iterations; it++) {
    let changes = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        const uOff = idx * k;
        for (let p = 0; p < k; p++) costs[p] = unary[uOff + p];

        if (c > 0) {
          const w = lam * wh[r * (cols - 1) + (c - 1)];
          const nb = labels[idx - 1];
          for (let p = 0; p < k; p++) if (p !== nb) costs[p] += w;
        }
        if (c < cols - 1) {
          const w = lam * wh[r * (cols - 1) + c];
          const nb = labels[idx + 1];
          for (let p = 0; p < k; p++) if (p !== nb) costs[p] += w;
        }
        if (r > 0) {
          const w = lam * wv[(r - 1) * cols + c];
          const nb = labels[idx - cols];
          for (let p = 0; p < k; p++) if (p !== nb) costs[p] += w;
        }
        if (r < rows - 1) {
          const w = lam * wv[r * cols + c];
          const nb = labels[idx + cols];
          for (let p = 0; p < k; p++) if (p !== nb) costs[p] += w;
        }

        let best = 0;
        let bestVal = costs[0];
        for (let p = 1; p < k; p++) {
          if (costs[p] < bestVal) {
            bestVal = costs[p];
            best = p;
          }
        }
        if (best !== labels[idx]) {
          labels[idx] = best;
          changes++;
        }
      }
    }
    if (changes === 0) break;
  }
  return labels;
}

// --- Rendering & Metrics ---

function renderPixelPreview(labels: Uint8Array, geom: GridGeometry, palette: PaletteColor[]): RgbImage {
  const { rows, cols, xe, ye } = geom;
  const w = xe[cols];
  const h = ye[rows];
  const data = new Uint8Array(w * h * 3);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const [cr, cg, cb] = palette[labels[r * cols + c]].rgb;
      for (let y = ye[r]; y < ye[r + 1]; y++) {
        for (let x = xe[c]; x < xe[c + 1]; x++) {
          const o = (y * w + x) * 3;
          data[o] = cr;
          data[o + 1] = cg;
          data[o + 2] = cb;
        }
      }
    }
  }
  return { data, width: w, height: h };
}

async function renderNumberedGridPng(
  labels: Uint8Array, rows: number, cols: number, palette: PaletteColor[],
  cellPx: number, lineWidth: number,
): Promise<Buffer> {
  const cp = Math.max(20, Math.min(100, Math.trunc(cellPx)));
  const lw = Math.max(1, Math.min(5, Math.trunc(lineWidth)));
  const w = cols * cp + 1;
  const h = rows * cp + 1;
  const fontSize = Math.max(9, Math.round(cp * 0.34));

  const lines: string[] = [];
  for (let r = 0; r <= rows; r++) {
    const y = r * cp + 0.5;
    lines.push(`<line x1="0" y1="${y}" x2="${w}" y2="${y}"/>`);
  }
  for (let c = 0; c <= cols; c++) {
    const x = c * cp + 0.5;
    lines.push(`<line x1="${x}" y1="0" x2="${x}" y2="${h}"/>`);
  }

  const texts: string[] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const id = palette[labels[r * cols + c]].id;
      const cx = c * cp + cp / 2;
      const cy = r * cp + cp / 2 + fontSize * 0.35;
      texts.push(`<text x="${cx}" y="${cy}">${id}</text>`);
    }
  }

  const svg = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">`,
    `<rect width="100%" height="100%" fill="#ffffff"/>`,
    `<g stroke="#0f0f0f" stroke-width="${lw}">`,
    lines.join(''),
    `</g>`,
    `<g fill="#0a0a0a" font-family="DejaVu Sans, Arial, Helvetica, sans-serif" font-size="${fontSize}" text-anchor="middle">`,
    texts.join(''),
    `</g>`,
    `</svg>`,
  ].join('');

  return sharp(Buffer.from(svg)).png().toBuffer();
}

function gridMetrics(labels: Uint8Array, rows: number, cols: number) {
  const total = rows * cols;
  const visited = new Uint8Array(total);
  let singleton = 0;
  let tiny = 0;
  const queue = new Int32Array(total);

  for (let i = 0; i < total; i++) {
    if (visited[i]) continue;
    const color = labels[i];
    visited[i] = 1;
    let head = 0, tail = 0;
    queue[tail++] = i;
    while (head < tail) {
      const cur = queue[head++];
      const x = cur % cols;
      const y = (cur - x) / cols;
      if (x > 0 && !visited[cur - 1] && labels[cur - 1] === color) {
        visited[cur - 1] = 1; queue[tail++] = cur - 1;
      }
      if (x < cols - 1 && !visited[cur + 1] && labels[cur + 1] === color) {
        visited[cur + 1] = 1; queue[tail++] = cur + 1;
      }
      if (y > 0 && !visited[cur - cols] && labels[cur - cols] === color) {
        visited[cur - cols] = 1; queue[tail++] = cur - cols;
      }
      if (y < rows - 1 && !visited[cur + cols] && labels[cur + cols] === color) {
        visited[cur + cols] = 1; queue[tail++] = cur + cols;
      }
    }
    if (tail === 1) singleton++;
    if (tail <= 2) tiny++;
  }

  let htrans = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols - 1; c++) {
      if (labels[r * cols + c] !== labels[r * cols + c + 1]) htrans++;
    }
  }
  let vtrans = 0;
  for (let r = 0; r < rows - 1; r++) {
    for (let c = 0; c < cols; c++) {
      if (labels[r * cols + c] !== labels[(r + 1) * cols + c]) vtrans++;
    }
  }
  const adj = rows * Math.max(0, cols - 1) + Math.max(0, rows - 1) * cols;
  return {
    singleton_ratio_pct: (100 * singleton) / Math.max(1, total),
    tiny_component_ratio_pct: (100 * tiny) / Math.max(1, total),
    transition_rate_pct: (100 * (htrans + vtrans)) / Math.max(1, adj),
  };
}

function edgeF1(ref: RgbImage, mapped: RgbImage): { f1: number; precision: number; recall: number } {
  const { width: w, height: h } = ref;
  const a = cannyEdges(rgbToGray(ref), w, h, 55, 150);
  const b = cannyEdges(rgbToGray(mapped), w, h, 55, 150);
  const ad = dilate3x3(a, w, h);
  const bd = dilate3x3(b, w, h);

  let aSum = 0, bSum = 0, tpB = 0, tpA = 0;
  for (let i = 0; i < w * h; i++) {
    const ab = a[i] > 0;
    const bb = b[i] > 0;
    if (ab) {
      aSum++;
      if (bd[i] > 0) tpA++;
    }
    if (bb) {
      bSum++;
      if (ad[i] > 0) tpB++;
    }
  }
  const precision = tpB / Math.max(1, bSum);
  const recall = tpA / Math.max(1, aSum);
  const f1 = (2 * precision * recall) / Math.max(1e-12, precision + recall);
  return { f1, precision, recall };
}

function computeSsim(ref: RgbImage, mapped: RgbImage): number {
  const { width: w, height: h } = ref;
  const n = w * h;
  const c1 = (0.01 * 255) ** 2;
  const c2 = (0.03 * 255) ** 2;
  let totalSsim = 0;
  for (let ch = 0; ch < 3; ch++) {
    let meanX = 0, meanY = 0;
    for (let i = 0; i < n; i++) {
      meanX += ref.data[i * 3 + ch];
      meanY += mapped.data[i * 3 + ch];
    }
    meanX /= n;
    meanY /= n;
    let varX = 0, varY = 0, covXY = 0;
    for (let i = 0; i < n; i++) {
      const dx = ref.data[i * 3 + ch] - meanX;
      const dy = mapped.data[i * 3 + ch] - meanY;
      varX += dx * dx;
      varY += dy * dy;
      covXY += dx * dy;
    }
    varX /= Math.max(1, n - 1);
    varY /= Math.max(1, n - 1);
    covXY /= Math.max(1, n - 1);
    const num = (2 * meanX * meanY + c1) * (2 * covXY + c2);
    const den = (meanX * meanX + meanY * meanY + c1) * (varX + varY + c2);
    totalSsim += num / den;
  }
  return totalSsim / 3;
}

function evaluateGrid(
  refRgb: RgbImage, procLab: Float32Array, mappedRgb: RgbImage,
  labels: Uint8Array, geom: GridGeometry,
) {
  const n = refRgb.width * refRgb.height;
  const refLab = rgbBufferToLab(refRgb.data, n);
  const mapLab = rgbBufferToLab(mappedRgb.data, n);

  // Sample up to 24,000 pixels evenly for fast, accurate CIEDE2000 summary stats
  const step = Math.max(1, Math.floor(n / 24000));
  const deRefList: number[] = [];
  let deProcSum = 0;
  let sampleCount = 0;
  for (let i = 0; i < n; i += step) {
    const o = i * 3;
    deRefList.push(ciede2000(refLab[o], refLab[o + 1], refLab[o + 2], mapLab[o], mapLab[o + 1], mapLab[o + 2]));
    deProcSum += ciede2000(procLab[o], procLab[o + 1], procLab[o + 2], mapLab[o], mapLab[o + 1], mapLab[o + 2]);
    sampleCount++;
  }
  deRefList.sort((a, b) => a - b);
  const deMean = deRefList.reduce((acc, v) => acc + v, 0) / Math.max(1, deRefList.length);
  const deMedian = deRefList[deRefList.length >> 1] ?? 0;
  const deP95 = deRefList[Math.min(deRefList.length - 1, Math.floor(deRefList.length * 0.95))] ?? 0;

  const ssim = computeSsim(refRgb, mappedRgb);
  const { f1, precision, recall } = edgeF1(refRgb, mappedRgb);
  const used = new Set(labels).size;

  return {
    DeltaE00_mean_vs_source: deMean,
    DeltaE00_median_vs_source: deMedian,
    DeltaE00_p95_vs_source: deP95,
    DeltaE00_mean_vs_processed: deProcSum / Math.max(1, sampleCount),
    SSIM_vs_source: ssim,
    Edge_F1_1px: f1,
    Edge_precision: precision,
    Edge_recall: recall,
    colors_used: used,
    ...gridMetrics(labels, geom.rows, geom.cols),
  };
}

async function saveRgbPng(img: RgbImage, filePath: string): Promise<void> {
  await sharp(Buffer.from(img.data), {
    raw: { width: img.width, height: img.height, channels: 3 },
  }).png().toFile(filePath);
}

function encodeLabelsNpy(labels: Uint8Array, rows: number, cols: number): Buffer {
  const headerStr = `{'descr': '|u1', 'fortran_order': False, 'shape': (${rows}, ${cols}), }`;
  const preLen = 10 + headerStr.length + 1;
  const padLen = (16 - (preLen % 16)) % 16;
  const padded = headerStr + ' '.repeat(padLen) + '\n';
  const headerBuf = Buffer.alloc(10 + padded.length);
  headerBuf[0] = 0x93;
  headerBuf.write('NUMPY', 1, 'ascii');
  headerBuf[6] = 1;
  headerBuf[7] = 0;
  headerBuf.writeUInt16LE(padded.length, 8);
  headerBuf.write(padded, 10, 'ascii');
  return Buffer.concat([headerBuf, Buffer.from(labels)]);
}

export interface EngineRunResult {
  key: string;
  name: string;
  labels: Uint8Array;
  pixel: RgbImage;
  numberedPng: Buffer;
  metrics: Record<string, number>;
}

export async function runImageAndSave(
  imagePath: string,
  outDir: string,
  rawParams: Record<string, unknown> | undefined,
  selectedEngine: string,
  includeStability: boolean,
): Promise<{ engine: string; confidence: string; totalRuntimeSeconds: number }> {
  const t0 = performance.now();
  const engineKey = selectedEngine.toUpperCase();
  if (!ENGINE_NAMES[engineKey]) throw new Error('selected_engine must be A, B, C or D');

  const palette = loadPalette(PALETTE_PATH);
  const p = validateParams(rawParams);

  const reference = await loadCropResize(imagePath, p);
  const illum = illuminationCorrection(reference, p.illumination_strength, p.illumination_sigma);
  const clahe = claheLuminance(illum, p.clahe_clip, p.clahe_tiles);
  const denoised = denoiseRgb(clahe, p.denoise_method, p.denoise_strength);
  const processed = multiscaleSharpen(denoised, p.sharpen_fine, p.sharpen_coarse);

  const n = processed.width * processed.height;
  const processedLab = rgbBufferToLab(processed.data, n);
  const edgeMap = extractEdges(processed, p.canny_low, p.canny_high, p.auto_canny, p.edge_close);
  const { unary, geometry } = buildCellUnary(
    processedLab, processed.width, processed.height, palette, p.grid_cols, p.max_samples, p.trim_fraction,
  );
  const slic = makeSlic(processedLab, processed.width, processed.height, p.slic_segments, p.slic_compactness);

  const runEngine = async (
    key: string, labBuf: Float32Array, cellUnary: Float32Array,
  ): Promise<EngineRunResult> => {
    const start = performance.now();
    let labels: Uint8Array;
    if (key === 'A') {
      labels = legacyMeanLabGrid(labBuf, processed.width, geometry, palette);
    } else if (key === 'B') {
      labels = argminUnary(cellUnary, geometry.rows, geometry.cols, palette.length);
    } else if (key === 'C') {
      const { wh, wv } = boundaryPairwiseWeights(edgeMap, processed.width, processed.height, geometry, p.boundary_beta);
      labels = icmOptimize(cellUnary, geometry.rows, geometry.cols, palette.length, wh, wv, p.smoothness_lambda, p.icm_iterations);
    } else {
      const regionUnary = buildRegionBalancedUnary(
        labBuf, processed.width, slic.labels, geometry, palette, p.max_samples, p.trim_fraction,
      );
      const blend = p.region_balance;
      const dUnary = new Float32Array(cellUnary.length);
      for (let i = 0; i < dUnary.length; i++) {
        dUnary[i] = (1 - blend) * cellUnary[i] + blend * regionUnary[i];
      }
      const combined = new Float32Array(edgeMap.length);
      for (let i = 0; i < combined.length; i++) {
        combined[i] = Math.max(edgeMap[i], 0.95 * slic.boundary[i]);
      }
      const { wh, wv } = boundaryPairwiseWeights(combined, processed.width, processed.height, geometry, p.boundary_beta);
      labels = icmOptimize(dUnary, geometry.rows, geometry.cols, palette.length, wh, wv, p.smoothness_lambda, p.icm_iterations);
    }
    const pixel = renderPixelPreview(labels, geometry, palette);
    const numberedPng = await renderNumberedGridPng(
      labels, geometry.rows, geometry.cols, palette, p.numbered_cell_px, p.numbered_line_width,
    );
    const metrics = evaluateGrid(reference, labBuf, pixel, labels, geometry) as Record<string, number>;
    metrics.runtime_seconds = (performance.now() - start) / 1000;
    return { key, name: ENGINE_NAMES[key], labels, pixel, numberedPng, metrics };
  };

  const results: Record<string, EngineRunResult> = {};
  for (const key of ['A', 'B', 'C', 'D']) {
    results[key] = await runEngine(key, processedLab, unary);
  }

  fs.mkdirSync(outDir, { recursive: true });
  const chosen = results[engineKey];
  fs.writeFileSync(path.join(outDir, 'labels.npy'), encodeLabelsNpy(chosen.labels, geometry.rows, geometry.cols));
  fs.writeFileSync(
    path.join(outDir, 'labels.json'),
    JSON.stringify({ rows: geometry.rows, cols: geometry.cols, data: Array.from(chosen.labels) }),
  );
  await saveRgbPng(chosen.pixel, path.join(outDir, 'color_preview.png'));
  fs.writeFileSync(path.join(outDir, 'numbered_preview.png'), chosen.numberedPng);

  const comparison: Record<string, Record<string, number>> = {};
  for (const key of ['A', 'B', 'C', 'D']) {
    const res = results[key];
    const folder = path.join(outDir, `engine_${key}`);
    fs.mkdirSync(folder, { recursive: true });
    await saveRgbPng(res.pixel, path.join(folder, 'color_preview.png'));
    fs.writeFileSync(path.join(folder, 'numbered_preview.png'), res.numberedPng);
    fs.writeFileSync(path.join(folder, 'labels.npy'), encodeLabelsNpy(res.labels, geometry.rows, geometry.cols));
    fs.writeFileSync(path.join(folder, 'metrics.json'), JSON.stringify(res.metrics, null, 2));
    comparison[key] = res.metrics;
  }

  const totalRuntimeSeconds = (performance.now() - t0) / 1000;
  const metricsPayload = {
    selected_engine: engineKey,
    total_runtime_seconds: totalRuntimeSeconds,
    engines: comparison,
    selected_metrics: chosen.metrics,
    palette_colors: palette.length,
  };
  fs.writeFileSync(path.join(outDir, 'engine_comparison_metrics.json'), JSON.stringify(metricsPayload, null, 2));

  if (includeStability) {
    const stabilityRows = [];
    for (const k of ['B', 'C', 'D']) {
      const base = results[k].labels;
      const agreements: number[] = [];
      for (const deltaL of [-1, 1]) {
        const pert = new Float32Array(processedLab);
        for (let i = 0; i < n; i++) {
          pert[i * 3] = Math.max(0, Math.min(100, pert[i * 3] + deltaL));
        }
        const { unary: pertUnary } = buildCellUnary(
          pert, processed.width, processed.height, palette, p.grid_cols, p.max_samples, p.trim_fraction,
        );
        const pertOut = await runEngine(k, pert, pertUnary);
        let match = 0;
        for (let i = 0; i < base.length; i++) if (pertOut.labels[i] === base[i]) match++;
        agreements.push((100 * match) / Math.max(1, base.length));
      }
      stabilityRows.push({
        Engine: ENGINE_NAMES[k],
        'Agreement after L* -1': agreements[0],
        'Agreement after L* +1': agreements[1],
        'Mean stability %': 0.5 * (agreements[0] + agreements[1]),
      });
    }
    fs.writeFileSync(path.join(outDir, 'stability_metrics.json'), JSON.stringify(stabilityRows, null, 2));
  }

  const configPayload = {
    schema: 'chromabook-advanced-mapping-v2',
    params: p,
    selected_engine: engineKey,
    run_stability: Boolean(includeStability),
    palette_ids: palette.map((c) => c.id),
  };
  fs.writeFileSync(path.join(outDir, 'config.json'), JSON.stringify(configPayload, null, 2));

  const summary = {
    name: path.basename(imagePath),
    grid: [geometry.cols, geometry.rows],
    colors_used: chosen.metrics.colors_used,
    engine: engineKey,
    confidence: 'not_computed',
    elapsed_s: Number(totalRuntimeSeconds.toFixed(2)),
    timings: { total: totalRuntimeSeconds },
    recommendation: {
      engine: engineKey,
      confidence: { level: 'not_computed' },
      metrics: chosen.metrics,
    },
  };
  fs.writeFileSync(path.join(outDir, 'asset_summary.json'), JSON.stringify(summary, null, 2));

  const renderPreflight = {
    passed: true,
    grid_rows: geometry.rows,
    grid_cols: geometry.cols,
    palette_colors: palette.length,
    colors_used: chosen.metrics.colors_used,
    numbered_cell_px: p.numbered_cell_px,
    numbered_line_width: p.numbered_line_width,
  };
  fs.writeFileSync(path.join(outDir, 'render_preflight.json'), JSON.stringify(renderPreflight, null, 2));

  return { engine: engineKey, confidence: 'not_computed', totalRuntimeSeconds };
}
