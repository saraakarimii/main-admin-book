"""Scoring: hard constraints first, then normalised multi-metric ranking inside a COMPARABLE group
(same framing, same grid). Candidates at different grid sizes are never ranked on one scale."""
from __future__ import annotations
import numpy as np
from ..color import rgb_to_lab

# metric -> (direction, weight, tie floor). Spread below the floor counts as a tie, so tiny
# differences cannot masquerade as a decisive win.
METRICS = {
    "DeltaE00_mean_vs_source": ("min", 0.35, 0.30),
    "DeltaE00_p95_vs_source": ("min", 0.15, 1.00),
    "cell_structure_f1": ("max", 0.20, 0.02),
    "singleton_ratio_pct": ("min", 0.15, 1.00),
    "transition_gap_pp": ("min", 0.15, 2.00),
}


def cell_structure(reference_rgb, labels, geo, palette_lab, thr=10.0):
    """Mosaic-aware structure check at CELL level (no pixel-edge artefacts from cell borders).
    Source: dE76 between neighbouring cell-mean Lab > thr.  Mapped: neighbouring labels differ."""
    lab = rgb_to_lab(reference_rgb)
    means = np.zeros((geo.rows, geo.cols, 3), np.float32)
    for r in range(geo.rows):
        for c in range(geo.cols):
            means[r, c] = lab[geo.ye[r]:geo.ye[r + 1], geo.xe[c]:geo.xe[c + 1]].reshape(-1, 3).mean(0)
    if not np.isfinite(means).all():
        raise ValueError("empty grid cell at this working resolution; raise the screening resolution")
    sh = np.linalg.norm(means[:, 1:] - means[:, :-1], axis=2) > thr
    sv = np.linalg.norm(means[1:] - means[:-1], axis=2) > thr
    mh, mv = labels[:, 1:] != labels[:, :-1], labels[1:] != labels[:-1]
    src = np.concatenate([sh.ravel(), sv.ravel()]); mp = np.concatenate([mh.ravel(), mv.ravel()])
    tp = float(np.sum(src & mp)); p = tp / max(1, mp.sum()); r = tp / max(1, src.sum())
    f1 = 2 * p * r / max(1e-12, p + r)
    return {"cell_structure_f1": f1, "source_transition_rate_pct": 100.0 * float(src.mean()),
            "transition_gap_pp": abs(100.0 * float(mp.mean()) - 100.0 * float(src.mean()))}


def rank_group(rows: list[dict]) -> list[dict]:
    """rows: candidate dicts with a 'metrics' dict. Adds 'norm' (0 best..1 worst per metric) and 'score' (0..100)."""
    if not rows: return rows
    for name, (direction, w, floor) in METRICS.items():
        vals = np.array([r["metrics"][name] for r in rows], float)
        best = vals.min() if direction == "min" else vals.max()
        dead = 0.5 * floor                      # dead band: gaps this small are noise and cost nothing
        denom = max(float(vals.max() - vals.min()) - dead, floor)
        for r, v in zip(rows, vals):
            gap = (v - best) if direction == "min" else (best - v)
            r.setdefault("norm", {})[name] = float(np.clip(max(gap - dead, 0.0) / denom, 0, 1))
    for r in rows:
        r["score"] = round(100.0 * (1.0 - sum(w * r["norm"][n] for n, (_, w, _) in METRICS.items())), 2)
    return sorted(rows, key=lambda r: (-r["score"], r["metrics"]["DeltaE00_mean_vs_source"]))


def confidence(ranked: list[dict], stability_pct: float | None) -> dict:
    """Confidence = how decisively the top candidate wins, not how pretty it is.
    Components: score margin to #2, how many metrics rank #1 first or second, stability under +/-1 L*."""
    if len(ranked) < 2:
        return {"level": "low", "value": 30, "reasons": ["only one candidate survived the constraints"]}
    a, b = ranked[0], ranked[1]
    margin = a["score"] - b["score"]
    names = list(METRICS)
    agree = np.mean([a["norm"][n] <= b["norm"][n] + 1e-9 for n in names])
    v = 0.0; reasons = []
    v += 50 * float(np.clip(margin / 12.0, 0, 1)); reasons.append(f"score margin to runner-up {margin:.1f} pts")
    v += 30 * float(agree); reasons.append(f"{int(round(agree * len(names)))}/{len(names)} metrics favour the winner")
    if stability_pct is not None:
        v += 20 * float(np.clip((stability_pct - 85) / 12.0, 0, 1)); reasons.append(f"+/-1 L* stability {stability_pct:.1f} %")
    else:
        v += 10; reasons.append("stability not measured")
    level = "high" if v >= 70 else "medium" if v >= 45 else "low"
    return {"level": level, "value": int(round(v)), "reasons": reasons}
