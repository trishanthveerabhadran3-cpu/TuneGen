"""Utility evaluator (Assignment 3.2).

U(i) = sum_k w_k * x_k(i) - LAMBDA * warranty_risk(i)
x_perf, x_eff : min-max normalised over the feasible pool (weather adjusted hp gain)
x_cost        : 1 - price / budget
x_rel, x_trust: rating / 5
x_deliv       : 1 - delivery_days / MAX_DAYS   (clipped to [0, 1])
Hard constraints discard an option BEFORE scoring.
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Optional, Tuple

NUM_FIELDS = ("price_inr", "hp_gain", "mileage_gain", "reliability", "seller_rating", "delivery_days", "warranty_risk", "aggressiveness")


def validate(p: Dict[str, Any]) -> Optional[str]:
    for f in NUM_FIELDS:
        v = p.get(f)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or math.isnan(v) or math.isinf(v):
            return f"invalid {f}={v!r}"
    if p["price_inr"] <= 0:
        return "non-positive price"
    if not (0 <= p["reliability"] <= 5 and 0 <= p["seller_rating"] <= 5):
        return "rating outside 0..5"
    if not (0 <= p["warranty_risk"] <= 1 and 0 <= p["aggressiveness"] <= 1):
        return "risk or aggressiveness outside 0..1"
    if p["delivery_days"] < 0:
        return "negative delivery_days"
    return None


def reject_reason(p: Dict[str, Any], c: Dict[str, Any]) -> Optional[str]:
    """c = {budget, car_id, max_aggr, avoid_tags, warranty_active}"""
    r = validate(p)
    if r:
        return r
    if c["car_id"] not in p.get("fits", []):
        return "does not fit this car"
    if not p.get("bs6_legal", False):
        return "not BS6 / road legal"
    if p["price_inr"] > c["budget"]:
        return f"over budget ({p['price_inr']:.0f} > {c['budget']:.0f})"
    if p["aggressiveness"] > c["max_aggr"]:
        return f"too aggressive for current conditions ({p['aggressiveness']} > {c['max_aggr']})"
    hit = set(p["tags"]) & set(c["avoid_tags"])
    if hit:
        return f"avoided by live conditions: {sorted(hit)}"
    if c["warranty_active"] and p["warranty_risk"] > 0.25:
        return f"may void warranty (risk {p['warranty_risk']})"
    return None


def adjusted_gain(p: Dict[str, Any], multipliers: Dict[str, float]) -> float:
    g = p["hp_gain"]
    for t in p["tags"]:
        g *= multipliers.get(t, 1.0)
    return g


def _mm(v: float, lo: float, hi: float) -> float:
    return 0.5 if hi == lo else (v - lo) / (hi - lo)


def normalise_weights(w: Dict[str, float]) -> Dict[str, float]:
    tot = sum(max(v, 0.0) for v in w.values())
    if tot <= 0:
        raise ValueError("weights need at least one positive value")
    return {k: max(v, 0.0) / tot for k, v in w.items()}


def score_all(pool: List[Dict[str, Any]], budget: float, weights: Dict[str, float],
              multipliers: Dict[str, float], lam: float, max_days: int) -> List[Dict[str, Any]]:
    if not pool:
        return []
    w = normalise_weights(weights)
    gains = {p["id"]: adjusted_gain(p, multipliers) for p in pool}
    g_lo, g_hi = min(gains.values()), max(gains.values())
    m_lo, m_hi = min(p["mileage_gain"] for p in pool), max(p["mileage_gain"] for p in pool)
    out = []
    for p in pool:
        x = {"perf": _mm(gains[p["id"]], g_lo, g_hi), "eff": _mm(p["mileage_gain"], m_lo, m_hi),
             "cost": max(0.0, 1 - p["price_inr"] / budget), "rel": p["reliability"] / 5,
             "trust": p["seller_rating"] / 5, "deliv": max(0.0, 1 - p["delivery_days"] / max_days)}
        u = sum(w[k] * x[k] for k in w) - lam * p["warranty_risk"]
        out.append({"part": p, "adj_gain": round(gains[p["id"]], 3), "x": x, "U": u})
    out.sort(key=lambda s: (-s["U"], s["part"]["price_inr"], s["part"]["name"]))   # deterministic ties
    return out


def best_bundle(scored: List[Dict[str, Any]], budget: float) -> Tuple[List[Dict[str, Any]], float]:
    """Exhaustive search: at most one part per category, total price <= budget, maximise sum of U."""
    best, best_u = [], 0.0
    for r in range(1, len(scored) + 1):
        for combo in itertools.combinations(scored, r):
            cats = [s["part"]["category"] for s in combo]
            if len(set(cats)) != len(cats) or sum(s["part"]["price_inr"] for s in combo) > budget:
                continue
            u = sum(s["U"] for s in combo)
            if u > best_u + 1e-12:
                best, best_u = list(combo), u
    return best, best_u
