"""Which net first? A transparent, recomputable priority score.

A recovery vessel has a day, a crew and a winch. The score decides the order,
so it must be something a survey lead can argue with: every term, its weight
and its contribution is in the output, and the score is recomputed from them
by one line of arithmetic.

    score = confidence_factor * sum_i( weight_i * value_i )

    value_i            each term, clipped to 0..1
    weight_i           PRIORITY_WEIGHTS, summing to 1.0
    contribution_i     weight_i * value_i, rounded, as written in the output
    confidence_factor  confidence_pct / 100 (verification's fused score when
                       present, else the detector's confidence)

The multiplier is deliberate. Context terms describe what the object WOULD do
if it is what the detector says; the confidence says how likely that is. A
40% detection beside a reef must not outrank a 90% detection beside a reef,
and must never be urgent on context alone.

TERMS (values; unavailable inputs take PRIORITY_NEUTRAL and say so)
    activity        watercolumn score (still catching)
    habitat         habitat_context score (what it lies on or near now)
    drift_impact    max impact probability on a sensitive layer within the
                    forecast horizon (reef, turtle, dugong, protected area...)
    people_risk     propeller hazard level mapped by PROPELLER_LEVEL_VALUE
    size            1 - exp(-plan_area_m2 / SIZE_SCALE_M2), saturating
    change          moved 1.0 / new 0.7 / persistent 0.4 (mobile gear keeps
                    killing along its path)
    recoverability  mean of depth ease and current ease, a small bonus so
                    quick wins surface

Tiers from PRIORITY_TIERS. All of it is a configurable heuristic, labelled as
such in the output; none of it is fitted to recovery outcomes.
"""

from __future__ import annotations

import math
from typing import Any

from ghosttrace import config_core as cfg


def _clip(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _term(name: str, value: float | None, basis: str) -> dict[str, Any]:
    neutral = cfg.PRIORITY_NEUTRAL[name]
    used = neutral if value is None else _clip(value)
    if value is None:
        basis = f"{basis}; unavailable, neutral value {neutral} used"
    w = cfg.PRIORITY_WEIGHTS[name]
    return {"value": round(used, 4), "weight": w, "contribution": round(w * round(used, 4), 6),
            "measured": value is not None, "basis": basis}


def _linear_ease(x: float, easy: float, hard: float) -> float:
    if x <= easy:
        return 1.0
    if x >= hard:
        return 0.0
    return (hard - x) / (hard - easy)


def _find_number(obj: Any, keys: tuple[str, ...]) -> float | None:
    """First numeric value under any of `keys`, searched shallowly (2 levels)."""
    if not isinstance(obj, dict):
        return None
    for k in keys:
        v = obj.get(k)
        if isinstance(v, (int, float)) and math.isfinite(v):
            return float(v)
    for v in obj.values():
        if isinstance(v, dict):
            for k in keys:
                x = v.get(k)
                if isinstance(x, (int, float)) and math.isfinite(x):
                    return float(x)
    return None


def confidence_factor(target: dict[str, Any]) -> tuple[float, str]:
    pct = target.get("confidence_pct")
    if isinstance(pct, (int, float)):
        return _clip(pct / 100.0), target.get("confidence_basis") or "confidence_pct / 100"
    return 0.0, "no confidence recorded; factor 0"


def terms_for(target: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every priority term for one assembled target block."""
    activity = target.get("activity") or {}
    habitat = target.get("habitat") or {}
    drift = target.get("drift") or {}
    people = target.get("people") or {}
    change = target.get("change") or {}
    dims = target.get("dimensions") or {}

    terms: dict[str, dict[str, Any]] = {}

    a = activity.get("score") if activity.get("available") else None
    terms["activity"] = _term("activity", a, "water-column echo enrichment score")

    h = None
    hb = "habitat_context score"
    if habitat.get("available", True) and isinstance(habitat.get("score"), (int, float)):
        if habitat.get("covered") is False:
            hb += " (position outside the bundled habitat layers' coverage)"
        else:
            h = float(habitat["score"])
    terms["habitat"] = _term("habitat", h, hb)

    d = None
    basis = "max drift impact probability on a sensitive layer"
    if drift and drift.get("available", True) and isinstance(drift.get("impacts"), list):
        probs = [float(i.get("probability") or 0.0) for i in drift["impacts"]
                 if any(k in normalize(i.get("kind")) for k in cfg.SENSITIVE_IMPACT_KINDS)
                 or any(k in normalize(i.get("name")) for k in cfg.SENSITIVE_IMPACT_KINDS)]
        d = max(probs) if probs else 0.0
        if not probs:
            basis += " (forecast ran; no sensitive layer reached)"
    terms["drift_impact"] = _term("drift_impact", d, basis)

    p = None
    level = None
    if people and people.get("available", True):
        level = str(((people.get("propeller_hazard") or {}).get("level")) or "").lower() or None
        if level in cfg.PROPELLER_LEVEL_VALUE:
            p = cfg.PROPELLER_LEVEL_VALUE[level]
    terms["people_risk"] = _term("people_risk", p,
                                 f"propeller hazard level {level!r} mapped "
                                 f"{cfg.PROPELLER_LEVEL_VALUE}")

    s = None
    if isinstance(dims, dict) and dims.get("length_m") and dims.get("width_m"):
        try:
            area = float(dims["length_m"]) * float(dims["width_m"])
            s = 1.0 - math.exp(-area / cfg.SIZE_SCALE_M2)
            sb = f"1 - exp(-{area:.1f} m2 / {cfg.SIZE_SCALE_M2:g})"
        except (TypeError, ValueError):
            sb = "dimensions unreadable"
    else:
        sb = "plan area from dimensions.length_m * width_m"
    terms["size"] = _term("size", s, sb)

    status = change.get("status")
    terms["change"] = _term("change", cfg.CHANGE_VALUE.get(status),
                            f"change status {status!r} mapped {cfg.CHANGE_VALUE}")

    depth = target.get("seabed_depth_m")
    if depth is None:
        depth = _find_number(people, ("depth_m", "seabed_depth_m", "water_depth_m"))
    current = _find_number(people, ("current_speed_ms", "current_ms", "current_mps"))
    if current is None:
        current = _find_number(drift, ("current_speed_ms", "mean_current_ms",
                                       "mean_speed_ms", "current_mps"))
    parts, notes = [], []
    if depth is not None:
        parts.append(_linear_ease(abs(depth), cfg.RECOVER_DEPTH_EASY_M, cfg.RECOVER_DEPTH_HARD_M))
        notes.append(f"depth {abs(depth):.1f} m")
    if current is not None:
        parts.append(_linear_ease(current, cfg.RECOVER_CURRENT_EASY_MS,
                                  cfg.RECOVER_CURRENT_HARD_MS))
        notes.append(f"current {current:.2f} m/s")
    terms["recoverability"] = _term(
        "recoverability", sum(parts) / len(parts) if parts else None,
        "mean of depth ease (1 at <= %g m, 0 at >= %g m) and current ease (1 at <= %g m/s, "
        "0 at >= %g m/s)%s" % (cfg.RECOVER_DEPTH_EASY_M, cfg.RECOVER_DEPTH_HARD_M,
                               cfg.RECOVER_CURRENT_EASY_MS, cfg.RECOVER_CURRENT_HARD_MS,
                               (": " + ", ".join(notes)) if notes else ""))
    return terms


def normalize(text: Any) -> str:
    return str(text or "").strip().lower().replace(" ", "-")


def tier_for(score: float) -> str:
    for name, floor in cfg.PRIORITY_TIERS:
        if score >= floor:
            return name
    return cfg.PRIORITY_TIERS[-1][0]


def score_target(target: dict[str, Any]) -> dict[str, Any]:
    """Priority block (rank filled later by rank_targets)."""
    terms = terms_for(target)
    factor, fbasis = confidence_factor(target)
    factor = round(factor, 4)
    weighted = sum(t["contribution"] for t in terms.values())
    score = round(factor * weighted, 4)
    terms["confidence"] = {"value": round(factor, 4), "weight": None, "contribution": None,
                           "role": "multiplier", "basis": fbasis}
    return {
        "score": score,
        "rank": None,
        "tier": tier_for(score),
        "terms": terms,
        "formula": "score = confidence.value * sum(weight * value for every other term) "
                   "(= confidence.value * sum(contribution))",
        "weighted_sum": round(weighted, 6),
        "tiers": {name: floor for name, floor in cfg.PRIORITY_TIERS},
        "basis": cfg.HEURISTIC_LABEL,
    }


def rank_targets(targets: list[dict[str, Any]]) -> None:
    """Assign rank 1..n in place: score descending, then detection_id."""
    ordered = sorted(targets, key=lambda t: (-(t.get("priority") or {}).get("score", 0.0),
                                             str(t.get("detection_id"))))
    for i, t in enumerate(ordered, start=1):
        t["priority"]["rank"] = i


def recompute(priority: dict[str, Any]) -> float:
    """Recompute a score from its own terms, as a reader of the JSON would."""
    terms = priority["terms"]
    s = sum(t["contribution"] for k, t in terms.items() if k != "confidence")
    return terms["confidence"]["value"] * s
