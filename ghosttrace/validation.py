"""Model trust: how GhostTrace's floating drift compares with real ocean drifters.

tools/validate_drift_gdp.py starts a floating-mode forecast from the real
position and time of a NOAA Global Drifter Program (GDP) surface drifter, runs
it over HYCOM surface currents for the same dates, and compares the forecast
with where the drifter actually went. It writes

    data/ghosttrace/validation/summary.json          metrics + per-segment rows
    data/ghosttrace/validation/segments/seg_NNN.json one segment's tracks and cones

This module holds the metric maths (so tests exercise exactly what the tool
uses) and reads those files for the API and the interface. It never computes
a result at request time and never invents one: with no summary on disk the
API says validation has not been run.

METRICS
    separation_km        geodesic distance between a predicted position and the
                         observed drifter position at the same elapsed time
    inside cone          whether the observed position lies inside the forecast's
                         50% / 90% probability cone; the fraction of segments
                         for which it does is the EMPIRICAL COVERAGE. A
                         well-calibrated 90% cone contains the truth about 90%
                         of the time. Much less means the cone is overconfident
                         (too narrow); much more means it is too wide.
    Liu-Weisberg skill   Liu, Y. and Weisberg, R. H. (2011), "Evaluation of
                         trajectory modeling in different dynamic regions using
                         normalized cumulative Lagrangian separation", Journal
                         of Geophysical Research: Oceans 116, C09013,
                         doi:10.1029/2010JC006837.
                             s  = sum_i d_i / sum_i l_i
                             ss = 1 - s / n   if s <= n, else 0   (n = 1)
                         d_i is the separation at observation time i and l_i
                         the length of the observed trajectory from the start
                         to time i. 1 is a perfect track; 0 means the
                         cumulative separation is at least the cumulative
                         distance travelled.
    baselines            persistence (the object does not move), the model
                         current at the start held constant, and the drifter's
                         own observed velocity at the start held constant.

Wilson score intervals (Wilson 1927, JASA 22:209-212) are reported for every
coverage fraction so a small sample is not read as a precise number.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

from ghosttrace import config_geo as cfg

VALIDATION_DIR = Path(os.environ.get("GHOSTTRACE_VALIDATION_DIR", str(cfg.DATA_DIR / "validation")))
SUMMARY_FILE = "summary.json"
SEGMENTS_DIR = "segments"
FORMAT = "deepecho-ghosttrace-validation/1"

LIU_WEISBERG_CITATION = (
    "Liu, Y. and Weisberg, R. H. (2011). Evaluation of trajectory modeling in different dynamic regions "
    "using normalized cumulative Lagrangian separation. Journal of Geophysical Research: Oceans 116, "
    "C09013. doi:10.1029/2010JC006837")

_R_EARTH_M = 6371008.8


# --- geometry ------------------------------------------------------------------------------


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km on a sphere of the mean Earth radius (error < 0.5%)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _R_EARTH_M * math.asin(min(1.0, math.sqrt(h))) / 1000.0


def displace(lat: float, lon: float, east_m: float, north_m: float) -> tuple[float, float]:
    """(lat, lon) after a local east/north displacement in metres (flat-earth, fine below ~500 km)."""
    dlat = math.degrees(north_m / _R_EARTH_M)
    dlon = math.degrees(east_m / (_R_EARTH_M * max(math.cos(math.radians(lat)), 1e-6)))
    return lat + dlat, lon + dlon


def point_in_geometry(geometry: dict[str, Any] | None, lat: float, lon: float) -> bool:
    """True when (lat, lon) lies inside a GeoJSON Polygon / MultiPolygon. None is never inside."""
    if not geometry:
        return False
    from shapely.geometry import Point, shape

    try:
        return bool(shape(geometry).covers(Point(lon, lat)))
    except (ValueError, TypeError, AttributeError):
        return False


# --- trajectory metrics --------------------------------------------------------------------


def separations_km(pred: Sequence[Sequence[float]], obs: Sequence[Sequence[float]]) -> list[float]:
    """Pairwise separation of two equally timed [lat, lon] tracks."""
    if len(pred) != len(obs):
        raise ValueError(f"tracks differ in length ({len(pred)} vs {len(obs)})")
    return [haversine_km(p[0], p[1], o[0], o[1]) for p, o in zip(pred, obs)]


def liu_weisberg_skill(pred: Sequence[Sequence[float]], obs: Sequence[Sequence[float]],
                       tolerance: float = 1.0) -> float | None:
    """Liu & Weisberg (2011) skill score over tracks that both START at index 0.

    pred and obs are [lat, lon] lists at the same times; index 0 is the common
    start. Returns None when the observed drifter did not move (the score is
    undefined when the cumulative observed length is zero).
    """
    if len(pred) != len(obs):
        raise ValueError(f"tracks differ in length ({len(pred)} vs {len(obs)})")
    if len(obs) < 2:
        return None
    seg = [haversine_km(obs[i - 1][0], obs[i - 1][1], obs[i][0], obs[i][1]) for i in range(1, len(obs))]
    cum = 0.0
    sum_l = 0.0
    sum_d = 0.0
    for i in range(1, len(obs)):
        cum += seg[i - 1]
        sum_l += cum
        sum_d += haversine_km(pred[i][0], pred[i][1], obs[i][0], obs[i][1])
    if sum_l <= 0:
        return None
    s = sum_d / sum_l
    return 0.0 if s > tolerance else 1.0 - s / tolerance


def constant_velocity_track(lat: float, lon: float, u_mps: float, v_mps: float,
                            hours: Iterable[float]) -> list[list[float]]:
    """Positions at each elapsed hour for a velocity held constant from the start."""
    out = []
    for h in hours:
        la, lo = displace(lat, lon, u_mps * h * 3600.0, v_mps * h * 3600.0)
        out.append([la, lo])
    return out


# --- aggregation -----------------------------------------------------------------------------


def quantile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated quantile (numpy's default), None for an empty list."""
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vals:
        return None
    pos = (len(vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float] | None:
    """95% Wilson score interval for a binomial proportion, or None when n == 0."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _r(x: float | None, nd: int = 3) -> float | None:
    return None if x is None else round(float(x), nd)


def aggregate(rows: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    """Pool per-segment results at one horizon.

    Each row carries, at key str(horizon): sep_model_km, sep_persistence_km,
    sep_const_model_km, sep_const_obs_km, in50, in90, skill_model,
    skill_const_model, skill_const_obs.
    """
    at = [r["horizons"][str(horizon)] for r in rows if str(horizon) in (r.get("horizons") or {})]
    n = len(at)

    def col(key):
        return [a[key] for a in at if a.get(key) is not None and math.isfinite(a[key])]

    out: dict[str, Any] = {"horizon_hours": horizon, "n_segments": n}
    for key, label in (("sep_model_km", "model"), ("sep_persistence_km", "persistence"),
                       ("sep_const_model_km", "constant_model_current"),
                       ("sep_const_obs_km", "constant_observed_velocity")):
        vals = col(key)
        out[f"separation_km_{label}"] = {"median": _r(quantile(vals, 0.5), 2), "p25": _r(quantile(vals, 0.25), 2),
                                         "p75": _r(quantile(vals, 0.75), 2), "n": len(vals)}
    for cone in ("in50", "in90"):
        flags = [bool(a[cone]) for a in at if a.get(cone) is not None]
        k = sum(flags)
        ci = wilson_interval(k, len(flags))
        out[f"coverage_{cone[2:]}"] = {"fraction": _r(k / len(flags)) if flags else None, "inside": k,
                                        "n": len(flags), "ci95": [_r(ci[0]), _r(ci[1])] if ci else None}
    for key, label in (("skill_model", "model"), ("skill_const_model", "constant_model_current"),
                       ("skill_const_obs", "constant_observed_velocity")):
        vals = col(key)
        out[f"liu_weisberg_skill_{label}"] = {"median": _r(quantile(vals, 0.5)), "mean": _r(sum(vals) / len(vals)) if vals else None,
                                              "n": len(vals)}
    pairs = [(a["sep_model_km"], a["sep_persistence_km"]) for a in at
             if a.get("sep_model_km") is not None and a.get("sep_persistence_km") is not None]
    med_m = quantile([p[0] for p in pairs], 0.5)
    med_p = quantile([p[1] for p in pairs], 0.5)
    out["model_vs_persistence"] = {
        "segments_model_closer": sum(1 for m, p in pairs if m < p), "n": len(pairs),
        "median_separation_reduction_pct": _r(100.0 * (1 - med_m / med_p), 1) if med_m is not None and med_p else None,
        "basis": "1 - median(model separation) / median(persistence separation), same segments",
    }
    pairs_c = [(a["sep_model_km"], a["sep_const_model_km"]) for a in at
               if a.get("sep_model_km") is not None and a.get("sep_const_model_km") is not None]
    med_c = quantile([p[1] for p in pairs_c], 0.5)
    out["model_vs_constant_model_current"] = {
        "segments_model_closer": sum(1 for m, c in pairs_c if m < c), "n": len(pairs_c),
        "median_separation_reduction_pct": _r(100.0 * (1 - quantile([p[0] for p in pairs_c], 0.5) / med_c), 1)
        if pairs_c and med_c else None,
    }
    pairs_o = [(a["sep_model_km"], a["sep_const_obs_km"]) for a in at
               if a.get("sep_model_km") is not None and a.get("sep_const_obs_km") is not None]
    med_o = quantile([p[1] for p in pairs_o], 0.5)
    out["model_vs_constant_observed_velocity"] = {
        "segments_model_closer": sum(1 for m, o in pairs_o if m < o), "n": len(pairs_o),
        "median_separation_reduction_pct": _r(100.0 * (1 - quantile([p[0] for p in pairs_o], 0.5) / med_o), 1)
        if pairs_o and med_o else None,
    }
    return out


# --- reading the bundled result --------------------------------------------------------------


def summary_path(directory: Path | None = None) -> Path:
    return Path(directory or VALIDATION_DIR) / SUMMARY_FILE


def load_summary(directory: Path | None = None) -> dict[str, Any] | None:
    """summary.json as written by tools/validate_drift_gdp.py, or None when absent/unreadable."""
    path = summary_path(directory)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("format") == FORMAT else None


def load_segment(index: int, directory: Path | None = None) -> dict[str, Any] | None:
    """One segment's display file, by its index in summary.segments, or None."""
    if not isinstance(index, int) or index < 0:
        return None
    summary = load_summary(directory)
    if summary is None:
        return None
    segments = summary.get("segments") or []
    if index >= len(segments):
        return None
    name = segments[index].get("file")
    if not isinstance(name, str) or "/" in name or "\\" in name or name.startswith("."):
        return None
    path = Path(directory or VALIDATION_DIR) / SEGMENTS_DIR / name
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
