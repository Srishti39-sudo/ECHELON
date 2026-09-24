#!/usr/bin/env python3
"""Tests for the GhostTrace drift validation ("Model trust").

    .venv/bin/python tests_ghosttrace_validation.py

Plain functions and asserts with an exit code, like the other tests_*.py.
The metric maths is checked on SYNTHETIC tracks whose answers are known in
closed form; the forecast-scoring path runs drift_forecast over a SYNTHETIC
analytic current field (labelled so by ghosttrace.currents.AnalyticField). No
network. The real bundled summary, when present, is only checked for internal
consistency, never for particular numbers.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ghosttrace" / "tools"))

WORK = Path(tempfile.mkdtemp(prefix="deepecho-gt-validation-"))
os.environ["DEEPECHO_GHOSTTRACE_DATA_DIR"] = str(WORK / "ghosttrace")
os.environ["DEEPECHO_SURVEYS_DIR"] = str(WORK / "surveys")
(WORK / "surveys").mkdir(parents=True)

import numpy as np  # noqa: E402

from ghosttrace import validation as V  # noqa: E402

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def close(a, b, tol):
    return abs(a - b) <= tol


# --- metric maths -----------------------------------------------------------------------------


@case
def haversine_and_displace():
    assert close(V.haversine_km(0, 0, 1, 0), 111.195, 0.01)
    lat, lon = V.displace(10.0, 80.0, 5000.0, 0.0)
    assert close(V.haversine_km(10.0, 80.0, lat, lon), 5.0, 0.005)
    lat, lon = V.displace(10.0, 80.0, 0.0, -3000.0)
    assert close(V.haversine_km(10.0, 80.0, lat, lon), 3.0, 0.003) and lat < 10.0


@case
def separations_known_offset():
    obs = [V.displace(12.0, 85.0, 10000.0 * i, 0.0) for i in range(5)]
    pred = [V.displace(la, lo, 0.0, 5000.0) for la, lo in obs]
    seps = V.separations_km(pred, obs)
    assert all(close(s, 5.0, 0.01) for s in seps), seps
    try:
        V.separations_km(pred[:2], obs)
        raise AssertionError("length mismatch accepted")
    except ValueError:
        pass


@case
def liu_weisberg_closed_forms():
    # Observed: straight line, 10 km per step.
    obs = [V.displace(12.0, 85.0, 10000.0 * i, 0.0) for i in range(7)]
    assert close(V.liu_weisberg_skill(obs, obs), 1.0, 1e-9)
    # Persistence: d_i == l_i for a straight track, so s = 1 and ss = 0.
    persist = [obs[0]] * len(obs)
    assert close(V.liu_weisberg_skill(persist, obs), 0.0, 1e-3)
    # Half way along the track at every time: d_i = l_i / 2, so s = 0.5 and ss = 0.5.
    half = [V.displace(12.0, 85.0, 5000.0 * i, 0.0) for i in range(7)]
    assert close(V.liu_weisberg_skill(half, obs), 0.5, 2e-3), V.liu_weisberg_skill(half, obs)
    # Worse than persistence (s > 1) is clipped to 0.
    wrong = [V.displace(12.0, 85.0, -20000.0 * i, 0.0) for i in range(7)]
    assert V.liu_weisberg_skill(wrong, obs) == 0.0
    # A drifter that never moved: undefined.
    assert V.liu_weisberg_skill(persist, persist) is None


@case
def cone_membership():
    square = {"type": "Polygon", "coordinates": [[[85.0, 12.0], [85.1, 12.0], [85.1, 12.1], [85.0, 12.1], [85.0, 12.0]]]}
    assert V.point_in_geometry(square, 12.05, 85.05) is True
    assert V.point_in_geometry(square, 12.2, 85.05) is False
    assert V.point_in_geometry(None, 12.05, 85.05) is False
    multi = {"type": "MultiPolygon", "coordinates": [square["coordinates"],
                                                     [[[86.0, 12.0], [86.1, 12.0], [86.1, 12.1], [86.0, 12.0]]]]}
    assert V.point_in_geometry(multi, 12.02, 86.08) is True


@case
def quantiles_and_wilson():
    vals = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6]
    for q in (0.25, 0.5, 0.75):
        assert close(V.quantile(vals, q), float(np.quantile(vals, q)), 1e-12)
    assert V.quantile([], 0.5) is None
    lo, hi = V.wilson_interval(9, 10)
    assert close(lo, 0.5958, 1e-3) and close(hi, 0.9821, 1e-3), (lo, hi)
    assert V.wilson_interval(0, 0) is None
    lo, hi = V.wilson_interval(0, 20)
    assert lo == 0.0 and 0.1 < hi < 0.2


@case
def constant_velocity_baseline():
    track = V.constant_velocity_track(10.0, 80.0, 0.0, 1.0, [0, 1, 2])
    assert close(V.haversine_km(10.0, 80.0, *track[1]), 3.6, 0.005)
    assert close(V.haversine_km(10.0, 80.0, *track[2]), 7.2, 0.01)
    assert track[0] == [10.0, 80.0]


@case
def aggregate_known_coverage_and_reduction():
    rows = []
    # 10 segments at 24 h: model separations 1..10 km, persistence 10..19 km, 9 inside the 90% cone, 4 inside 50%.
    for i in range(10):
        rows.append({"horizons": {"24": {
            "sep_model_km": float(i + 1), "sep_persistence_km": float(i + 10), "sep_const_model_km": float(i + 5),
            "sep_const_obs_km": float(i + 2), "in50": i < 4, "in90": i < 9,
            "skill_model": 0.5, "skill_const_model": 0.2, "skill_const_obs": None}}})
    a = V.aggregate(rows, 24)
    assert a["n_segments"] == 10
    assert a["coverage_90"]["fraction"] == 0.9 and a["coverage_90"]["inside"] == 9
    assert a["coverage_50"]["fraction"] == 0.4
    assert a["separation_km_model"]["median"] == 5.5 and a["separation_km_persistence"]["median"] == 14.5
    assert close(a["model_vs_persistence"]["median_separation_reduction_pct"], 100 * (1 - 5.5 / 14.5), 0.1)
    assert a["model_vs_persistence"]["segments_model_closer"] == 10
    assert a["liu_weisberg_skill_constant_observed_velocity"]["n"] == 0
    assert a["coverage_90"]["ci95"][0] < 0.9 < a["coverage_90"]["ci95"][1]
    assert V.aggregate(rows, 48)["n_segments"] == 0


# --- the forecast scoring path on a synthetic field -----------------------------------------------


@case
def score_segment_on_synthetic_uniform_current():
    import validate_drift_gdp as T
    from ghosttrace.currents import AnalyticField

    start = datetime(2025, 3, 1, 0, tzinfo=timezone.utc)
    u, v = 0.25, -0.10
    field = AnalyticField(bounds=(8.0, 83.0, 16.0, 91.0), t_start=start - timedelta(hours=3), hours=80,
                          velocity=(u, v), label="SYNTHETIC uniform test current")
    lat0, lon0 = 12.0, 87.0
    obs = V.constant_velocity_track(lat0, lon0, u, v, [6 * i for i in range(13)])
    seg = {"drifter_id": "SYNTHETIC", "typebuoy": "TEST", "start": start, "track": obs, "ve0": u, "vn0": v,
           "drogue": "drogued"}
    fc = T.run_forecast(seg, field, horizon_h=72, n_particles=300, k=1.0, snapshot_every_h=6, seed=1)
    assert fc["available"] and fc["synthetic_currents"] is True
    scored = T.score_segment(seg, fc, field, (6, 12, 24, 48, 72))
    h72 = scored["horizons"]["72"]
    # Truth moves exactly with the field, so the ensemble mean is within random-walk noise of it.
    assert h72["sep_model_km"] < 0.5, h72
    assert close(h72["sep_persistence_km"], V.haversine_km(lat0, lon0, *obs[-1]), 1e-3)
    assert h72["sep_const_model_km"] < 0.01 and h72["sep_const_obs_km"] < 0.01, h72
    assert h72["in90"] is True and h72["skill_model"] > 0.95, h72
    assert h72["particles_left_box"] == 0
    # The engine default is untouched by the per-run K.
    from ghosttrace import config_geo as cfg
    before = cfg.DRIFT_K_SURFACE_M2S
    T.run_forecast(seg, field, horizon_h=12, n_particles=50, k=999.0, snapshot_every_h=6, seed=1)
    assert cfg.DRIFT_K_SURFACE_M2S == before


@case
def score_segment_off_track_is_outside_cone():
    import validate_drift_gdp as T
    from ghosttrace.currents import AnalyticField

    start = datetime(2025, 3, 1, 0, tzinfo=timezone.utc)
    field = AnalyticField(bounds=(8.0, 83.0, 16.0, 91.0), t_start=start - timedelta(hours=3), hours=80,
                          velocity=(0.0, 0.0), label="SYNTHETIC still water")
    obs = [V.displace(12.0, 87.0, 3000.0 * i, 0.0) for i in range(13)]  # truth moves east 3 km / 6 h
    seg = {"drifter_id": "SYNTHETIC", "typebuoy": "TEST", "start": start, "track": [list(p) for p in obs],
           "ve0": 0.0, "vn0": 0.0, "drogue": "undrogued"}
    fc = T.run_forecast(seg, field, horizon_h=72, n_particles=300, k=1.0, snapshot_every_h=6, seed=2)
    scored = T.score_segment(seg, fc, field, (24, 72))
    h = scored["horizons"]["72"]
    assert close(h["sep_model_km"], 36.0, 0.5) and h["in90"] is False and h["in50"] is False, h
    assert close(h["sep_model_km"], h["sep_persistence_km"], 0.5)


@case
def drogue_status_rules():
    import validate_drift_gdp as T

    s = datetime(2025, 1, 10, tzinfo=timezone.utc)
    e = s + timedelta(hours=72)
    assert T.drogue_status("", s, e) == "drogued"
    assert T.drogue_status("1970-01-01T00:00:00Z", s, e) == "uncertain"
    assert T.drogue_status("2024-12-01T00:00:00Z", s, e) == "undrogued"
    assert T.drogue_status("2025-01-11T00:00:00Z", s, e) == "uncertain"
    assert T.drogue_status("2025-02-01T00:00:00Z", s, e) == "drogued"


@case
def cut_segments_needs_unbroken_record():
    import validate_drift_gdp as T

    t0 = datetime(2025, 2, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(40):
        if i == 5:
            continue  # a gap early on
        rows.append({"id": "A", "t": t0 + timedelta(hours=6 * i), "lat": 10 + 0.01 * i, "lon": 85.0, "ve": 0.1,
                     "vn": 0.0, "drogue_lost": "", "typebuoy": "SVPB"})
    segs = T.cut_segments(rows, horizon_h=72, per_drifter=5, gap_days=10, max_segments=10)
    assert segs and all(len(s["track"]) == 13 for s in segs)
    assert segs[0]["start"] == t0 + timedelta(hours=36), segs[0]["start"]
    gaps = [(b["start"] - a["start"]).total_seconds() / 86400 for a, b in zip(segs, segs[1:])]
    assert all(g >= 10 for g in gaps), gaps


# --- summary loading and API ------------------------------------------------------------------------


def write_fake_summary(directory: Path) -> dict:
    (directory / "segments").mkdir(parents=True, exist_ok=True)
    seg = {"format": V.FORMAT + "+segment", "index": 0, "observed": [[12.0, 87.0], [12.1, 87.1]], "simulated": False}
    (directory / "segments" / "seg_000.json").write_text(json.dumps(seg))
    summary = {"format": V.FORMAT, "simulated": False, "headline": {"n_segments": 1},
               "segments": [{"index": 0, "file": "seg_000.json"}, {"index": 1, "file": "../escape.json"}]}
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "escape.json").write_text("{}")
    return summary


@case
def summary_and_segment_loading():
    d = WORK / "load"
    assert V.load_summary(d) is None
    write_fake_summary(d)
    assert V.load_summary(d)["headline"]["n_segments"] == 1
    assert V.load_segment(0, d)["observed"][0] == [12.0, 87.0]
    assert V.load_segment(1, d) is None          # a file name with a path is refused
    assert V.load_segment(2, d) is None and V.load_segment(-1, d) is None
    (d / "summary.json").write_text(json.dumps({"format": "something-else"}))
    assert V.load_summary(d) is None


@case
def api_routes():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from ghosttrace.routes import api as gt

    app = FastAPI()
    app.include_router(gt.router)
    client = TestClient(app)
    r = client.get("/ghosttrace/validation")
    assert r.status_code == 404 and "validate_drift_gdp" in r.json()["detail"], r.text
    write_fake_summary(gt.VALIDATION_DIR)
    r = client.get("/ghosttrace/validation")
    assert r.status_code == 200 and r.json()["format"] == V.FORMAT
    assert client.get("/ghosttrace/validation/segments/0").json()["index"] == 0
    assert client.get("/ghosttrace/validation/segments/1").status_code == 404
    assert client.get("/ghosttrace/validation/segments/9").status_code == 404
    assert client.get("/ghosttrace/validation/segments/abc").status_code == 422


@case
def bundled_summary_is_consistent():
    """When the real validation has been run, its summary must agree with itself and with its files."""
    real = ROOT / "data" / "ghosttrace" / "validation"
    summary = V.load_summary(real)
    if summary is None:
        print("  note  no bundled validation summary; run tools/validate_drift_gdp.py")
        return
    assert summary["simulated"] is False
    segs = summary["segments"]
    assert summary["counts"]["segments"] == len(segs) == summary["headline"]["n_segments"]
    assert len({s["drifter_id"] for s in segs}) == summary["counts"]["drifters"]
    for s in segs:
        assert (real / "segments" / s["file"]).is_file(), s["file"]
    by_h = {m["horizon_hours"]: m for m in summary["metrics"]["all"]}
    for h, m in by_h.items():
        flags = [s["horizons"][str(h)]["in90"] for s in segs if str(h) in s["horizons"]]
        assert m["coverage_90"]["inside"] == sum(flags) and m["coverage_90"]["n"] == len(flags)
        seps = [s["horizons"][str(h)]["sep_model_km"] for s in segs if str(h) in s["horizons"]]
        assert close(m["separation_km_model"]["median"], V.quantile(seps, 0.5), 0.01)
    size = sum(p.stat().st_size for p in [real / "summary.json", *(real / "segments").glob("*.json")])
    assert size < 5_000_000, f"bundled validation is {size / 1e6:.2f} MB"
    cal = summary.get("calibration")
    if cal:
        assert cal["applied"] is False
        from ghosttrace import config_geo as cfg
        assert cal["current_default_k_surface_m2s"] == float(os.environ.get("GHOSTTRACE_DRIFT_K_SURFACE_M2S", "1.0")) \
            or cal["current_default_k_surface_m2s"] == cfg.DRIFT_K_SURFACE_M2S


def main() -> int:
    failures = 0
    try:
        for fn in CASES:
            try:
                fn()
                print(f"  ok    {fn.__name__}")
            except Exception:
                failures += 1
                print(f"  FAIL  {fn.__name__}")
                traceback.print_exc()
    finally:
        shutil.rmtree(WORK, ignore_errors=True)
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
