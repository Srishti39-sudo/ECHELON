#!/usr/bin/env python3
"""Tests for the GhostTrace decision core.

    .venv/bin/python tests_ghosttrace_core.py
    .venv/bin/python tests_ghosttrace_core.py --verbose
    .venv/bin/python tests_ghosttrace_core.py --write-fixture   # also refresh
                                                                # tests/fixtures/ghosttrace_example.json

Plain functions and asserts, exit code 1 on any failure, matching the rest of
the repository (unit_tests.py, tests_jobs.py).

Everything here is SYNTHETIC. The survey directories, navigation sidecars,
water-column arrays and the geo stages (habitat, drift, safety) are built or
faked inside this file so the core can be tested before, and independently of,
the geo agent's modules and any real sonar. No network, no model.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from ghosttrace import alerts, changes, config_core as cfg, priority, recovery, watercolumn  # noqa: E402
from ghosttrace.engine import run_ghosttrace, to_geojson  # noqa: E402
from survey_hazard_map.hazard_geo import offset_latlon  # noqa: E402

LAT0, LON0 = 11.60, 92.70
M_ALONG, M_ACROSS, NADIR, WIDTH, ROWS, BINS, M_BIN = 0.25, 0.1, 500.0, 1000, 4000, 120, 0.125

# Rows of deterministic background echo blobs, and the rows of the aggregation
# injected beside detection D1. The quiet probe sits at row 3300, clear of both.
BACKGROUND_BLOB_ROWS = [150, 900, 1300, 1900, 2300, 2700, 3100, 3500, 3900]
AGGREGATION_ROWS = [340, 360, 385, 400, 425, 445, 470, 490]


# --- synthetic survey builders ------------------------------------------------------

def pos(row: float, x: float) -> tuple[float, float]:
    """lat/lon of pixel (x, row) on a due-north track, port on the left."""
    lat_n, lon_n = offset_latlon(LAT0, LON0, 0.0, row * M_ALONG)
    return offset_latlon(lat_n, lon_n, (x - NADIR) * M_ACROSS, 0.0)


def det(ident, cls, row, x, *, conf=0.8, pct=None, dims=None, suppressed=None, strip="line1",
        geo=True, h=20):
    lat, lon = pos(row, x) if geo else (None, None)
    d = {"id": ident, "object_class": cls, "class_normalized": cls, "confidence": conf,
         "severity": 0.5 * conf, "severity_tier": "medium", "recommended_action": "Schedule cleanup",
         "latitude": None if lat is None else round(lat, 8),
         "longitude": None if lon is None else round(lon, 8),
         "global_x": float(x), "global_y": float(row),
         "bbox_global": [x - 10.0, row - h / 2, x + 10.0, row + h / 2],
         "width_px": 20.0, "height_px": float(h), "provenance": {"strip": strip}}
    if pct is not None:
        d["confidence_pct"] = pct
    if dims is not None:
        d["dimensions"] = dims
    if suppressed is not None:
        d["suppressed"] = suppressed
    return d


def write_water_column(nav_dir: Path, strip: str, aggregation: bool = True) -> None:
    rng = np.random.default_rng(7)
    port = rng.rayleigh(1.0, size=(ROWS, BINS)).astype(np.float32)
    star = rng.rayleigh(1.0, size=(ROWS, BINS)).astype(np.float32)
    bottom = np.full(ROWS, 12.5, dtype=np.float32)
    bottom_bin = int(12.5 / M_BIN)
    port[:, bottom_bin:] = np.nan
    star[:, bottom_bin:] = np.nan
    for r in BACKGROUND_BLOB_ROWS:
        port[r:r + 4, 40:44] = 12.0
        star[r:r + 4, 60:64] = 12.0
    if aggregation:
        for i, r in enumerate(AGGREGATION_ROWS):
            port[r:r + 5, 20 + 8 * i % 60:24 + 8 * i % 60] = 14.0
    np.savez(nav_dir / f"{strip}.wc.npz", port=port, starboard=star, bottom_range_m=bottom)


def write_nav(nav_dir: Path, strip: str, start="2026-09-01T04:30:00+00:00") -> None:
    from datetime import datetime, timedelta
    t0 = datetime.fromisoformat(start)
    rows = []
    for r in range(ROWS):
        lat, lon = pos(r + 0.5, NADIR)
        rows.append({"row": r, "lat": round(lat, 8), "lon": round(lon, 8), "heading_deg": 0.0,
                     "altitude_m": 10.0, "depth_m": 5.0, "quality": "ok",
                     "time": (t0 + timedelta(seconds=r * 0.1)).isoformat()})
    sidecar = {"format": "deepecho-strip-nav/1", "synthetic": True, "nadir_col": NADIR,
               "m_per_px_across": M_ACROSS, "m_per_px_along": M_ALONG, "port_is_left": True,
               "width": WIDTH, "height": ROWS, "rows": rows,
               "water_column": {"path": f"{strip}.wc.npz", "m_per_bin": M_BIN,
                                "max_range_m": BINS * M_BIN}}
    (nav_dir / f"{strip}.nav.json").write_text(json.dumps(sidecar), encoding="utf-8")


def tiles_for(strip: str, geo=True) -> list[dict]:
    out = []
    for y in range(250, ROWS, 500):
        for x in (100, 500, 900):
            lat, lon = pos(y, x) if geo else (None, None)
            out.append({"tile": f"{strip}_{x}_{y}.jpg", "strip": strip, "x": x, "y": y,
                        "lat": lat, "lon": lon, "center_x": x, "center_y": y})
    return out


def write_survey(root: Path, name: str, detections: list[dict], *, processed_at: str,
                 nav=True, wc=True, aggregation=True, geo=True) -> Path:
    d = root / name
    (d / "nav").mkdir(parents=True, exist_ok=True)
    navigation = {"mode": "none"}
    if nav:
        write_nav(d / "nav", "line1")
        navigation = {"mode": "ping", "sidecars": {"line1": "nav/line1.nav.json"}}
        if wc:
            write_water_column(d / "nav", "line1", aggregation)
            navigation["water_columns"] = {"line1": "nav/line1.wc.npz"}
    export = {"metadata": {"survey_id": name, "processed_at": processed_at, "demo": True,
                           "data_source": "SYNTHETIC TEST DATA (tests_ghosttrace_core.py)"},
              "survey_summary": {}, "detections": detections, "hotspots": []}
    manifest = {"survey": {"navigation": navigation, "strips": [{"strip": "line1", "width": WIDTH,
                                                                  "height": ROWS}]},
                "tiles": tiles_for("line1", geo)}
    (d / "export.json").write_text(json.dumps(export, indent=1), encoding="utf-8")
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return d


def current_detections() -> list[dict]:
    return [
        det("S2_D1", "net", 415, 130, conf=0.9, pct=88.0,
            dims={"length_m": 12.0, "width_m": 4.0, "height_m": 1.5}),
        det("S2_D2", "net", 1510, 800, conf=0.7),
        det("S2_D3", "drum", 1000, 300, conf=0.6, pct=45.0,
            dims={"length_m": 0.9, "width_m": 0.6, "height_m": 0.6}),
        det("S2_D4", "ship", 2000, 700, conf=0.8),                # not a GhostTrace class
        det("S2_D5", "net", 2400, 200, conf=0.5, pct=20.0, suppressed=True),
        det("S2_Q", "rope", 3300, 150, conf=0.55, pct=55.0),       # quiet water column
    ]


def previous_detections() -> list[dict]:
    lat1, lon1 = pos(415, 130)
    p1 = offset_latlon(lat1, lon1, 5.0, 0.0)            # 5 m away -> persistent
    lat2, lon2 = pos(1510, 800)
    p2 = offset_latlon(lat2, lon2, 0.0, 45.0)           # 45 m away -> moved
    p3 = pos(2800, 300)                                 # inside current coverage, gone -> removed
    p4 = offset_latlon(LAT0, LON0, 3000.0, 500.0)       # outside current coverage -> not removed
    out = []
    for ident, (la, lo) in (("S1_P1", p1), ("S1_P2", p2), ("S1_P3", p3), ("S1_P4", p4)):
        d = det(ident, "net", 0, 0, conf=0.8)
        d["latitude"], d["longitude"] = round(la, 8), round(lo, 8)
        out.append(d)
    return out


# --- fakes for the geo agent's stages ---------------------------------------------------

def fake_habitat(lat, lon):
    return {"covered": True, "inside": [],
            "nearest": [{"layer": "reefs", "kind": "coral_reef", "name": "Synthetic Reef A",
                         "distance_m": 850.0, "bearing_deg": 95.0,
                         "source": {"name": "Synthetic test reef layer", "url": None, "licence": None,
                                    "snapshot": "test", "used_for": "habitat proximity"},
                         "geometry_quality": "synthetic"}],
            "score": 0.7, "terms": {"distance_decay": 0.7}}


def fake_drift(lat, lon, start_time, *, mode="seabed", horizon_hours=240, n_particles=500,
               dt_minutes=30, field=None, layers=None, seed=0):
    ring = [[lon, lat], [lon + 0.01, lat + 0.002], [lon + 0.012, lat - 0.002], [lon, lat]]
    return {"mode": mode, "horizon_hours": horizon_hours,
            "snapshots": [{"t_hours": 0, "points": [[lon, lat]], "cone50": None, "cone90": None},
                          {"t_hours": horizon_hours, "points": [], "cone50": ring, "cone90": ring}],
            "impacts": [{"kind": "coral_reef", "name": "Synthetic Reef A",
                         "probability": 0.35 if mode == "seabed" else 0.6,
                         "first_arrival_hours": 40.0, "source": "synthetic"}],
            "stranding_probability": 0.05, "assumptions": ["synthetic uniform current"],
            "limitations": ["test fake"], "mean_current_ms": 0.3,
            "current_source": {"name": "Synthetic uniform current (test)", "url": None,
                               "licence": None, "snapshot": "test", "used_for": "drift forecast"}}


def fake_people(target, habitat, drift, field, bathymetry):
    shallow = (target.get("seabed_depth_m") or 99) < 20
    return {"propeller_hazard": {"level": "high" if shallow and "net" in target["object_class"] else "low",
                                 "reasons": ["synthetic: net in shallow water"], "terms": {}},
            "diver_brief": {"summary": "synthetic brief - entanglement risk, carry cutting tools"}}


def fake_stages(**over):
    s = {"habitat": fake_habitat, "drift": fake_drift, "people": fake_people, "field": {"synthetic": True},
         "layers": None, "bathymetry": None,
         "harbours": [{"name": "Synthetic Harbour", "latitude": LAT0 - 0.02, "longitude": LON0 - 0.01,
                       "source": "synthetic test harbour"}],
         "data_sources": []}
    s.update(over)
    return s


# --- contract validator -------------------------------------------------------------------

def validate_contract(r: dict) -> None:
    for key in ("format", "survey_id", "generated_at", "demo", "synthetic_inputs", "data_sources",
                "caveats", "targets", "removed_since_previous", "change_summary", "recovery_plan", "summary"):
        assert key in r, f"missing top-level {key}"
    assert r["format"] == "deepecho-ghosttrace/1"
    assert isinstance(r["demo"], bool) and isinstance(r["synthetic_inputs"], bool)
    for ds in r["data_sources"]:
        assert set(("name", "url", "licence", "snapshot", "used_for")) <= set(ds), ds
    assert all(isinstance(c, str) for c in r["caveats"])
    ranks = []
    for t in r["targets"]:
        for key in ("detection_id", "object_class", "latitude", "longitude", "confidence_pct",
                    "dimensions", "suppressed", "activity", "habitat", "drift", "people", "change",
                    "priority", "alert"):
            assert key in t, f"target missing {key}"
        a = t["activity"]
        assert isinstance(a["available"], bool)
        assert a["level"] in ("high", "moderate", "low", "unknown")
        assert a["score"] is None or 0 <= a["score"] <= 1
        if a["available"]:
            for k in ("echo_clusters_near", "echo_area_near_m2", "background_clusters_per_window",
                      "enrichment_ratio", "window_m", "side"):
                assert k in a["evidence"], k
        assert "basis" in a and "limitations" in a
        for stage in ("habitat", "drift", "people"):
            if t[stage].get("available") is False:
                assert t[stage].get("reason"), f"{stage} unavailable without reason"
        c = t["change"]
        assert c["status"] in ("new", "moved", "persistent", "first_survey", "unmatched_no_prior")
        for k in ("previous_survey_id", "previous_detection_id", "previous_latitude",
                  "previous_longitude", "moved_m", "basis"):
            assert k in c
        if c["status"] in ("moved", "persistent"):
            assert isinstance(c["previous_latitude"], float) and isinstance(c["previous_longitude"], float), c
        else:
            assert c["previous_latitude"] is None and c["previous_longitude"] is None, c
        p = t["priority"]
        assert 0 <= p["score"] <= 1 and p["tier"] in ("urgent", "high", "routine")
        assert isinstance(p["rank"], int)
        ranks.append(p["rank"])
        assert "formula" in p
        for name, term in p["terms"].items():
            assert set(("value", "weight", "contribution")) <= set(term), name
        al = t["alert"]
        for k in ("authorities", "subject", "draft_text", "generated_by", "citations", "basis"):
            assert k in al
        assert al["generated_by"] in ("template", "assistant")
        for au in al["authorities"]:
            assert set(("name", "role", "contact_basis")) <= set(au)
    assert sorted(ranks) == list(range(1, len(ranks) + 1))
    for rem in r["removed_since_previous"]:
        for k in ("previous_survey_id", "previous_detection_id", "object_class", "latitude", "longitude",
                  "basis"):
            assert k in rem
    cs = r["change_summary"]
    for k in ("compared_with", "new", "moved", "persistent", "removed"):
        assert k in cs
    rp = r["recovery_plan"]
    for k in ("start", "order", "legs", "total_km", "method", "notes"):
        assert k in rp
    for leg in rp["legs"]:
        assert set(("from", "to", "distance_km")) <= set(leg)
    for k in ("targets", "urgent", "high", "actively_fishing", "near_sensitive_habitat", "propeller_hazards"):
        assert isinstance(r["summary"][k], int)
    json.dumps(r, allow_nan=False)


# --- tests ---------------------------------------------------------------------------------

class Ctx:
    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gt-core-"))

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


def test_class_matching():
    assert changes.is_target_class("net") and changes.is_target_class("Ghost_Net")
    assert changes.is_target_class("fishing nets") and changes.is_target_class("tyre")
    assert not changes.is_target_class("cabinet") and not changes.is_target_class("ship")
    assert not changes.is_target_class("unknown", include_unknown=False)
    assert changes.is_target_class("unknown", include_unknown=True)
    assert changes.class_family("ghost-gear") == "gear" and changes.class_family("tire") == "debris"


def test_watercolumn_aggregation_high_control_low(ctx):
    dets = current_detections()
    s = write_survey(ctx.tmp / "wc", "wc-survey", dets, processed_at="2026-09-02T00:00:00+00:00")
    cache = {}
    near = watercolumn.activity_evidence(dets[0], s, other_detections=dets, cache=cache)
    assert near["available"], near
    assert near["evidence"]["side"] == "port"
    assert near["evidence"]["echo_clusters_near"] >= 6, near["evidence"]
    assert near["level"] == "high", near
    # score recomputes from its own evidence
    e = near["evidence"]
    enr = e["echo_clusters_near"] / (e["background_clusters_per_window"] + cfg.WC_EPSILON)
    assert abs(enr - e["enrichment_ratio"]) < 1e-3
    assert abs(watercolumn.logistic_score(enr) - near["score"]) < 1e-3
    quiet = watercolumn.activity_evidence(dets[5], s, other_detections=dets, cache=cache)
    assert quiet["available"] and quiet["level"] == "low", quiet
    assert quiet["evidence"]["echo_clusters_near"] == 0
    # starboard side detection resolves to starboard
    star = watercolumn.activity_evidence(dets[1], s, other_detections=dets, cache=cache)
    assert star["evidence"]["side"] == "starboard"
    assert "not proof of entanglement" in near["limitations"]


def test_watercolumn_no_water_column(ctx):
    dets = current_detections()
    s = write_survey(ctx.tmp / "nowc", "plain", dets, processed_at="2026-09-02T00:00:00+00:00", nav=False)
    r = watercolumn.activity_evidence(dets[0], s)
    assert r["available"] is False and r["score"] is None and r["level"] == "unknown"
    assert "water-column" in r["reason"]
    s2 = write_survey(ctx.tmp / "navonly", "navonly", dets, processed_at="2026-09-02T00:00:00+00:00", wc=False)
    r2 = watercolumn.activity_evidence(dets[0], s2)
    assert r2["available"] is False and r2["reason"]


def _targets_from(dets):
    out = []
    for d in dets:
        if changes.is_target_class(d["object_class"]) and not d.get("suppressed"):
            out.append({"detection_id": d["id"], "object_class": d["object_class"],
                        "latitude": d["latitude"], "longitude": d["longitude"],
                        "dimensions": d.get("dimensions")})
    return out


def test_changes_statuses_and_coverage(ctx):
    root = ctx.tmp / "surveys"
    write_survey(root, "s1", previous_detections(), processed_at="2026-06-01T00:00:00+00:00",
                 nav=False)
    dets = current_detections()
    s2 = write_survey(root, "s2", dets, processed_at="2026-09-02T00:00:00+00:00")
    res = changes.compare_with_previous(s2, _targets_from(dets), surveys_root=root)
    ch = res["changes"]
    assert ch["S2_D1"]["status"] == "persistent" and ch["S2_D1"]["previous_detection_id"] == "S1_P1"
    assert ch["S2_D1"]["moved_m"] <= cfg.STATIONARY_M
    assert ch["S2_D2"]["status"] == "moved" and ch["S2_D2"]["previous_detection_id"] == "S1_P2"
    assert 30 < ch["S2_D2"]["moved_m"] < 60
    assert ch["S2_D3"]["status"] == "new", ch["S2_D3"]          # drum: no debris-family prior
    # previous position: the matched prior target's own position, for the map arrow
    prior = {d["id"]: d for d in previous_detections()}
    for cur_id, prev_id in (("S2_D1", "S1_P1"), ("S2_D2", "S1_P2")):
        assert ch[cur_id]["previous_latitude"] == prior[prev_id]["latitude"], ch[cur_id]
        assert ch[cur_id]["previous_longitude"] == prior[prev_id]["longitude"], ch[cur_id]
        cur = next(d for d in dets if d["id"] == cur_id)
        back = changes.haversine_m(cur["latitude"], cur["longitude"], ch[cur_id]["previous_latitude"],
                                   ch[cur_id]["previous_longitude"])
        assert abs(back - ch[cur_id]["moved_m"]) < 0.1, (back, ch[cur_id]["moved_m"])
    assert ch["S2_D3"]["previous_latitude"] is None and ch["S2_D3"]["previous_longitude"] is None
    removed = {r["previous_detection_id"] for r in res["removed_since_previous"]}
    assert removed == {"S1_P3"}, removed                          # P4 is outside coverage
    assert res["change_summary"]["compared_with"] == ["s1"]
    assert res["change_summary"]["removed"] == 1 and res["change_summary"]["moved"] == 1


def test_changes_later_survey_not_prior_and_first_survey(ctx):
    root = ctx.tmp / "later"
    write_survey(root, "future", previous_detections(), processed_at="2027-01-01T00:00:00+00:00", nav=False)
    dets = current_detections()
    s2 = write_survey(root, "s2", dets, processed_at="2026-09-02T00:00:00+00:00")
    res = changes.compare_with_previous(s2, _targets_from(dets), surveys_root=root)
    assert all(c["status"] == "first_survey" for c in res["changes"].values())
    assert res["removed_since_previous"] == []


def test_changes_class_family_gate(ctx):
    root = ctx.tmp / "family"
    prev = previous_detections()
    prev[0]["object_class"] = prev[0]["class_normalized"] = "tyre"   # debris beside a net
    write_survey(root, "s1", prev, processed_at="2026-06-01T00:00:00+00:00", nav=False)
    dets = current_detections()
    s2 = write_survey(root, "s2", dets, processed_at="2026-09-02T00:00:00+00:00")
    res = changes.compare_with_previous(s2, _targets_from(dets), surveys_root=root)
    assert res["changes"]["S2_D1"]["status"] == "new"


def test_changes_not_georeferenced(ctx):
    root = ctx.tmp / "nogeo"
    write_survey(root, "s1", previous_detections(), processed_at="2026-06-01T00:00:00+00:00", nav=False)
    dets = [det("X1", "net", 400, 100, geo=False), det("X2", "drum", 900, 700, geo=False)]
    s2 = write_survey(root, "s2", dets, processed_at="2026-09-02T00:00:00+00:00", nav=False, geo=False)
    res = changes.compare_with_previous(s2, _targets_from(dets), surveys_root=root)
    assert all(c["status"] == "unmatched_no_prior" and c["basis"] for c in res["changes"].values())
    assert res["removed_since_previous"] == []


def test_changes_order_by_ping_time_not_processing(ctx):
    """A survey recorded later but processed earlier is still the later one."""
    root = ctx.tmp / "order"
    recorded_first = write_survey(root, "recorded-first", previous_detections(),
                                  processed_at="2026-09-10T00:00:00+00:00", nav=False)
    nav_first = recorded_first / "nav"
    write_nav(nav_first, "line1", start="2026-09-01T04:30:00+00:00")
    dets = current_detections()
    s2 = write_survey(root, "recorded-later", dets, processed_at="2026-09-02T00:00:00+00:00")
    write_nav(s2 / "nav", "line1", start="2026-09-05T04:30:00+00:00")     # recorded 4 days later
    t_first, basis_first = changes.survey_time(recorded_first)
    t_later, basis_later = changes.survey_time(s2)
    assert "ping time" in basis_first and "ping time" in basis_later, (basis_first, basis_later)
    res = changes.compare_with_previous(s2, _targets_from(dets), surveys_root=root)
    assert res["change_summary"]["compared_with"] == ["recorded-first"], (t_first, t_later, res)
    # and the other way round: the earlier recording has no previous survey
    back = changes.compare_with_previous(recorded_first, _targets_from(previous_detections()),
                                         surveys_root=root)
    assert back["change_summary"]["compared_with"] == [], back["change_summary"]


def test_safety_thresholds_exposed():
    from ghosttrace import config_geo as geo_cfg
    from ghosttrace.safety import people_safety

    target = {"latitude": 9.12, "longitude": 79.05, "seabed_depth_m": 16.0,
              "dimensions": {"length_m": 9.0, "width_m": 6.0}}
    out = people_safety(target, {"available": False, "reason": "test"}, {"available": False, "reason": "test"},
                        None, None)
    th = out["diver_brief"]["thresholds"]
    for k in ("recreational_depth_m", "advanced_depth_m", "max_current_mps", "basis",
              "depth_basis", "current_basis", "depth_label", "current_label"):
        assert k in th, k
    assert th["recreational_depth_m"] == geo_cfg.DIVE_LIMIT_OPEN_WATER_M
    assert th["advanced_depth_m"] == geo_cfg.DIVE_LIMIT_ADVANCED_M
    assert th["max_current_mps"] == geo_cfg.DIVER_CURRENT_LIMIT_MPS
    assert th["depth_check_uses"] == "advanced_depth_m"
    assert th["current_label"] in ("cited", "heuristic") and th["depth_label"] in ("cited", "heuristic")
    if th["current_label"] == "cited":
        assert "1910.424" in th["current_basis"]
    else:
        assert "heuristic" in th["current_basis"]
    brief = out["diver_brief"]
    assert brief["within_recreational_limit"] is (16.0 <= th["advanced_depth_m"])
    assert brief["within_open_water_limit"] is (16.0 <= th["recreational_depth_m"])
    assert brief["current_mps_at_depth"] is None and brief["current_ok_for_divers"] is None


def _target_for_priority(**kw):
    t = {"detection_id": "T", "object_class": "net", "confidence_pct": 90.0,
         "activity": {"available": True, "score": 0.5},
         "habitat": {"covered": True, "score": 0.5, "nearest": [], "inside": []},
         "drift": {"impacts": [{"kind": "coral_reef", "name": "R", "probability": 0.5}]},
         "people": {"propeller_hazard": {"level": "moderate"}},
         "dimensions": {"length_m": 5, "width_m": 2},
         "change": {"status": "persistent"}, "seabed_depth_m": 20.0}
    t.update(kw)
    return t


def test_priority_recompute_and_monotonic():
    base = priority.score_target(_target_for_priority())
    assert abs(priority.recompute(base) - base["score"]) < 1e-3
    assert abs(sum(cfg.PRIORITY_WEIGHTS.values()) - 1.0) < 1e-9
    variants = {
        "activity": [_target_for_priority(activity={"available": True, "score": v}) for v in (0.1, 0.5, 0.9)],
        "habitat": [_target_for_priority(habitat={"score": v}) for v in (0.1, 0.5, 0.9)],
        "drift": [_target_for_priority(drift={"impacts": [{"kind": "reef", "probability": v}]})
                  for v in (0.0, 0.4, 0.9)],
        "people": [_target_for_priority(people={"propeller_hazard": {"level": v}})
                   for v in ("none", "low", "moderate", "high")],
        "size": [_target_for_priority(dimensions={"length_m": v, "width_m": 2}) for v in (1, 10, 100)],
        "change": [_target_for_priority(change={"status": v}) for v in ("persistent", "new", "moved")],
        "depth": [_target_for_priority(seabed_depth_m=v) for v in (60, 25, 5)],
        "confidence": [_target_for_priority(confidence_pct=v) for v in (20, 50, 95)],
    }
    for name, series in variants.items():
        scores = [priority.score_target(t)["score"] for t in series]
        assert scores == sorted(scores), (name, scores)
        for t in series:
            p = priority.score_target(t)
            assert abs(priority.recompute(p) - p["score"]) < 1e-3, name


def test_priority_weak_detection_never_urgent_and_neutral():
    maxed = _target_for_priority(confidence_pct=40.0, activity={"available": True, "score": 1.0},
                                 habitat={"score": 1.0}, drift={"impacts": [{"kind": "reef", "probability": 1}]},
                                 people={"propeller_hazard": {"level": "high"}},
                                 dimensions={"length_m": 100, "width_m": 100},
                                 change={"status": "moved"}, seabed_depth_m=1.0)
    p = priority.score_target(maxed)
    assert p["tier"] != "urgent" and p["score"] <= 0.4 + 1e-9
    unknown = priority.score_target({"detection_id": "U", "object_class": "net", "confidence_pct": 80,
                                     "activity": {"available": False}, "habitat": {"available": False},
                                     "drift": {"available": False}, "people": {"available": False},
                                     "change": {"status": "unmatched_no_prior"}})
    for name, term in unknown["terms"].items():
        if name != "confidence":
            assert term["value"] == cfg.PRIORITY_NEUTRAL[name] and "neutral" in term["basis"], name


def test_priority_confidence_missing_is_flagged_not_zero():
    """A missing confidence is a data fault: neutral factor, loud flag, still in the queue."""
    for absent in ({}, {"confidence_pct": None}, {"confidence_pct": float("nan")}):
        t = _target_for_priority(**absent)
        if not absent:
            del t["confidence_pct"]
        p = priority.score_target(t)
        assert p["confidence_missing"] is True
        assert p["terms"]["confidence"]["value"] == cfg.PRIORITY_NEUTRAL_CONFIDENCE
        assert p["terms"]["confidence"]["measured"] is False
        assert p["terms"]["confidence"]["basis"].startswith("CONFIDENCE MISSING")
        assert p["score"] > 0 and abs(priority.recompute(p) - p["score"]) < 1e-3
    ok = priority.score_target(_target_for_priority())
    assert ok["confidence_missing"] is False and ok["terms"]["confidence"]["measured"] is True
    # neutral confidence alone can never make a target urgent
    maxed = _target_for_priority(activity={"available": True, "score": 1.0}, habitat={"score": 1.0},
                                 drift={"impacts": [{"kind": "reef", "probability": 1}]},
                                 people={"propeller_hazard": {"level": "high"}},
                                 dimensions={"length_m": 100, "width_m": 100},
                                 change={"status": "moved"}, seabed_depth_m=1.0)
    del maxed["confidence_pct"]
    assert priority.score_target(maxed)["tier"] != "urgent"


def test_priority_obvious_targets_land_in_expected_tiers():
    """Known-answer targets: the tier cutoffs must put an obviously urgent net in 'urgent'."""
    urgent = _target_for_priority(
        confidence_pct=85.0, activity={"available": True, "score": 0.9},
        habitat={"covered": True, "score": 0.9, "inside": [{"kind": "reef", "name": "R"}], "nearest": []},
        drift={"impacts": [{"kind": "reef", "name": "R", "probability": 0.6}]},
        people={"propeller_hazard": {"level": "high"}},
        dimensions={"length_m": 20, "width_m": 5}, change={"status": "moved"}, seabed_depth_m=8.0)
    p = priority.score_target(urgent)
    assert p["tier"] == "urgent", p["score"]
    # the same net on a first survey (no change history) at a typical detector confidence
    first = dict(urgent, confidence_pct=77.0, change={"status": "unmatched_no_prior"})
    p1 = priority.score_target(first)
    assert p1["tier"] == "urgent", p1["score"]
    # the demo survey's actual best net: moderate activity, a reef 4-5 km off, no drift impact,
    # low propeller risk, no history -> high, not urgent, and that is the intended reading
    demo = _target_for_priority(
        confidence_pct=76.5, activity={"available": True, "score": 0.68},
        habitat={"covered": True, "score": 0.41, "nearest": [], "inside": []},
        drift={"impacts": []}, people={"propeller_hazard": {"level": "low"}},
        dimensions={"length_m": 12, "width_m": 8}, change={"status": "unmatched_no_prior"},
        seabed_depth_m=14.0)
    assert priority.score_target(demo)["tier"] == "high"
    # an inert, deep, small piece of debris far from anything stays routine
    routine = _target_for_priority(
        confidence_pct=70.0, activity={"available": True, "score": 0.05},
        habitat={"covered": True, "score": 0.05, "nearest": [], "inside": []},
        drift={"impacts": []}, people={"propeller_hazard": {"level": "low"}},
        dimensions={"length_m": 1, "width_m": 1}, change={"status": "persistent"}, seabed_depth_m=45.0)
    assert priority.score_target(routine)["tier"] == "routine"


def test_watercolumn_high_needs_several_clusters():
    """One or two blobs on a quiet line reach a high ratio; they are not a school."""
    one = watercolumn.logistic_score(1 / (0.0 + cfg.WC_EPSILON))
    two = watercolumn.logistic_score(2 / (0.0 + cfg.WC_EPSILON))
    assert watercolumn.level_for(two) == "high", two          # the ratio alone would say high
    level, why = watercolumn.gated_level(two, 2)
    assert level == "moderate" and "capped" in why
    level, why = watercolumn.gated_level(one, 1)
    assert level != "high" and (why is None or "capped" in why)
    many = watercolumn.logistic_score(cfg.WC_MIN_CLUSTERS_HIGH / (0.0 + cfg.WC_EPSILON))
    assert watercolumn.gated_level(many, cfg.WC_MIN_CLUSTERS_HIGH) == ("high", None)
    assert watercolumn.gated_level(0.2, 0) == ("low", None)
    assert watercolumn.gated_level(None, 0) == ("unknown", None)


def test_safety_low_supported_without_depth():
    """Unknown depth alone must not turn a fully assessed, all-negative net into 'unknown'."""
    from ghosttrace import safety
    no_bathy = type("NoBathy", (), {"depth_at": lambda self, a, b: {"depth_m": None, "note": "none"}})()
    habitat = {"available": True, "covered": True, "nearest": [
        {"kind": "harbour", "name": "Far Jetty", "distance_m": 30_000.0}], "inside": []}
    drift = {"available": True, "mode": "seabed", "impacts": [], "time_mapping": {}}
    small = {"latitude": 9.12, "longitude": 79.05, "dimensions": {"length_m": 4.0, "width_m": 2.0}}
    p = safety.people_safety(small, habitat, drift, None, no_bathy)["propeller_hazard"]
    assert p["level"] == "low", p
    assert p["terms"]["shallow"]["assessed"] is False
    assert all(p["terms"][k]["assessed"] and not p["terms"][k]["applied"]
               for k in ("floating", "harbour_near", "harbour_drift", "large_net"))
    assert any("depth unknown" in r for r in p["reasons"])
    # the same net with no size recorded: one check short of support -> unknown, and it says which
    nosize = {"latitude": 9.12, "longitude": 79.05}
    q = safety.people_safety(nosize, habitat, drift, None, no_bathy)["propeller_hazard"]
    assert q["level"] == "unknown" and any("large_net" in r for r in q["reasons"]), q
    # a harbour drift check that could not run: also unknown
    r = safety.people_safety(small, habitat, {"available": False}, None, no_bathy)["propeller_hazard"]
    assert r["level"] == "unknown", r
    # measured depth keeps low supported on its own
    with_depth = dict(small, seabed_depth_m=25.0)
    s = safety.people_safety(with_depth, {"available": False}, {"available": False}, None, no_bathy)
    assert s["propeller_hazard"]["level"] == "low", s["propeller_hazard"]


def test_recovery_route():
    tiers = ["routine", "urgent", "high", "urgent", "routine", "high", "urgent"]
    rng = np.random.default_rng(3)
    ts = []
    for i, tier in enumerate(tiers):
        la, lo = offset_latlon(LAT0, LON0, float(rng.uniform(-3000, 3000)), float(rng.uniform(-3000, 3000)))
        ts.append({"detection_id": f"T{i}", "latitude": la, "longitude": lo, "priority": {"tier": tier}})
    ts.append({"detection_id": "NOPOS", "latitude": None, "longitude": None, "priority": {"tier": "urgent"}})
    harb = [{"name": "Far", "latitude": LAT0 + 1, "longitude": LON0, "source": "x"},
            {"name": "Near", "latitude": LAT0 - 0.01, "longitude": LON0, "source": "y"}]
    plan = recovery.plan_route(ts, harb)
    assert plan["start"]["name"] == "Near"
    assert sorted(plan["order"]) == sorted(t["detection_id"] for t in ts[:-1])
    order_tiers = [next(t for t in ts if t["detection_id"] == i)["priority"]["tier"] for i in plan["order"]]
    rank = {n: k for k, n in enumerate(recovery.TIER_ORDER)}
    assert [rank[x] for x in order_tiers] == sorted(rank[x] for x in order_tiers)
    assert len(plan["legs"]) == len(plan["order"]) and plan["legs"][0]["from"] == "Near"
    assert abs(sum(l["distance_km"] for l in plan["legs"]) - plan["total_km"]) < 1e-6
    assert any("NOPOS" in n for n in plan["notes"]) and any("straight" in n.lower() for n in plan["notes"])
    no_harbour = recovery.plan_route(ts, None)
    assert no_harbour["start"]["name"] == f"target {no_harbour['order'][0]}"
    assert len(no_harbour["legs"]) == len(no_harbour["order"]) - 1
    # 2-opt never lengthens a nearest-neighbour tour
    items = [(t["detection_id"], (t["latitude"], t["longitude"])) for t in ts[:-1]]
    nn = recovery._nearest_neighbour(None, items)
    opt = recovery._two_opt(None, nn)
    assert recovery._path_len(None, [p for _, p in opt]) <= recovery._path_len(None, [p for _, p in nn]) + 1e-9
    assert opt[0] == nn[0]


PHONE = re.compile(r"\+?\d[\d\s()-]{7,}\d")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def test_alerts_never_invent_contacts():
    kb_text = " ".join(re.sub(r"\s+", " ", p.read_text(encoding="utf-8")).lower()
                       for p in (REPO / "rag_assistant" / "kb").glob("*.md"))
    cases = [
        _target_for_priority(object_class="net", latitude=LAT0, longitude=LON0,
                             people={"propeller_hazard": {"level": "high", "reasons": ["shallow"]}},
                             habitat={"inside": [], "nearest": [{"name": "Reef", "kind": "reef",
                                                                 "distance_m": 300}], "score": 0.8}),
        _target_for_priority(object_class="drum", latitude=None, longitude=None),
        _target_for_priority(object_class="tyre", latitude=LAT0, longitude=LON0,
                             habitat={"available": False, "reason": "x"}, drift={"available": False, "reason": "x"},
                             people={"available": False, "reason": "x"}),
    ]
    for t in cases:
        t["priority"] = priority.score_target(t)
        t["priority"]["rank"] = 1
        a = alerts.draft_alert(t, survey_id="S", use_assistant=False)
        assert a["generated_by"] == "template"
        for au in a["authorities"]:
            if au["name"] != cfg.NOT_IN_CORPUS:
                assert re.sub(r"\s+", " ", au["name"]).lower() in kb_text, au["name"]
        assert not PHONE.search(a["draft_text"]), PHONE.search(a["draft_text"])
        assert not EMAIL.search(a["draft_text"])
        assert "verify before action" in a["draft_text"].lower()
        for au in a["authorities"]:
            assert not PHONE.search(au["contact_basis"]) and not EMAIL.search(au["contact_basis"])
    net_alert = alerts.draft_alert(cases[0], survey_id="S", use_assistant=False)
    names: dict = {}
    for au in net_alert["authorities"]:
        names.setdefault(au["situation"], au["name"])
    assert names["fisheries"] == "State Fisheries Departments"
    assert "Fishery Survey of India" in {au["name"] for au in net_alert["authorities"]}
    assert names["navigation_hazard"] == "National Hydrographic Office"
    # "Reef" names no place any registered office covers, so no park office is
    # guessed for it.
    assert names["protected_habitat"] == cfg.NOT_IN_CORPUS
    mannar = dict(cases[0], habitat={"inside": [{"name": "Gulf of Mannar Marine National Park"}],
                                     "nearest": [], "score": 0.9})
    mannar_names = [au["name"] for au in alerts.draft_alert(mannar, survey_id="S",
                                                            use_assistant=False)["authorities"]
                    if au["situation"] == "protected_habitat"]
    assert mannar_names == ["Wildlife Warden, Ramanathapuram"], mannar_names
    odisha = dict(cases[0], habitat={"inside": [], "score": 0.9, "nearest": [
        {"name": "Gahirmatha Marine Wildlife Sanctuary", "kind": "protected_area", "distance_m": 400}]})
    odisha_names = {au["name"] for au in alerts.draft_alert(odisha, survey_id="S",
                                                           use_assistant=False)["authorities"]
                    if au["situation"] == "protected_habitat"}
    assert "Wildlife Warden, Ramanathapuram" not in odisha_names
    assert "Mangrove Forest Division (Wildlife), Rajnagar" in odisha_names
    assert "position unavailable" in alerts.draft_alert(cases[1], survey_id="S", use_assistant=False)["draft_text"]
    drum = alerts.draft_alert(cases[1], survey_id="S", use_assistant=False)
    assert any("Coast Guard" in au["name"] for au in drum["authorities"])
    assert all(c["doc"].startswith("kb/") for c in drum["citations"])
    # A registry name missing from the corpus degrades to "not in corpus".
    saved = alerts.AUTHORITY_REGISTRY["possible_pollution"][0]["name"]
    try:
        alerts.AUTHORITY_REGISTRY["possible_pollution"][0]["name"] = "Invented Marine Office"
        a = alerts.draft_alert(cases[1], survey_id="S", use_assistant=False)
        assert all(au["name"] != "Invented Marine Office" for au in a["authorities"])
    finally:
        alerts.AUTHORITY_REGISTRY["possible_pollution"][0]["name"] = saved


def test_alert_assistant_fallbacks():
    t = _target_for_priority(latitude=LAT0, longitude=LON0)
    t["priority"] = priority.score_target(t)

    def boom(message, detection):
        raise RuntimeError("no provider")
    a = alerts.draft_alert(t, survey_id="S", use_assistant=True, assistant_fn=boom)
    assert a["generated_by"] == "template" and "assistant failed" in a["basis"]

    def refuses(message, detection):
        return {"answer": "The sources do not cover this.", "refusal": True, "grounded": False, "sources": []}
    a = alerts.draft_alert(t, survey_id="S", use_assistant=True, assistant_fn=refuses)
    assert a["generated_by"] == "template" and "refusal" in a["basis"]

    saved = cfg.ASSISTANT_TIMEOUT_S
    cfg.ASSISTANT_TIMEOUT_S = 0.1
    try:
        a = alerts.draft_alert(t, survey_id="S", use_assistant=True,
                               assistant_fn=lambda m, d: (time.sleep(0.5), {})[1])
        assert a["generated_by"] == "template" and "timed out" in a["basis"]
    finally:
        cfg.ASSISTANT_TIMEOUT_S = saved

    def ok(message, detection):
        return {"answer": "Report the derelict net [S1].", "refusal": False, "grounded": True,
                "sources": [{"n": 1, "title": "Reporting", "doc_id": "reporting-authorities-india"}]}
    a = alerts.draft_alert(t, survey_id="S", use_assistant=True, assistant_fn=ok)
    assert a["generated_by"] == "assistant" and a["authorities"]


def build_example_root(ctx) -> tuple[Path, Path]:
    root = ctx.tmp / "e2e"
    write_survey(root, "gt-synthetic-s1", previous_detections(), processed_at="2026-06-01T00:00:00+00:00",
                 nav=False)
    s2 = write_survey(root, "gt-synthetic-s2", current_detections(), processed_at="2026-09-02T00:00:00+00:00")
    return root, s2


def test_engine_end_to_end(ctx):
    root, s2 = build_example_root(ctx)
    events = []
    r = run_ghosttrace(s2, surveys_root=root, on_event=events.append, stages=fake_stages(),
                       n_particles=50, horizon_hours=72)
    validate_contract(r)
    disk = json.loads((s2 / "ghosttrace.json").read_text(encoding="utf-8"))
    validate_contract(disk)
    ids = {t["detection_id"] for t in r["targets"]}
    assert ids == {"S2_D1", "S2_D2", "S2_D3", "S2_Q"}, ids          # ship excluded, suppressed excluded
    assert r["summary"]["suppressed_excluded"] == 1
    by = {t["detection_id"]: t for t in r["targets"]}
    assert by["S2_D1"]["activity"]["level"] == "high"
    assert by["S2_D1"]["change"]["status"] == "persistent" and by["S2_D2"]["change"]["status"] == "moved"
    assert by["S2_D1"]["priority"]["rank"] == 1, [(t["detection_id"], t["priority"]["score"]) for t in r["targets"]]
    assert by["S2_D1"]["drift"]["start_time_basis"].startswith("ping time")
    assert by["S2_D1"]["seabed_depth_m"] == 15.0
    assert by["S2_D2"]["confidence_pct"] == 70.0 and "detector confidence" in by["S2_D2"]["confidence_basis"]
    assert r["demo"] is True and r["synthetic_inputs"] is True
    assert [x["previous_detection_id"] for x in r["removed_since_previous"]] == ["S1_P3"]
    assert r["recovery_plan"]["start"]["name"] == "Synthetic Harbour"
    assert r["summary"]["actively_fishing"] >= 1 and r["summary"]["propeller_hazards"] >= 1
    assert any(ds["name"] == "Synthetic uniform current (test)" for ds in r["data_sources"])
    stages_seen = {e["stage"] for e in events}
    assert {"load", "select", "target", "change", "priority", "alerts", "recovery", "write"} <= stages_seen
    assert all(e["type"] == "ghosttrace" for e in events)
    gj = json.loads((s2 / "ghosttrace.geojson").read_text(encoding="utf-8"))
    kinds = [f["properties"]["kind"] for f in gj["features"]]
    assert kinds.count("target") == 4 and "drift_cone90" in kinds and "recovery_route" in kinds
    cone = next(f for f in gj["features"] if f["properties"]["kind"] == "drift_cone90")
    lon, lat = cone["geometry"]["coordinates"][0][0]
    assert abs(lat - LAT0) < 0.1 and abs(lon - LON0) < 0.1
    assert not list(s2.glob(".ghosttrace*.tmp"))
    return r


def test_engine_failure_isolation(ctx):
    root, s2 = build_example_root(ctx)

    def flaky_drift(lat, lon, start_time, **kw):
        if abs(lat - pos(1510, 800)[0]) < 1e-6:
            raise ValueError("synthetic drift failure")
        return fake_drift(lat, lon, start_time, **kw)

    def broken_people(*a, **k):
        raise RuntimeError("synthetic safety failure")

    r = run_ghosttrace(s2, surveys_root=root, stages=fake_stages(drift=flaky_drift, people=broken_people),
                       n_particles=20, horizon_hours=24, on_event=lambda e: 1 / 0)   # listener also broken
    validate_contract(r)
    by = {t["detection_id"]: t for t in r["targets"]}
    assert by["S2_D2"]["drift"]["available"] is False and "synthetic drift failure" in by["S2_D2"]["drift"]["reason"]
    assert by["S2_D1"]["drift"].get("impacts"), "other targets keep their forecast"
    assert all(t["people"]["available"] is False for t in r["targets"])
    stages = [(f["detection_id"], f["stage"]) for f in r["run"]["failures"]]
    assert ("S2_D2", "drift") in stages and len([s for s in stages if s[1] == "people"]) == 4
    assert r["summary"]["stage_failures"] == len(r["run"]["failures"])
    # And with no geo modules at all (the real state before the geo agent lands).
    r2 = run_ghosttrace(s2, surveys_root=root, stages={"habitat": None, "drift": None, "people": None,
                                                        "field": None, "layers": None, "bathymetry": None,
                                                        "harbours": None}, write=False)
    validate_contract(r2)
    assert all(t["drift"]["available"] is False for t in r2["targets"])


def test_engine_plain_survey(ctx):
    root = ctx.tmp / "plain"
    dets = [det("P1", "net", 400, 100, geo=False), det("P2", "mine", 900, 700, geo=False)]
    s = write_survey(root, "plain", dets, processed_at="2026-09-02T00:00:00+00:00", nav=False, geo=False)
    r = run_ghosttrace(s, stages=fake_stages())
    validate_contract(r)
    assert len(r["targets"]) == 1 and r["targets"][0]["activity"]["available"] is False
    assert r["targets"][0]["change"]["status"] == "unmatched_no_prior"
    assert r["recovery_plan"]["order"] == []


def write_fixture(ctx) -> Path:
    root, s2 = build_example_root(ctx)
    r = run_ghosttrace(s2, surveys_root=root, stages=fake_stages(), n_particles=50, horizon_hours=72)
    validate_contract(r)
    example = {"synthetic_example": True,
               "synthetic_example_note": ("Generated by tests_ghosttrace_core.py --write-fixture from a "
                                          "synthetic survey pair and FAKE habitat/drift/safety stages. "
                                          "Not a real survey, not real habitat, not a real forecast."),
               **r}
    out = REPO / "tests" / "fixtures" / "ghosttrace_example.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(example, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


TESTS = [
    ("class matching", test_class_matching, False),
    ("watercolumn: aggregation high, control low", test_watercolumn_aggregation_high_control_low, True),
    ("watercolumn: no water column -> unavailable", test_watercolumn_no_water_column, True),
    ("changes: persistent/moved/new/removed + coverage", test_changes_statuses_and_coverage, True),
    ("changes: later survey is not prior", test_changes_later_survey_not_prior_and_first_survey, True),
    ("changes: class family gate", test_changes_class_family_gate, True),
    ("changes: not georeferenced", test_changes_not_georeferenced, True),
    ("changes: order by ping time, not processing time", test_changes_order_by_ping_time_not_processing, True),
    ("safety: diver brief thresholds exposed", test_safety_thresholds_exposed, False),
    ("priority: recompute + monotonic", test_priority_recompute_and_monotonic, False),
    ("priority: weak never urgent, neutral values", test_priority_weak_detection_never_urgent_and_neutral, False),
    ("priority: missing confidence flagged, not zero", test_priority_confidence_missing_is_flagged_not_zero, False),
    ("priority: obvious targets land in expected tiers", test_priority_obvious_targets_land_in_expected_tiers, False),
    ("watercolumn: high needs several clusters", test_watercolumn_high_needs_several_clusters, False),
    ("safety: low supported without depth", test_safety_low_supported_without_depth, False),
    ("recovery: route validity", test_recovery_route, False),
    ("alerts: never invent contacts", test_alerts_never_invent_contacts, False),
    ("alerts: assistant fallbacks", test_alert_assistant_fallbacks, False),
    ("engine: end to end contract", test_engine_end_to_end, True),
    ("engine: per-target failure isolation", test_engine_failure_isolation, True),
    ("engine: plain image survey", test_engine_plain_survey, True),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--write-fixture", action="store_true")
    args = ap.parse_args()
    failed = 0
    for name, fn, needs_ctx in TESTS:
        ctx = Ctx() if needs_ctx else None
        try:
            fn(ctx) if needs_ctx else fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            print(traceback.format_exc() if args.verbose else "      " + traceback.format_exc().strip().splitlines()[-1])
        finally:
            if ctx:
                ctx.close()
    if args.write_fixture:
        ctx = Ctx()
        try:
            print(f"wrote {write_fixture(ctx)}")
        finally:
            ctx.close()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
