"""Tests for the GhostTrace geographic stages: layers, habitat, currents, drift, safety.

    .venv/bin/python tests_ghosttrace_geo.py

Plain asserts, no test framework. Exit status 1 when any test fails. Needs the
bundled data in data/ghosttrace (run tools/fetch_ghosttrace_data.py first).

Synthetic current fields appear ONLY in tests that check mechanics (stranding,
the seabed threshold, determinism, cones). Each of those builds an
AnalyticField, which labels itself synthetic. The last section runs a real
forecast from the demo position on the bundled HYCOM currents and prints a
summary.
"""

from __future__ import annotations

import json
import math
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import shapely
from shapely.geometry import Point, shape

from ghosttrace import config_geo as cfg
from ghosttrace.currents import AnalyticField, FieldCollection, load_default_field
from ghosttrace.drift import drift_forecast
from ghosttrace.habitat import habitat_context
from ghosttrace.layers import GEOD, load_bathymetry, load_default_layers
from ghosttrace.safety import people_safety

DEMO_LAT, DEMO_LON = 9.12, 79.05
RESULTS: list[tuple[str, bool, str]] = []


def test(fn):
    name = fn.__name__
    t0 = time.time()
    try:
        fn()
        RESULTS.append((name, True, f"{time.time() - t0:.1f}s"))
        print(f"PASS {name} ({time.time() - t0:.1f}s)")
    except Exception as exc:
        RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
    return fn


LAYERS = load_default_layers()
FIELD = load_default_field()


def _gom_field_window():
    f = FIELD.field_for(DEMO_LAT, DEMO_LON)
    return f, f.time_range


# --- layers ----------------------------------------------------------------------------------

@test
def layers_load():
    assert LAYERS.layers, "no layers loaded"
    kinds = set(LAYERS.kinds_present("gulf_of_mannar_palk_bay"))
    for k in ("land", "protected_area", "reef", "harbour", "dugong"):
        assert k in kinds, f"{k} missing from gulf_of_mannar_palk_bay: {kinds}"
    kinds_o = set(LAYERS.kinds_present("odisha_coast"))
    for k in ("land", "protected_area", "turtle_nesting", "harbour"):
        assert k in kinds_o, f"{k} missing from odisha_coast: {kinds_o}"
    for layer in LAYERS.layers.values():
        for p in layer.properties:
            assert p.get("source"), f"{layer.name}: feature without source"
            assert p.get("geometry_quality"), f"{layer.name}: feature without geometry_quality"
    assert LAYERS.harbours and all(-90 <= h["latitude"] <= 90 for h in LAYERS.harbours)
    assert LAYERS.data_sources, "manifest lists no sources"


@test
def coverage_flag():
    c = LAYERS.region_for(DEMO_LAT, DEMO_LON)
    assert c["covered"] and c["region"] == "gulf_of_mannar_palk_bay", c
    c = LAYERS.region_for(19.4, 85.1)
    assert c["covered"] and c["region"] == "odisha_coast", c
    c = LAYERS.region_for(15.0, 73.0)  # Goa: inside the EEZ, outside the bundled boxes
    assert not c["covered"] and "no bundled data" in c["reason"], c


# --- habitat ---------------------------------------------------------------------------------

@test
def habitat_inside_marine_national_park():
    park = next(g for l in LAYERS.by_kind("protected_area") for g, p in zip(l.geometries, l.properties)
                if p.get("name") == "Gulf of Mannar Marine National Park")
    pt = park.representative_point()
    h = habitat_context(pt.y, pt.x)
    assert h["covered"]
    assert any(i["name"] == "Gulf of Mannar Marine National Park" for i in h["inside"]), h["inside"]
    pa = next(n for n in h["nearest"] if n["kind"] == "protected_area")
    assert pa["distance_m"] == 0.0 and pa["bearing_deg"] is None
    assert h["score"] >= cfg.KIND_SENSITIVITY["protected_area"] - 1e-9


@test
def habitat_known_distance_sanity():
    h = habitat_context(DEMO_LAT, DEMO_LON)
    assert h["covered"] and 0.0 <= h["score"] <= 1.0
    pa = next(n for n in h["nearest"] if n["name"] == "Gulf of Mannar Marine National Park")
    park = next(g for l in LAYERS.by_kind("protected_area") for g, p in zip(l.geometries, l.properties)
                if p.get("name") == "Gulf of Mannar Marine National Park")
    # Independent check: geodesic distance to every vertex. The true nearest distance can only be
    # smaller than the nearest vertex, and on a densely digitised boundary not by much.
    coords = np.concatenate([np.asarray(r.coords) for poly in getattr(park, "geoms", [park])
                             for r in [poly.exterior, *poly.interiors]])
    _, _, d = GEOD.inv(np.full(len(coords), DEMO_LON), np.full(len(coords), DEMO_LAT), coords[:, 0], coords[:, 1])
    vmin = float(np.min(d))
    assert pa["distance_m"] <= vmin + 1.0, (pa["distance_m"], vmin)
    assert pa["distance_m"] >= 0.8 * vmin, (pa["distance_m"], vmin)
    assert 1000 < pa["distance_m"] < 20000, pa
    # the bearing points at the park: moving 1.1x the distance along it lands inside or next to it
    lon2, lat2, _ = GEOD.fwd(DEMO_LON, DEMO_LAT, pa["bearing_deg"], pa["distance_m"] + 50)
    assert park.distance(Point(lon2, lat2)) < 0.002, "bearing does not point at the park"


@test
def habitat_outside_coverage():
    h = habitat_context(15.0, 73.0)
    assert h["covered"] is False and h["score"] is None and h["nearest"] == [] and h["inside"] == []
    assert "no bundled data" in h["reason"]


# --- currents ---------------------------------------------------------------------------------

@test
def currents_finite_over_window():
    assert FIELD is not None, "no bundled currents"
    f, (t0, t1) = _gom_field_window()
    assert not f.synthetic
    assert (t1 - t0) >= timedelta(days=7), (t0, t1)
    for level in ("surface", "bottom"):
        for ts in f.times_s:
            u, v = f.sample(DEMO_LAT, DEMO_LON, float(ts), level)
            assert math.isfinite(u) and math.isfinite(v), (level, ts)
        mid = float(f.times_s[3]) + 5400.0  # between steps: linear time interpolation
        u, v = f.sample(DEMO_LAT, DEMO_LON, mid, level)
        ua, va = f.sample(DEMO_LAT, DEMO_LON, float(f.times_s[3]), level)
        ub, vb = f.sample(DEMO_LAT, DEMO_LON, float(f.times_s[4]), level)
        assert abs(u - 0.5 * (ua + ub)) < 1e-6 and abs(v - 0.5 * (va + vb)) < 1e-6
    assert all(abs(x) < 3.0 for x in f.sample(DEMO_LAT, DEMO_LON, float(f.times_s[0])))
    u, _ = f.sample(DEMO_LAT, DEMO_LON, float(f.times_s[0]) - 86400.0)
    assert math.isnan(u), "a time outside the window must be NaN, not clamped"
    u, _ = f.sample(9.92, 78.12, float(f.times_s[0]))  # Madurai, far inland
    assert math.isnan(u), "an inland point must be NaN"
    d = f.describe()
    assert d["latest_model_run"] and d["url"] and d["synthetic"] is False
    of = FIELD.field_for(19.5, 85.5)
    assert of is not None and math.isfinite(of.speed_at(19.3, 85.3, float(of.times_s[0])))


# --- drift -------------------------------------------------------------------------------------

T0 = datetime(2030, 1, 1, tzinfo=timezone.utc)


def _synthetic(bounds, **kw):
    return AnalyticField(bounds=bounds, t_start=T0, hours=500, **kw)


@test
def drift_deterministic_with_seed():
    f = _synthetic(cfg.REGIONS["gulf_of_mannar_palk_bay"],
                   velocity=lambda la, lo, th: (0.1 * np.sin(la * 40), 0.1 * np.cos(lo * 40)))
    a = drift_forecast(DEMO_LAT, DEMO_LON, T0, mode="floating", horizon_hours=48, n_particles=200, field=f, seed=7)
    b = drift_forecast(DEMO_LAT, DEMO_LON, T0, mode="floating", horizon_hours=48, n_particles=200, field=f, seed=7)
    c = drift_forecast(DEMO_LAT, DEMO_LON, T0, mode="floating", horizon_hours=48, n_particles=200, field=f, seed=8)
    strip = lambda d: json.dumps({k: v for k, v in d.items() if k != "generated_at"}, sort_keys=True, default=str)
    assert strip(a) == strip(b), "same seed must give identical output"
    assert strip(a) != strip(c), "a different seed should change the random walk"
    assert a["synthetic_currents"] is True and "SYNTHETIC" in json.dumps(a["assumptions"])
    assert [s["t_hours"] for s in a["snapshots"]] == [0, 12, 24, 36, 48]
    assert all(len(s["points"]) <= cfg.DRIFT_MAX_SNAPSHOT_POINTS for s in a["snapshots"])


@test
def drift_strands_on_land_with_onshore_current():
    # Odisha coast near Rushikulya runs north-east; onshore is north-west.
    lat, lon = 19.22, 85.16
    land = LAYERS.land("odisha_coast")
    assert not land.contains(Point(lon, lat)), "test start point must be at sea"
    f = _synthetic(cfg.REGIONS["odisha_coast"], velocity=(-0.35, 0.35))
    d = drift_forecast(lat, lon, T0, mode="floating", horizon_hours=72, n_particles=200, field=f, seed=1)
    assert d["available"], d
    assert d["stranding_probability"] > 0.95, d["stranding_probability"]
    assert d["stranding_split"]["stranded_land"] > 0.95
    assert d["stranded_where"] and sum(s["count"] for s in d["stranded_where"]) == round(d["stranding_probability"] * 200)
    for s in d["stranded_where"]:
        dist = land.distance(Point(s["lon"], s["lat"]))
        assert dist < 0.02, f"stranded cluster {s} is {dist:.3f} deg from land"


@test
def drift_cones_valid():
    f = _synthetic(cfg.REGIONS["gulf_of_mannar_palk_bay"], velocity=(0.0, -0.05))
    d = drift_forecast(9.0, 78.9, T0, mode="floating", horizon_hours=48, n_particles=500, field=f, seed=3)
    for s in d["snapshots"][1:]:
        for key in ("cone50", "cone90"):
            g = s[key]
            assert g is not None and g["type"] in ("Polygon", "MultiPolygon"), (key, s["cone_method"])
            geom = shape(g)
            assert geom.is_valid and geom.area > 0
            x, y = geom.representative_point().x, geom.representative_point().y
            assert 77 < x < 81 and 8 < y < 11, "cone coordinates must be GeoJSON [lon, lat]"
        c50, c90 = shape(s["cone50"]), shape(s["cone90"])
        assert c50.area <= c90.area * 1.0001
        pts = np.array(s["points"])
        inside90 = shapely.contains_xy(c90, pts[:, 1], pts[:, 0]).mean()
        assert inside90 >= 0.75, f"cone90 holds only {inside90:.2f} of sampled points"


@test
def drift_seabed_subcritical_does_not_move():
    below = cfg.SEABED_U_CRIT_MPS * (1 - cfg.SEABED_U_CRIT_SPREAD) * 0.8
    f = _synthetic(cfg.REGIONS["gulf_of_mannar_palk_bay"], velocity=(0.8, 0.0), bottom_velocity=(below, 0.0))
    d = drift_forecast(DEMO_LAT, DEMO_LON, T0, mode="seabed", horizon_hours=96, n_particles=200, field=f, seed=2)
    assert d["displacement_m"]["max"] == 0.0, d["displacement_m"]
    assert d["mobility"]["particles_ever_moved"] == 0.0
    assert d["snapshots"][-1]["cone90"] is None
    above = cfg.SEABED_U_CRIT_MPS * (1 + cfg.SEABED_U_CRIT_SPREAD) * 1.2
    f2 = _synthetic(cfg.REGIONS["gulf_of_mannar_palk_bay"], velocity=(0.0, 0.0), bottom_velocity=(0.0, -above))
    d2 = drift_forecast(DEMO_LAT, DEMO_LON, T0, mode="seabed", horizon_hours=12, n_particles=100, field=f2, seed=2)
    expected = cfg.SEABED_MOBILITY_FRACTION * above * 12 * 3600
    med = d2["displacement_m"]["median"]
    assert abs(med - expected) / expected < 0.05, (med, expected)


@test
def drift_time_shift_is_stated():
    f, (t0, t1) = _gom_field_window()
    d = drift_forecast(DEMO_LAT, DEMO_LON, t0 - timedelta(days=400), mode="floating", horizon_hours=24,
                       n_particles=50, field=FIELD, seed=0)
    assert d["available"] and d["time_mapping"]["shifted"] is True
    assert any("outside the bundled current window" in a for a in d["assumptions"])
    d2 = drift_forecast(DEMO_LAT, DEMO_LON, (t1 - timedelta(hours=6)).isoformat(), mode="floating",
                        horizon_hours=24, n_particles=20, field=FIELD, seed=0)
    assert d2["available"] and d2["time_mapping"]["shifted"] is False and d2["horizon_hours"] <= 6.0
    assert any("truncated" in x for x in d2["limitations"])


@test
def drift_refuses_without_data():
    d = drift_forecast(15.0, 73.0, T0, mode="floating", horizon_hours=24, n_particles=10, field=FIELD)
    assert d["available"] is False and "no bundled current data" in d["reason"]


# --- safety ------------------------------------------------------------------------------------

@test
def safety_outputs_present():
    f, (t0, _) = _gom_field_window()
    h = habitat_context(DEMO_LAT, DEMO_LON)
    d = drift_forecast(DEMO_LAT, DEMO_LON, t0, mode="seabed", horizon_hours=24, n_particles=50, field=FIELD)
    target = {"latitude": DEMO_LAT, "longitude": DEMO_LON, "dimensions": {"length_m": 25.0, "width_m": 3.0},
              "seabed_depth_m": None}
    p = people_safety(target, h, d, FIELD, load_bathymetry())
    ph, db = p["propeller_hazard"], p["diver_brief"]
    assert ph["level"] in ("high", "moderate", "low", "unknown") and isinstance(ph["reasons"], list)
    assert set(ph["terms"]) == {"shallow", "floating", "harbour_near", "harbour_drift", "large_net"}
    assert ph["terms"]["large_net"]["applied"] is True
    for key in ("seabed_depth_m", "net_size_m", "current_mps_at_depth", "within_recreational_limit",
                "current_ok_for_divers", "entanglement_risk", "recommended_method", "notes"):
        assert key in db, key
    assert db["seabed_depth_m"] is not None and "ETOPO" in db["depth_source"]
    assert db["current_mps_at_depth"] is not None
    target2 = {"latitude": DEMO_LAT, "longitude": DEMO_LON, "seabed_depth_m": 45.0}
    p2 = people_safety(target2, h, d, FIELD, None)
    assert p2["diver_brief"]["within_recreational_limit"] is False
    assert "ROV" in p2["diver_brief"]["recommended_method"]
    p3 = people_safety({"latitude": 15.0, "longitude": 73.0}, {"available": False}, {"available": False}, None,
                       type("NoBathy", (), {"depth_at": lambda self, a, b: {"depth_m": None, "note": "none"}})())
    assert p3["propeller_hazard"]["level"] == "unknown", p3["propeller_hazard"]


@test
def engine_contract():
    import ghosttrace.currents as c
    import ghosttrace.layers as l
    assert callable(c.load_default_field) and callable(l.load_default_layers) and callable(l.load_bathymetry)
    assert isinstance(l.load_default_layers().harbours, list)
    assert isinstance(cfg.DATA_SOURCES, list) and cfg.DATA_SOURCES
    for s in cfg.DATA_SOURCES:
        assert {"name", "url", "licence", "snapshot", "used_for"} <= set(s)


# --- the real forecast -------------------------------------------------------------------------

@test
def real_forecast_demo_position():
    f, (t0, t1) = _gom_field_window()
    start = t0 + timedelta(hours=24)
    print(f"\n  REAL FORECAST from {DEMO_LAT} N {DEMO_LON} E, start {start.isoformat()}, "
          f"currents: {f.describe()['name']} (latest run {f.model_run}), window {t0} .. {t1}")
    h = habitat_context(DEMO_LAT, DEMO_LON)
    print(f"  habitat score {h['score']}; nearest: " + "; ".join(
        f"{n['kind']} '{n['name']}' {n['distance_m'] / 1000:.2f} km @ {n['bearing_deg']}" for n in h["nearest"]))
    for mode in ("floating", "seabed"):
        t = time.time()
        d = drift_forecast(DEMO_LAT, DEMO_LON, start, mode=mode, horizon_hours=240, n_particles=500,
                           dt_minutes=30, field=FIELD, seed=0)
        assert d["available"] and not d["synthetic_currents"]
        assert len(d["snapshots"]) == 21
        print(f"  [{mode}] {time.time() - t:.1f}s  horizon {d['horizon_hours']} h, mean current "
              f"{d['mean_current_ms']} m/s, displacement median {d['displacement_m']['median'] / 1000:.2f} km "
              f"p90 {d['displacement_m']['p90'] / 1000:.2f} km, stranding {d['stranding_probability']:.2f} "
              f"{d['stranding_split']}, left_domain {d['left_domain']}")
        for i in d["impacts"][:6]:
            print(f"      impact {i['kind']:<15} {i['name'][:48]:<48} p={i['probability']:.2f} "
                  f"first {i['first_arrival_hours']} h")
        for s in d["stranded_where"][:3]:
            print(f"      stranded {s['count']} near ({s['lat']}, {s['lon']}) {s.get('name', '')}")
        if mode == "seabed":
            mob = d["mobility"]
            print(f"      mobility: ever moved {mob['particles_ever_moved']}, max bottom speed at start "
                  f"{mob['bottom_speed_at_start']['max_mps']} m/s vs U_CRIT {mob['u_crit_mps']} m/s "
                  f"(model bottom level {mob['bottom_level_depth_m_at_start']} m)")
        tgt = {"latitude": DEMO_LAT, "longitude": DEMO_LON, "dimensions": {"length_m": 12.0, "width_m": 4.0}}
        p = people_safety(tgt, h, d, FIELD, load_bathymetry())
        print(f"      propeller {p['propeller_hazard']['level']}; diver: {p['diver_brief']['summary']}")


def main() -> int:
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    for name, _, msg in failed:
        print(f"  FAILED {name}: {msg}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
