"""Where could this net go: Monte Carlo Lagrangian drift over the bundled currents.

    drift_forecast(lat, lon, start_time, *, mode="seabed" | "floating",
                   horizon_hours=240, n_particles=500, dt_minutes=30,
                   field=None, layers=None, seed=0) -> dict

A cloud of particles starts at the net's position. Each step every particle is
advected by the model current (RK4 by default, RK2 selectable), then given a
random-walk displacement standing in for motion the model cannot resolve.
Particles that enter the land layer strand; particles that leave the current
grid are frozen and counted as left_domain. Every 12 hours the cloud is
summarised as sampled points and 50% / 90% probability areas, and every step
the cloud is tested against buffered habitat and harbour features, giving an
impact probability (fraction of particles that ever came within the buffer)
and a first arrival time.

TWO MODES, AND WHY THEY ARE DIFFERENT QUESTIONS
    floating   A net held up by floats, or a buoyant tangle in the water column.
               Moved by the SURFACE current. Wind drag (windage) matters for a
               floating net, but no wind data is bundled, so it is not included
               and the output says so. A floating forecast therefore
               under-represents wind-driven drift toward the coast.

    seabed     A net lying on the bottom, which side-scan sonar sees most often.
               Moved by the NEAR-BOTTOM current (the model's deepest valid
               level in each cell), and only through a MOBILITY HEURISTIC:

                   speed = |U_bottom|
                   moving when speed > U_CRIT_i
                   velocity = SEABED_MOBILITY_FRACTION * U_bottom, else 0

               U_CRIT_i is drawn per particle uniformly within
               +/- SEABED_U_CRIT_SPREAD of SEABED_U_CRIT_MPS, so the ensemble
               carries the fact that nobody knows the threshold. The numbers
               are uncalibrated assumptions (see config_geo.py): no published
               mobility threshold for lost nets was found, and derelict gillnets
               are documented remaining in place for years when snagged (Good
               et al. 2010, Mar. Pollut. Bull. 60:39-50). A seabed forecast is
               POSSIBLE displacement if the net is not snagged, not an expected
               track. A particle that is not moving is not diffused either: a
               net that the current cannot lift does not random-walk.

TIME
    Forecast times are hours after the requested start. When the requested start
    lies outside the bundled current window (the demo survey's time is not the
    date the currents were downloaded), the forecast runs on the nearest window
    that fits and says so in `time_mapping` and `assumptions`: same physics,
    different calendar dates, so the result is an illustration of drift in this
    season's model currents, not a forecast for the survey's actual dates. When
    the start is inside the window but the horizon runs past its end, the
    horizon is truncated and `effective_horizon_hours` says by how much.

LAND AND COAST
    Two things can stop a particle at the coast, and they are counted apart:
      stranded_land      entered the bundled land layer (OpenStreetMap coastline);
                         held at its last position in water
      stalled_model_land reached cells the current model treats as land while
                         still in water by the land layer (the model's ~9 km
                         grid does not resolve narrow channels and very shallow
                         coast); held there. This is a model-resolution stop,
                         not a verified stranding.
    stranding_probability counts both, and the split is reported.

CONES
    For each snapshot the particle positions are projected to local metres,
    binned on a CONE_GRID_CELLS x CONE_GRID_CELLS grid and smoothed with a
    Gaussian kernel whose width follows Scott's rule (n^(-1/6) times the
    cloud's standard deviation per axis). The 50% (90%) cone is the smallest
    set of cells holding 50% (90%) of the smoothed mass: a highest-density
    region. It may be a MultiPolygon when the cloud splits. When fewer than
    CONE_MIN_DISTINCT distinct positions exist or the cloud is smaller than
    CONE_MIN_EXTENT_M, no density is estimated and the cone is the convex hull
    of the 50% (90%) of particles nearest the cloud centre (method recorded),
    or null when all particles share one position. Coordinates are GeoJSON
    [lon, lat]; `points` are [lat, lon] as the output contract asks.

DETERMINISM
    One numpy Generator seeded with `seed` drives the random walk, the per
    particle thresholds and the snapshot subsample, so the same inputs give the
    same output bit for bit.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import shapely
import shapely.ops
from scipy.ndimage import gaussian_filter
from shapely.geometry import MultiPoint, Point, mapping

from ghosttrace import config_geo as cfg
from ghosttrace.currents import FieldCollection, iso, load_default_field, to_epoch_seconds
from ghosttrace.layers import GEOD, LayerSet, load_default_layers

_WGS84_A = 6378137.0
_WGS84_E2 = 6.69437999014e-3


def _metres_per_degree(lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(metres per degree latitude, metres per degree longitude) on WGS84."""
    s = np.sin(np.radians(lat))
    w = 1.0 - _WGS84_E2 * s * s
    m_lat = np.radians(1.0) * _WGS84_A * (1.0 - _WGS84_E2) / w ** 1.5
    m_lon = np.radians(1.0) * _WGS84_A / np.sqrt(w) * np.cos(np.radians(lat))
    return m_lat, np.maximum(m_lon, 1e-6)


def _unavailable(reason: str, **extra: Any) -> dict[str, Any]:
    return {"available": False, "reason": reason, **extra}


# --- cones -------------------------------------------------------------------------------


def _cone(lat: np.ndarray, lon: np.ndarray, mass: float) -> tuple[dict[str, Any] | None, str]:
    """(GeoJSON geometry or None, method) for the highest-density region holding `mass`."""
    lat0, lon0 = float(np.mean(lat)), float(np.mean(lon))
    m_lat, m_lon = _metres_per_degree(np.array([lat0]))
    x = (lon - lon0) * m_lon[0]
    y = (lat - lat0) * m_lat[0]

    def to_lonlat(geom):
        return shapely.transform(geom, lambda c: np.column_stack([c[:, 0] / m_lon[0] + lon0,
                                                                  c[:, 1] / m_lat[0] + lat0]))

    distinct = len(np.unique(np.round(np.column_stack([x, y]), 1), axis=0))
    extent = max(float(np.ptp(x)), float(np.ptp(y)))
    if distinct <= 2 or extent < 1e-6:
        return None, "null: all particles at (nearly) one position"
    if distinct < cfg.CONE_MIN_DISTINCT or extent < cfg.CONE_MIN_EXTENT_M:
        d = np.hypot(x - np.median(x), y - np.median(y))
        keep = np.argsort(d)[:max(3, int(math.ceil(mass * len(d))))]
        hull = MultiPoint(np.column_stack([x[keep], y[keep]])).convex_hull
        if hull.area <= 0:
            return None, "null: nearest particles are collinear"
        return mapping(to_lonlat(hull)), "convex hull of nearest particles (too few distinct positions for a density)"

    n = len(x)
    sx, sy = float(np.std(x)), float(np.std(y))
    h = n ** (-1.0 / 6.0)
    bx, by = max(sx * h, 1.0), max(sy * h, 1.0)
    pad_x, pad_y = 3 * bx, 3 * by
    nx = ny = cfg.CONE_GRID_CELLS
    xe = np.linspace(x.min() - pad_x, x.max() + pad_x, nx + 1)
    ye = np.linspace(y.min() - pad_y, y.max() + pad_y, ny + 1)
    hist, _, _ = np.histogram2d(x, y, bins=[xe, ye])
    cx, cy = xe[1] - xe[0], ye[1] - ye[0]
    dens = gaussian_filter(hist, sigma=(bx / cx, by / cy), mode="constant")
    total = dens.sum()
    if total <= 0:
        return None, "null: empty density"
    flat = np.sort(dens.ravel())[::-1]
    cum = np.cumsum(flat) / total
    thresh = flat[min(int(np.searchsorted(cum, mass)), len(flat) - 1)]
    ii, jj = np.nonzero(dens >= thresh)
    boxes = shapely.box(xe[ii], ye[jj], xe[ii + 1], ye[jj + 1])
    region = shapely.union_all(boxes).simplify(min(cx, cy) * 0.5)
    if region.is_empty or region.area <= 0:
        return None, "null: density region empty"
    return mapping(to_lonlat(region)), (f"highest-density region of a Gaussian KDE (Scott bandwidth "
                                        f"{bx:.0f} m x {by:.0f} m, {nx}x{ny} grid)")


# --- the forecast ---------------------------------------------------------------------------


def drift_forecast(lat: float, lon: float, start_time: Any, *, mode: str = "seabed",
                   horizon_hours: float = cfg.DRIFT_DEFAULT_HORIZON_H,
                   n_particles: int = cfg.DRIFT_DEFAULT_PARTICLES,
                   dt_minutes: float = cfg.DRIFT_DEFAULT_DT_MIN,
                   field: Any = None, layers: LayerSet | None = None, seed: int = 0,
                   integrator: str | None = None, snapshot_every_hours: float | None = None,
                   max_snapshot_points: int | None = None) -> dict[str, Any]:
    """Monte Carlo drift forecast for one object. See the module docstring.

    snapshot_every_hours / max_snapshot_points override DRIFT_SNAPSHOT_EVERY_H /
    DRIFT_MAX_SNAPSHOT_POINTS for this call only (the field-kit live comparison
    wants hourly snapshots; the defaults are unchanged).
    """
    if mode not in ("seabed", "floating"):
        raise ValueError(f"mode must be 'seabed' or 'floating', not {mode!r}")
    integrator = (integrator or cfg.DRIFT_INTEGRATOR).lower()
    if integrator not in ("rk2", "rk4"):
        raise ValueError("integrator must be rk2 or rk4")
    n_particles = int(n_particles)
    base = {"mode": mode, "requested_horizon_hours": float(horizon_hours), "n_particles": n_particles}
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return _unavailable("position is not a finite number", **base)

    layers = layers if layers is not None else load_default_layers()
    field = field if field is not None else load_default_field()
    if field is None:
        return _unavailable("no current data bundled (run tools/fetch_ghosttrace_data.py --only currents); "
                            "GhostTrace does not substitute synthetic currents", **base)
    f = field.field_for(lat, lon) if isinstance(field, FieldCollection) else field
    if f is None or not bool(f.contains(lat, lon)):
        return _unavailable(f"no bundled current data for this location ({lat:.4f}, {lon:.4f})", **base)

    coverage = layers.region_for(lat, lon)
    region = coverage["region"] if coverage["covered"] else None
    land = layers.land(region) if region else None
    if land is not None and bool(shapely.contains_xy(land, lon, lat)):
        return _unavailable("start position lies on land in the bundled land layer; check the position",
                            **base, region=region)

    level = "bottom" if mode == "seabed" else "surface"
    assumptions: list[str] = []
    limitations: list[str] = []

    # -- time mapping ----------------------------------------------------------------------
    s_req = to_epoch_seconds(start_time)
    t0, t1 = float(f.times_s[0]), float(f.times_s[-1])
    horizon_s = float(horizon_hours) * 3600.0
    shifted = False
    if s_req < t0 or s_req > t1:
        s_model = max(t0, min(s_req, t1 - horizon_s))
        shifted = True
        assumptions.append(
            f"requested start {iso(s_req)} is outside the bundled current window {iso(t0)} to {iso(t1)}; "
            f"the forecast runs on model currents starting {iso(s_model)} "
            f"({(s_model - s_req) / 86400.0:+.1f} days from the requested start). This shows drift in real "
            "model currents for this location and season window, NOT a forecast for the requested dates.")
    else:
        s_model = s_req
    eff_horizon_s = min(horizon_s, t1 - s_model)
    if eff_horizon_s < horizon_s - 1:
        limitations.append(f"horizon truncated from {horizon_hours:g} h to {eff_horizon_s / 3600.0:.1f} h: "
                           f"the bundled current window ends {iso(t1)}")
    dt = float(dt_minutes) * 60.0
    n_steps = int(math.floor(eff_horizon_s / dt + 1e-9))
    if n_steps < 1:
        return _unavailable("the bundled current window leaves no time to integrate", **base)

    rng = np.random.default_rng(seed)
    plat = np.full(n_particles, float(lat))
    plon = np.full(n_particles, float(lon))
    state = np.zeros(n_particles, dtype=np.int8)  # 0 active, 1 stranded_land, 2 stalled_model_land, 3 left_domain
    K = cfg.DRIFT_K_BOTTOM_M2S if mode == "seabed" else cfg.DRIFT_K_SURFACE_M2S
    if mode == "seabed":
        spread = cfg.SEABED_U_CRIT_SPREAD
        ucrit = cfg.SEABED_U_CRIT_MPS * rng.uniform(1.0 - spread, 1.0 + spread, n_particles)
    else:
        ucrit = None
    max_points = int(max_snapshot_points) if max_snapshot_points is not None else cfg.DRIFT_MAX_SNAPSHOT_POINTS
    sample_idx = np.sort(rng.choice(n_particles, size=min(max_points, n_particles), replace=False))

    mobile_steps = np.zeros(n_particles, dtype=np.int32)
    speed_track: list[float] = []

    def velocity(la: np.ndarray, lo: np.ndarray, s: float, idx: np.ndarray | None = None):
        u, v = f.sample(la, lo, s, level)
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        if mode == "seabed":
            speed = np.hypot(u, v)
            uc = ucrit if idx is None else ucrit[idx]
            moving = np.isfinite(speed) & (speed > uc)
            factor = np.where(moving, cfg.SEABED_MOBILITY_FRACTION, 0.0)
            u = np.where(np.isfinite(u), u * factor, np.nan)
            v = np.where(np.isfinite(v), v * factor, np.nan)
            return u, v, moving
        return u, v, np.isfinite(u)

    def to_deg(u, v, la):
        m_lat, m_lon = _metres_per_degree(la)
        return v / m_lat, u / m_lon

    # -- impact features -------------------------------------------------------------------
    groups: dict[tuple, dict[str, Any]] = {}
    feature_list = []
    if region:
        for kind in cfg.IMPACT_KINDS:
            buf = cfg.IMPACT_BUFFER_M.get(kind, 0.0)
            for feat in layers.buffered_features(kind, buf, region):
                p = feat["properties"]
                key = (kind, p.get("name"), feat["layer"])
                g = groups.setdefault(key, {
                    "kind": kind, "name": p.get("name"), "layer": feat["layer"], "buffer_m": buf,
                    "source": p.get("source"), "qualities": set(), "n_features": 0,
                    "first": np.full(n_particles, np.inf)})
                g["qualities"].add(p.get("geometry_quality"))
                g["n_features"] += 1
                feature_list.append((key, feat["geometry"], np.array(feat["geometry"].bounds)))
    fbounds = np.array([b for _, _, b in feature_list]) if feature_list else np.zeros((0, 4))

    def check_impacts(t_hours: float) -> None:
        if not feature_list:
            return
        w, s, e, n = plon.min(), plat.min(), plon.max(), plat.max()
        cand = np.nonzero((fbounds[:, 0] <= e) & (fbounds[:, 2] >= w) & (fbounds[:, 1] <= n) & (fbounds[:, 3] >= s))[0]
        for ci in cand:
            key, geom, b = feature_list[ci]
            first = groups[key]["first"]
            pending = np.isinf(first) & (plon >= b[0]) & (plon <= b[2]) & (plat >= b[1]) & (plat <= b[3])
            if not pending.any():
                continue
            ids = np.nonzero(pending)[0]
            hit = shapely.contains_xy(geom, plon[ids], plat[ids])
            first[ids[hit]] = t_hours

    snapshots: list[dict[str, Any]] = []
    every_h = float(snapshot_every_hours) if snapshot_every_hours is not None else cfg.DRIFT_SNAPSHOT_EVERY_H
    every = max(1, int(round(every_h * 3600.0 / dt)))

    def snapshot(step: int) -> None:
        t_h = step * dt / 3600.0
        pts = [[round(float(plat[i]), 5), round(float(plon[i]), 5)] for i in sample_idx]
        snap = {"t_hours": round(t_h, 2), "time": iso(s_req + step * dt), "model_time": iso(s_model + step * dt),
                "points": pts,
                "counts": {"active": int((state == 0).sum()), "stranded_land": int((state == 1).sum()),
                           "stalled_model_land": int((state == 2).sum()), "left_domain": int((state == 3).sum())}}
        if step == 0:
            snap["cone50"] = snap["cone90"] = None
            snap["cone_method"] = "null at t = 0"
        else:
            snap["cone50"], m50 = _cone(plat, plon, 0.5)
            snap["cone90"], m90 = _cone(plat, plon, 0.9)
            snap["cone_method"] = m90 if m50 == m90 else f"50%: {m50}; 90%: {m90}"
        snapshots.append(snap)

    check_impacts(0.0)
    snapshot(0)
    s, w_, n_, e_ = f.bounds
    kstep = cfg.DRIFT_IMPACT_CHECK_EVERY_STEPS
    max_step_m = 0.0

    for step in range(1, n_steps + 1):
        s_now = s_model + (step - 1) * dt
        act = np.nonzero(state == 0)[0]
        if act.size:
            la, lo = plat[act], plon[act]
            u1, v1, mov1 = velocity(la, lo, s_now, act)
            speed_track.append(float(np.nanmean(np.hypot(*f.sample(la, lo, s_now, level)))) if act.size else float("nan"))
            dlat1, dlon1 = to_deg(u1, v1, la)
            if integrator == "rk2":
                u2, v2, _ = velocity(la + 0.5 * dt * dlat1, lo + 0.5 * dt * dlon1, s_now + 0.5 * dt, act)
                dlat, dlon = to_deg(u2, v2, la + 0.5 * dt * dlat1)
                inc_lat, inc_lon = dt * dlat, dt * dlon
            else:
                lb, ob = la + 0.5 * dt * dlat1, lo + 0.5 * dt * dlon1
                u2, v2, _ = velocity(lb, ob, s_now + 0.5 * dt, act)
                dlat2, dlon2 = to_deg(u2, v2, lb)
                lc, oc = la + 0.5 * dt * dlat2, lo + 0.5 * dt * dlon2
                u3, v3, _ = velocity(lc, oc, s_now + 0.5 * dt, act)
                dlat3, dlon3 = to_deg(u3, v3, lc)
                ld, od = la + dt * dlat3, lo + dt * dlon3
                u4, v4, _ = velocity(ld, od, s_now + dt, act)
                dlat4, dlon4 = to_deg(u4, v4, ld)
                inc_lat = dt / 6.0 * (dlat1 + 2 * dlat2 + 2 * dlat3 + dlat4)
                inc_lon = dt / 6.0 * (dlon1 + 2 * dlon2 + 2 * dlon3 + dlon4)
            # A later RK stage that stepped onto model land falls back to the first-stage (Euler) increment.
            bad = ~np.isfinite(inc_lat) | ~np.isfinite(inc_lon)
            euler_lat, euler_lon = dt * dlat1, dt * dlon1
            inc_lat = np.where(bad, euler_lat, inc_lat)
            inc_lon = np.where(bad, euler_lon, inc_lon)
            # No valid current even at the particle's own position: model land.
            dead = ~np.isfinite(inc_lat) | ~np.isfinite(inc_lon)
            moving = mov1 & ~dead
            mobile_steps[act[moving]] += 1
            if K > 0:
                m_lat, m_lon = _metres_per_degree(la)
                sigma = math.sqrt(2.0 * K * dt)
                rw = rng.standard_normal((2, act.size)) * sigma
                rw = np.where(moving, rw, 0.0)
                inc_lat = np.where(dead, 0.0, inc_lat) + rw[1] / m_lat
                inc_lon = np.where(dead, 0.0, inc_lon) + rw[0] / m_lon
            else:
                inc_lat = np.where(dead, 0.0, inc_lat)
                inc_lon = np.where(dead, 0.0, inc_lon)
            new_lat, new_lon = la + inc_lat, lo + inc_lon
            m_lat, m_lon = _metres_per_degree(la)
            if act.size:
                max_step_m = max(max_step_m, float(np.max(np.hypot(inc_lat * m_lat, inc_lon * m_lon))))
            outside = ~((new_lat >= s) & (new_lat <= n_) & (new_lon >= w_) & (new_lon <= e_))
            on_land = np.zeros(act.size, dtype=bool)
            if land is not None:
                on_land = shapely.contains_xy(land, new_lon, new_lat) & ~outside
            # stranded particles keep their last position in water; others move
            keep = on_land
            plat[act] = np.where(keep, la, new_lat)
            plon[act] = np.where(keep, lo, new_lon)
            new_state = np.where(on_land, 1, np.where(outside, 3, np.where(dead, 2, 0)))
            state[act] = new_state
        if step % kstep == 0 or step == n_steps:
            check_impacts(step * dt / 3600.0)
        if step % every == 0 or step == n_steps:
            snapshot(step)

    # -- summaries ---------------------------------------------------------------------------
    impacts = []
    for key, g in groups.items():
        reached = np.isfinite(g["first"])
        if not reached.any():
            continue
        name = g["name"] or f"unnamed {g['kind'].replace('_', ' ')} feature(s)"
        impacts.append({
            "kind": g["kind"], "name": name, "named": g["name"] is not None, "layer": g["layer"],
            "probability": round(float(reached.mean()), 4),
            "first_arrival_hours": round(float(g["first"][reached].min()), 2),
            "median_arrival_hours": round(float(np.median(g["first"][reached])), 2),
            "buffer_m": g["buffer_m"], "features_in_group": g["n_features"],
            "source": g["source"], "geometry_quality": sorted(q for q in g["qualities"] if q)})
    impacts.sort(key=lambda r: (-r["probability"], r["first_arrival_hours"]))

    stranded = (state == 1) | (state == 2)
    stranded_where = []
    if stranded.any():
        cell = cfg.STRAND_GROUP_DEG
        keys = np.column_stack([np.floor(plat[stranded] / cell), np.floor(plon[stranded] / cell)]).astype(int)
        uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        sl, so, ss = plat[stranded], plon[stranded], state[stranded]
        named = [(p, feat) for p in ("turtle_nesting", "harbour", "protected_area", "dugong")
                 for feat in (layers.buffered_features(p, 0.0, region) if region else [])]
        for gi in np.argsort(-counts):
            members = inv.ravel() == gi
            clat, clon = float(np.mean(sl[members])), float(np.mean(so[members]))
            rec = {"lat": round(clat, 5), "lon": round(clon, 5), "count": int(counts[gi]),
                   "stranded_land": int((ss[members] == 1).sum()), "stalled_model_land": int((ss[members] == 2).sum())}
            best = None
            for kind, feat in named:
                nm = feat["properties"].get("name")
                if not nm:
                    continue
                gb = feat["geometry"]
                pt = shapely.ops.nearest_points(gb, Point(clon, clat))[0]
                _, _, dist = GEOD.inv(clon, clat, pt.x, pt.y)
                if dist <= cfg.STRAND_NAME_RADIUS_M and (best is None or dist < best[0]):
                    best = (dist, nm, kind)
            if best:
                rec["name"] = best[1]
                rec["name_kind"] = best[2]
                rec["name_distance_m"] = round(best[0], 0)
            stranded_where.append(rec)

    _, _, disp = GEOD.inv(np.full(n_particles, lon), np.full(n_particles, lat), plon, plat)
    disp = np.asarray(disp, dtype=float)
    start_speed = f.max_speed(lat, lon, s_model, min(eff_horizon_s / 3600.0, 24 * 10), level)
    mean_speed = float(np.nanmean(speed_track)) if speed_track and np.isfinite(speed_track).any() else None

    if mode == "seabed":
        mobility = {
            "applies": True, "level": "bottom",
            "u_crit_mps": cfg.SEABED_U_CRIT_MPS, "u_crit_spread": cfg.SEABED_U_CRIT_SPREAD,
            "mobility_fraction": cfg.SEABED_MOBILITY_FRACTION,
            "particles_ever_moved": round(float((mobile_steps > 0).mean()), 4),
            "mean_fraction_of_time_mobile": round(float(mobile_steps.mean() / n_steps), 4),
            "bottom_speed_at_start": start_speed,
            "bottom_level_depth_m_at_start": f.bottom_depth_at(lat, lon) if hasattr(f, "bottom_depth_at") else None,
            "basis": ("UNCALIBRATED ASSUMPTION: moves only while the near-bottom current exceeds a per-particle "
                      "threshold, then at a fraction of it. No published net-mobility threshold was found; "
                      "snagged nets may not move at all (Good et al. 2010, Mar. Pollut. Bull. 60:39-50)."),
            "interpretation": ("possible displacement if the net is not snagged; a stationary result means the "
                               "modelled near-bottom current never exceeded the assumed threshold"),
        }
    else:
        mobility = {"applies": False, "level": "surface", "surface_speed_at_start": start_speed,
                    "basis": "floating object assumed to move with the surface current (no threshold)"}

    synthetic = bool(getattr(f, "synthetic", False))
    desc = f.describe()
    assumptions += [
        f"{'near-bottom' if mode == 'seabed' else 'surface'} current from {desc.get('name')}"
        + (" [SYNTHETIC TEST FIELD]" if synthetic else ""),
        f"horizontal random walk with K = {K:g} m^2/s (uncalibrated assumption), applied only to moving particles"
        if mode == "seabed" else f"horizontal random walk with K = {K:g} m^2/s (uncalibrated assumption)",
        f"{integrator.upper()} advection, dt = {dt_minutes:g} min, {n_particles} particles, seed {seed}",
        "all particles start at the reported position; position uncertainty of the detection is not modelled",
        f"a particle 'reaches' a feature when inside its buffer: {cfg.IMPACT_BUFFER_M}",
    ]
    if mode == "seabed":
        assumptions.append(f"seabed mobility: U_CRIT {cfg.SEABED_U_CRIT_MPS:g} m/s +/- "
                           f"{100 * cfg.SEABED_U_CRIT_SPREAD:.0f}%, moving at {cfg.SEABED_MOBILITY_FRACTION:g} x "
                           "near-bottom current; heuristic, uncalibrated")
    else:
        assumptions.append("wind not included: no wind data is bundled, so there is no windage term")
    limitations += [
        "model grid ~0.08 x 0.04 deg (~9 x 4.4 km): channels between islands, Pamban Pass and nearshore "
        "circulation are not resolved",
        "tides are only as represented in the global model output; no separate tidal forcing",
        "near the coast the current is extrapolated from the nearest valid model cells",
        "no beaching / re-floating physics: a stranded particle stays stranded",
        "no sinking, burial, biofouling or snagging dynamics beyond the seabed threshold heuristic",
        "habitat and harbour layers are incomplete (see data/ghosttrace/SOURCES.md); an impact list that "
        "omits a feature is not evidence the feature is safe",
    ]
    if max_step_m > 1000.0:
        limitations.append(f"largest single-step displacement {max_step_m:.0f} m exceeds the 1 km reef buffer; "
                           "a narrow buffer could be crossed between checks")
    if not coverage["covered"]:
        limitations.append("no bundled habitat/land layers for this location: stranding and impacts not assessed")
    if region and "turtle_nesting" not in coverage["kinds_present"]:
        limitations.append("no turtle nesting layer in this region")

    return {
        "available": True,
        "mode": mode, "level": level,
        "horizon_hours": round(eff_horizon_s / 3600.0, 2), "effective_horizon_hours": round(n_steps * dt / 3600.0, 2),
        "requested_horizon_hours": float(horizon_hours),
        "n_particles": n_particles, "dt_minutes": float(dt_minutes), "seed": int(seed), "integrator": integrator,
        "region": region,
        "current_source": {**desc, "synthetic": synthetic},
        "synthetic_currents": synthetic,
        "time_mapping": {"requested_start": iso(s_req), "model_start": iso(s_model), "shifted": shifted,
                         "offset_hours": round((s_model - s_req) / 3600.0, 2)},
        "snapshots": snapshots,
        "impacts": impacts,
        "stranding_probability": round(float(stranded.mean()), 4),
        "stranding_split": {"stranded_land": round(float((state == 1).mean()), 4),
                            "stalled_model_land": round(float((state == 2).mean()), 4)},
        "left_domain": int((state == 3).sum()),
        "left_domain_probability": round(float((state == 3).mean()), 4),
        "stranded_where": stranded_where,
        "displacement_m": {"median": round(float(np.median(disp)), 1), "p90": round(float(np.percentile(disp, 90)), 1),
                           "max": round(float(np.max(disp)), 1)},
        "mean_current_ms": None if mean_speed is None else round(mean_speed, 3),
        "mobility": mobility,
        "assumptions": assumptions,
        "limitations": limitations,
        "heuristic": cfg.HEURISTIC_LABEL,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
