"""Validate GhostTrace's floating drift against real NOAA Global Drifter Program drifters.

    .venv/bin/python tools/validate_drift_gdp.py                    # full run
    .venv/bin/python tools/validate_drift_gdp.py --max-segments 20  # smaller
    .venv/bin/python tools/validate_drift_gdp.py --no-calibration   # skip the K sweep

WHAT IT DOES
    1. Downloads 6-hourly quality-controlled positions of every GDP drifter in
       the Northern Indian Ocean (0-25 N, 50-100 E: Arabian Sea, Laccadive Sea,
       Bay of Bengal) from NOAA OSMC ERDDAP, dataset `drifter_6hour_qc`, for the
       period the HYCOM ESPC-D-V02 archive covers (from 2024-08-10) up to the
       dataset's last QC'd time.
    2. Cuts each drifter's track into 72 h segments with an unbroken 6-hourly
       record, at most --per-drifter per drifter and at least --gap-days apart,
       so one long-lived drifter cannot dominate the statistics.
    3. For each segment, downloads HYCOM ESPC-D-V02 surface u/v (z = 0 m, 3-hourly,
       1/12 deg) for a box around the observed track over the segment's dates,
       via HYCOM's THREDDS OPeNDAP archive, and caches it under
       data/ghosttrace/validation/currents/ (git-ignored, re-downloadable).
    4. Runs ghosttrace.drift.drift_forecast in mode "floating" from the
       drifter's real start position and time, and compares the forecast with
       the real track at 6, 12, 24, 48 and 72 h (metrics in ghosttrace/validation.py).
    5. Optionally re-runs every segment over a grid of horizontal diffusivities
       K and reports the smallest K whose 90% cone reaches nominal coverage. The
       engine's default K is NOT changed; the recommendation is written to the
       summary for a person to decide.

WHAT IT IS NOT
    The HYCOM archive fields are the model's ANALYSIS for those dates, not the
    forecast that was available beforehand, so this is a hindcast: an upper
    bound on forecast skill. Drifters are drogued at 15 m (SVP design) or have
    lost their drogue and ride at the surface with wind drag; GhostTrace has no
    windage, so undrogued drifters are expected to do worse and are reported
    separately. A drifter is a proxy for floating gear, not for a net on the
    seabed.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import requests

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ghosttrace import config_geo as cfg  # noqa: E402
from ghosttrace import validation as V  # noqa: E402
from ghosttrace.currents import CurrentField, to_epoch_seconds  # noqa: E402
from ghosttrace.drift import drift_forecast  # noqa: E402
from ghosttrace.layers import LayerSet  # noqa: E402

OUT = V.VALIDATION_DIR
UA = {"User-Agent": "DeepEcho-GhostTrace-validation/1.0 (SIH26057 research prototype)"}

ERDDAP = "https://osmc.noaa.gov/erddap/tabledap"
GDP_DATASET = "drifter_6hour_qc"
GDP_VARS = "ID,time,latitude,longitude,ve,vn,err_lat,err_lon,drogue_lost_date,typebuoy"
GDP_LICENCE = ("Creative Commons Attribution 4.0 (https://creativecommons.org/licenses/by/4.0/). "
               "Attribution requested by the dataset: 'This study used data collected and made freely "
               "available by the NOAA Global Drifter Program (https://www.aoml.noaa.gov/phod/gdp/)'")
GDP_CITATION = ("Lumpkin, R. and Centurioni, L. (2019). Global Drifter Program quality-controlled 6-hour "
                "interpolated data from ocean surface drifting buoys. NOAA National Centers for Environmental "
                "Information. doi:10.25921/7ntx-z961. Accessed through NOAA OSMC ERDDAP.")

HYCOM_OPENDAP = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/{var}3z/{year}"
HYCOM_LICENCE = ("Freely available (HYCOM.org THREDDS catalog: 'rights: Freely available'; distribution "
                 "statement: Approved for public release; distribution unlimited)")
HYCOM_FIRST = datetime(2024, 8, 10, 12, tzinfo=timezone.utc)

BOX = {"south": 0.0, "north": 25.0, "west": 50.0, "east": 100.0}
BOX_NAME = "Northern Indian Ocean (0-25 N, 50-100 E: Arabian Sea, Laccadive Sea, Bay of Bengal)"
HORIZONS = (6, 12, 24, 48, 72)
STEP_H = 6


def log(msg: str) -> None:
    print(f"[validate {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _get(url: str, *, timeout: float = 300, retries: int = 3) -> requests.Response:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as exc:  # retried, then raised
            last = exc
            log(f"  attempt {attempt + 1}/{retries} failed: {type(exc).__name__}: {str(exc)[:160]}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} attempts: {url[:200]} ({last})")


# --- drifters -----------------------------------------------------------------------------------


def dataset_end() -> datetime:
    r = _get(f"https://osmc.noaa.gov/erddap/info/{GDP_DATASET}/index.csv", timeout=120)
    for row in csv.reader(io.StringIO(r.text)):
        if len(row) >= 5 and row[1] == "NC_GLOBAL" and row[2] == "time_coverage_end":
            return datetime.fromisoformat(row[4].replace("Z", "+00:00"))
    raise RuntimeError("time_coverage_end not found in the ERDDAP dataset info")


def fetch_drifters(cache: Path, refresh: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    end = dataset_end()
    query = (f"{GDP_VARS}&latitude>={BOX['south']}&latitude<={BOX['north']}&longitude>={BOX['west']}"
             f"&longitude<={BOX['east']}&time>={HYCOM_FIRST:%Y-%m-%dT%H:%M:%SZ}&time<={end:%Y-%m-%dT%H:%M:%SZ}")
    url = f"{ERDDAP}/{GDP_DATASET}.csv?" + quote(query, safe="=&,")
    if refresh or not cache.exists():
        log(f"GDP: downloading {GDP_DATASET} for {BOX_NAME}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(_get(url).content)
    text = cache.read_text(encoding="utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    header = rows[0]
    out = []
    for row in rows[2:]:
        rec = dict(zip(header, row))
        try:
            out.append({
                "id": rec["ID"], "t": datetime.fromisoformat(rec["time"].replace("Z", "+00:00")),
                "lat": float(rec["latitude"]), "lon": float(rec["longitude"]),
                "ve": float(rec["ve"]) if rec["ve"] not in ("", "NaN") else float("nan"),
                "vn": float(rec["vn"]) if rec["vn"] not in ("", "NaN") else float("nan"),
                "drogue_lost": rec["drogue_lost_date"], "typebuoy": rec["typebuoy"],
            })
        except (KeyError, ValueError):
            continue
    meta = {"dataset_id": GDP_DATASET, "erddap": ERDDAP, "url": url, "time_coverage_end": end.isoformat(),
            "rows": len(out), "drifters": len({r["id"] for r in out}),
            "accessed": datetime.fromtimestamp(cache.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")}
    return out, meta


def drogue_status(lost: str, start: datetime, end: datetime) -> str:
    """'drogued', 'undrogued', or 'uncertain' for one segment from the GDP drogue_lost_date field.

    ERDDAP renders the dataset's fill value (drogue still attached) as empty and
    0 (status uncertain from the beginning) as 1970-01-01.
    """
    if lost in ("", "NaN"):
        return "drogued"
    t = datetime.fromisoformat(lost.replace("Z", "+00:00"))
    if t.year == 1970:
        return "uncertain"
    if t <= start:
        return "undrogued"
    if t >= end:
        return "drogued"
    return "uncertain"  # lost during the segment


def cut_segments(rows: list[dict[str, Any]], *, horizon_h: int, per_drifter: int, gap_days: float,
                 max_segments: int) -> list[dict[str, Any]]:
    by_id: dict[str, list] = defaultdict(list)
    for r in rows:
        by_id[r["id"]].append(r)
    n_obs = horizon_h // STEP_H + 1
    per: dict[str, list] = {}
    for did, track in sorted(by_id.items()):
        track.sort(key=lambda r: r["t"])
        candidates: list[dict[str, Any]] = []
        i = 0
        while i + n_obs <= len(track):
            win = track[i:i + n_obs]
            t0 = win[0]["t"]
            ok = (all((win[k]["t"] - t0) == timedelta(hours=STEP_H * k) for k in range(n_obs))
                  and t0 >= HYCOM_FIRST + timedelta(hours=3)
                  and t0.year == win[-1]["t"].year  # the HYCOM archive is split by year
                  and (win[-1]["t"] + timedelta(hours=3)).year == t0.year
                  and all(math.isfinite(w["lat"]) and math.isfinite(w["lon"]) for w in win)
                  and math.isfinite(win[0]["ve"]) and math.isfinite(win[0]["vn"]))
            if not ok:
                i += 1
                continue
            candidates.append({"drifter_id": did, "typebuoy": win[0]["typebuoy"], "start": t0,
                               "track": [[w["lat"], w["lon"]] for w in win],
                               "ve0": win[0]["ve"], "vn0": win[0]["vn"],
                               "drogue": drogue_status(win[0]["drogue_lost"], t0, win[-1]["t"])})
            # next candidate at least gap_days later
            t_next = t0 + timedelta(days=gap_days)
            while i < len(track) and track[i]["t"] < t_next:
                i += 1
        # Spread the chosen segments evenly over the drifter's usable record, not its first weeks.
        if len(candidates) > per_drifter:
            picks = sorted({int(round(x)) for x in np.linspace(0, len(candidates) - 1, per_drifter)})
            chosen = [candidates[k] for k in picks]
        else:
            chosen = candidates
        if chosen:
            per[did] = chosen
    # Round-robin across drifters so the sample spreads over as many drifters as possible.
    out: list[dict[str, Any]] = []
    depth = 0
    while len(out) < max_segments and any(len(v) > depth for v in per.values()):
        for did in sorted(per):
            if depth < len(per[did]) and len(out) < max_segments:
                out.append(per[did][depth])
        depth += 1
    out.sort(key=lambda s: (s["start"], s["drifter_id"]))
    return out


# --- currents -----------------------------------------------------------------------------------


class Hycom:
    """HYCOM ESPC-D-V02 archive surface u/v via OPeNDAP, axes read once per year."""

    def __init__(self):
        self._axes: dict[int, dict[str, Any]] = {}

    def _open(self, var: str, year: int):
        import netCDF4
        return netCDF4.Dataset(HYCOM_OPENDAP.format(var=var, year=year))

    def axes(self, year: int) -> dict[str, Any]:
        if year not in self._axes:
            import netCDF4
            with self._open("u", year) as ds:
                t = ds.variables["time"]
                times = netCDF4.num2date(t[:], t.units, only_use_cftime_datetimes=False,
                                         only_use_python_datetimes=True)
                self._axes[year] = {
                    "times_s": np.array([x.replace(tzinfo=timezone.utc).timestamp() for x in times]),
                    "lat": np.asarray(ds.variables["lat"][:], dtype=float),
                    "lon": np.asarray(ds.variables["lon"][:], dtype=float),
                    "generating_model": str(getattr(ds, "generating_model", "")),
                    "institution": str(getattr(ds, "institution", "")),
                }
        return self._axes[year]

    def fetch(self, path: Path, *, south: float, west: float, north: float, east: float,
              t0: datetime, t1: datetime) -> None:
        import netCDF4

        ax = self.axes(t0.year)
        s0, s1 = t0.timestamp(), t1.timestamp()
        k0 = int(np.searchsorted(ax["times_s"], s0, side="right") - 1)
        k1 = int(np.searchsorted(ax["times_s"], s1, side="left"))
        if k0 < 0 or k1 >= len(ax["times_s"]):
            raise RuntimeError(f"segment {t0}..{t1} is outside the HYCOM {t0.year} archive")
        i0 = max(0, int(np.searchsorted(ax["lat"], south)) - 1)
        i1 = min(len(ax["lat"]), int(np.searchsorted(ax["lat"], north)) + 1)
        j0 = max(0, int(np.searchsorted(ax["lon"], west)) - 1)
        j1 = min(len(ax["lon"]), int(np.searchsorted(ax["lon"], east)) + 1)
        data = {}
        for var in ("u", "v"):
            for attempt in range(4):
                try:
                    with self._open(var, t0.year) as ds:
                        ds.set_auto_maskandscale(True)
                        arr = ds.variables[f"water_{var}"][k0:k1 + 1, 0, i0:i1, j0:j1]
                    data[var] = np.ma.filled(arr.astype("float32"), np.nan)
                    break
                except Exception as exc:
                    log(f"  HYCOM {var} attempt {attempt + 1} failed: {type(exc).__name__}: {str(exc)[:120]}")
                    time.sleep(10 * (attempt + 1))
            else:
                raise RuntimeError("HYCOM OPeNDAP failed after 4 attempts")
        path.parent.mkdir(parents=True, exist_ok=True)
        epoch = "hours since 1970-01-01 00:00:00"
        tmp = path.with_suffix(".tmp")
        with netCDF4.Dataset(tmp, "w") as nc:
            nc.createDimension("time", k1 - k0 + 1)
            nc.createDimension("lat", i1 - i0)
            nc.createDimension("lon", j1 - j0)
            vt = nc.createVariable("time", "f8", ("time",))
            vt.units = epoch
            vt[:] = ax["times_s"][k0:k1 + 1] / 3600.0
            vla = nc.createVariable("lat", "f8", ("lat",))
            vla[:] = ax["lat"][i0:i1]
            vlo = nc.createVariable("lon", "f8", ("lon",))
            vlo[:] = ax["lon"][j0:j1]
            for name in ("u", "v"):
                var = nc.createVariable(f"{name}_surface", "i2", ("time", "lat", "lon"), zlib=True, complevel=6,
                                        fill_value=np.int16(-32768))
                var.scale_factor = 0.001
                var.add_offset = 0.0
                var.units = "m s-1"
                arr = data[name]
                var[:] = np.ma.masked_array(np.nan_to_num(arr, nan=0.0), mask=~np.isfinite(arr))
            nc.title = "HYCOM ESPC-D-V02 surface currents (z = 0 m), subset for GhostTrace drift validation"
            nc.source_url = HYCOM_OPENDAP.format(var="u|v", year=t0.year)
            nc.generating_model = ax["generating_model"]
            nc.institution = ax["institution"]
            nc.accessed = datetime.now(timezone.utc).isoformat(timespec="seconds")
            nc.note = "archive (analysis) fields: a hindcast, not the forecast available beforehand"
        tmp.replace(path)


def load_surface_field(path: Path) -> CurrentField:
    """A CurrentField with the cached surface level; the bottom level is all NaN (not fetched)."""
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        ds.set_auto_maskandscale(True)
        times_s = np.asarray(ds.variables["time"][:], dtype=float) * 3600.0
        u = np.ma.filled(ds.variables["u_surface"][:].astype("float32"), np.nan)
        v = np.ma.filled(ds.variables["v_surface"][:].astype("float32"), np.nan)
        attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
        lat = np.asarray(ds.variables["lat"][:], dtype=float)
        lon = np.asarray(ds.variables["lon"][:], dtype=float)
    nan = np.full_like(u, np.nan)
    attrs.update({"region": "validation", "accessed": attrs.get("accessed")})
    return CurrentField(lat=lat, lon=lon, times_s=times_s, u={"surface": u, "bottom": nan},
                        v={"surface": v, "bottom": nan.copy()}, attrs=attrs, path=path)


# --- forecasting --------------------------------------------------------------------------------


@contextmanager
def drift_settings(**values: Any):
    """Temporarily set config_geo knobs in THIS process only (the engine default is untouched)."""
    old = {k: getattr(cfg, k) for k in values}
    try:
        for k, v in values.items():
            setattr(cfg, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(cfg, k, v)


def run_forecast(seg: dict[str, Any], field: CurrentField, *, horizon_h: int, n_particles: int, k: float,
                 snapshot_every_h: float, seed: int) -> dict[str, Any]:
    lat0, lon0 = seg["track"][0]
    with drift_settings(DRIFT_K_SURFACE_M2S=float(k), DRIFT_SNAPSHOT_EVERY_H=float(snapshot_every_h),
                        DRIFT_MAX_SNAPSHOT_POINTS=int(n_particles)):
        return drift_forecast(lat0, lon0, seg["start"], mode="floating", horizon_hours=horizon_h,
                              n_particles=n_particles, dt_minutes=30, field=field, layers=LayerSet([]), seed=seed)


def score_segment(seg: dict[str, Any], fc: dict[str, Any], field: CurrentField,
                  horizons: tuple[int, ...]) -> dict[str, Any]:
    """Per-horizon metrics for one segment and forecast (see ghosttrace/validation.py)."""
    if not fc.get("available"):
        raise RuntimeError(f"forecast unavailable: {fc.get('reason')}")
    if fc["time_mapping"]["shifted"]:
        raise RuntimeError("forecast ran on shifted dates; the currents do not cover the segment")
    obs = seg["track"]
    lat0, lon0 = obs[0]
    snaps = {int(round(s["t_hours"])): s for s in fc["snapshots"]}
    hours_all = [STEP_H * i for i in range(len(obs))]
    mean_track = []
    for h in hours_all:
        pts = np.array(snaps[h]["points"], dtype=float) if h in snaps else None
        mean_track.append([float(pts[:, 0].mean()), float(pts[:, 1].mean())] if pts is not None else None)
    u0, v0 = field.sample(lat0, lon0, seg["start"], "surface")
    const_model = V.constant_velocity_track(lat0, lon0, u0, v0, hours_all) if math.isfinite(u0) else None
    const_obs = V.constant_velocity_track(lat0, lon0, seg["ve0"], seg["vn0"], hours_all)
    out: dict[str, Any] = {}
    for h in horizons:
        i = h // STEP_H
        if i >= len(obs) or mean_track[i] is None:
            continue
        snap = snaps[h]
        truth = obs[i]
        sub = slice(0, i + 1)
        row = {
            "sep_model_km": round(V.haversine_km(*mean_track[i], *truth), 3),
            "sep_persistence_km": round(V.haversine_km(lat0, lon0, *truth), 3),
            "sep_const_model_km": round(V.haversine_km(*const_model[i], *truth), 3) if const_model else None,
            "sep_const_obs_km": round(V.haversine_km(*const_obs[i], *truth), 3),
            "in50": V.point_in_geometry(snap.get("cone50"), *truth),
            "in90": V.point_in_geometry(snap.get("cone90"), *truth),
            "skill_model": V.liu_weisberg_skill(mean_track[sub], obs[sub]),
            "skill_const_model": V.liu_weisberg_skill(const_model[sub], obs[sub]) if const_model else None,
            "skill_const_obs": V.liu_weisberg_skill(const_obs[sub], obs[sub]),
            "observed_path_km": round(sum(V.haversine_km(*obs[j - 1], *obs[j]) for j in range(1, i + 1)), 3),
            "particles_left_box": snap["counts"]["left_domain"],
            "particles_stalled_model_land": snap["counts"]["stalled_model_land"],
        }
        for key in ("skill_model", "skill_const_model", "skill_const_obs"):
            if row[key] is not None:
                row[key] = round(row[key], 4)
        out[str(h)] = row
    return {"horizons": out, "mean_track": mean_track, "const_model": const_model, "const_obs": const_obs,
            "u0": u0, "v0": v0}


def _round_geom(geom: dict[str, Any] | None, tol_deg: float = 0.003) -> dict[str, Any] | None:
    if not geom:
        return None
    from shapely.geometry import mapping, shape

    g = shape(geom).simplify(tol_deg, preserve_topology=True)
    if g.is_empty:
        return None

    def rnd(c):
        if isinstance(c, (list, tuple)) and c and isinstance(c[0], (int, float)):
            return [round(float(c[0]), 4), round(float(c[1]), 4)]
        return [rnd(x) for x in c]

    m = mapping(g)
    return {"type": m["type"], "coordinates": rnd(m["coordinates"])}


def _forecast_job(job: tuple) -> dict[str, Any]:
    seg, cache, horizon, particles, k, every, seed, horizons = job
    try:
        field = load_surface_field(Path(cache))
        fc = run_forecast(seg, field, horizon_h=horizon, n_particles=particles, k=k, snapshot_every_h=every, seed=seed)
        return {"fc": fc, "scored": score_segment(seg, fc, field, horizons)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def _coverage_job(job: tuple) -> dict[int, tuple[bool, bool]]:
    seg, cache, horizon, particles, k, seed, hours = job
    field = load_surface_field(Path(cache))
    fc = run_forecast(seg, field, horizon_h=horizon, n_particles=particles, k=k, snapshot_every_h=24, seed=seed)
    snaps = {int(round(s["t_hours"])): s for s in fc["snapshots"]}
    out = {}
    for h in hours:
        truth = seg["track"][h // STEP_H]
        out[h] = (V.point_in_geometry(snaps[h].get("cone90"), *truth), V.point_in_geometry(snaps[h].get("cone50"), *truth))
    return out


# --- main ---------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-segments", type=int, default=60)
    ap.add_argument("--per-drifter", type=int, default=2)
    ap.add_argument("--gap-days", type=float, default=10.0)
    ap.add_argument("--horizon", type=int, default=72)
    ap.add_argument("--particles", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-grid", default="1,250,500,1000,2000,3000,5000,8000",
                    help="diffusivities (m^2/s) for the coverage calibration sweep")
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="parallel forecast processes")
    ap.add_argument("--refresh", action="store_true", help="re-download the drifter table")
    args = ap.parse_args(argv)
    horizons = tuple(h for h in HORIZONS if h <= args.horizon)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / V.SEGMENTS_DIR).mkdir(exist_ok=True)
    rows, gdp_meta = fetch_drifters(OUT / "raw" / "gdp_6hour_qc_nio.csv", args.refresh)
    log(f"GDP: {gdp_meta['rows']} fixes from {gdp_meta['drifters']} drifters, dataset ends {gdp_meta['time_coverage_end']}")
    segments = cut_segments(rows, horizon_h=args.horizon, per_drifter=args.per_drifter, gap_days=args.gap_days,
                            max_segments=args.max_segments)
    log(f"segments: {len(segments)} from {len({s['drifter_id'] for s in segments})} drifters")
    if not segments:
        log("no usable segments; nothing written")
        return 1

    hycom = Hycom()
    k_default = float(cfg.DRIFT_K_SURFACE_M2S)
    failures = []
    fetched = []
    # Phase 1: currents, one segment at a time (polite to the HYCOM server; cached).
    for idx, seg in enumerate(segments):
        lats = [p[0] for p in seg["track"]]
        lons = [p[1] for p in seg["track"]]
        pad = 2.5
        box = dict(south=min(lats) - pad, north=max(lats) + pad, west=min(lons) - pad, east=max(lons) + pad)
        t0 = seg["start"] - timedelta(hours=3)
        t1 = seg["start"] + timedelta(hours=args.horizon + 3)
        cache = OUT / "currents" / f"hycom_{seg['drifter_id']}_{seg['start']:%Y%m%dT%H}.nc"
        try:
            if not cache.exists():
                log(f"[{idx + 1}/{len(segments)}] HYCOM for drifter {seg['drifter_id']} {seg['start']:%Y-%m-%d %H}Z")
                hycom.fetch(cache, t0=t0, t1=t1, **box)
            fetched.append((seg, cache))
        except Exception as exc:
            failures.append({"drifter_id": seg["drifter_id"], "start": seg["start"].isoformat(), "stage": "currents",
                             "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            log(f"  currents failed: {failures[-1]['error']}")

    # Phase 2: forecasts at the engine's default K, in parallel.
    results = []
    jobs = [(seg, str(cache), args.horizon, args.particles, k_default, STEP_H, args.seed, horizons)
            for seg, cache in fetched]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for (seg, cache), outcome in zip(fetched, pool.map(_forecast_job, jobs)):
            if "error" in outcome:
                failures.append({"drifter_id": seg["drifter_id"], "start": seg["start"].isoformat(),
                                 "stage": "forecast", "error": outcome["error"]})
                log(f"  forecast failed: {outcome['error']}")
                continue
            results.append((seg, outcome["fc"], outcome["scored"], cache))
            hmax = outcome["scored"]["horizons"].get(str(max(horizons)), {})
            log(f"  {seg['drifter_id']} {seg['start']:%Y-%m-%d}: drogue={seg['drogue']} "
                f"sep{max(horizons)}={hmax.get('sep_model_km')} km persist={hmax.get('sep_persistence_km')} "
                f"in90={hmax.get('in90')}")

    # -- per-segment files -------------------------------------------------------------------
    seg_rows = []
    for n, (seg, fc, scored, cache) in enumerate(results):
        name = f"seg_{n:03d}.json"
        snaps = []
        for s in fc["snapshots"]:
            h = int(round(s["t_hours"]))
            if h % STEP_H:
                continue
            pts = s["points"]
            step = max(1, len(pts) // 80)
            snaps.append({"t_hours": h, "cone50": _round_geom(s.get("cone50")), "cone90": _round_geom(s.get("cone90")),
                          "particles": [[round(p[0], 4), round(p[1], 4)] for p in pts[::step]]})
        doc = {
            "format": V.FORMAT + "+segment", "index": n,
            "drifter_id": seg["drifter_id"], "buoy_type": seg["typebuoy"], "drogue": seg["drogue"],
            "start_time": seg["start"].isoformat().replace("+00:00", "Z"),
            "hours": [STEP_H * i for i in range(len(seg["track"]))],
            "observed": [[round(p[0], 4), round(p[1], 4)] for p in seg["track"]],
            "forecast_mean": [None if p is None else [round(p[0], 4), round(p[1], 4)] for p in scored["mean_track"]],
            "baseline_constant_model_current": None if scored["const_model"] is None else
            [[round(p[0], 4), round(p[1], 4)] for p in scored["const_model"]],
            "baseline_constant_observed_velocity": [[round(p[0], 4), round(p[1], 4)] for p in scored["const_obs"]],
            "snapshots": snaps,
            "horizons": scored["horizons"],
            "k_m2s": k_default, "n_particles": args.particles,
            "currents_file": cache.name,
            "observed_source": f"NOAA GDP {GDP_DATASET} (OSMC ERDDAP), drifter {seg['drifter_id']}",
            "simulated": False,
            "note": ("observed = real GDP drifter positions (6-hourly, QC-interpolated); forecast = GhostTrace floating "
                     "drift over HYCOM ESPC-D-V02 archive (analysis) surface currents; hindcast, no windage"),
        }
        (OUT / V.SEGMENTS_DIR / name).write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
        seg_rows.append({"index": n, "file": name, "drifter_id": seg["drifter_id"], "buoy_type": seg["typebuoy"],
                         "drogue": seg["drogue"], "start_time": doc["start_time"],
                         "start_lat": round(seg["track"][0][0], 4), "start_lon": round(seg["track"][0][1], 4),
                         "horizons": scored["horizons"]})
    for stale in (OUT / V.SEGMENTS_DIR).glob("seg_*.json"):
        if stale.name not in {r["file"] for r in seg_rows}:
            stale.unlink()

    groups = {"all": seg_rows, "drogued": [r for r in seg_rows if r["drogue"] == "drogued"],
              "undrogued": [r for r in seg_rows if r["drogue"] == "undrogued"],
              "uncertain": [r for r in seg_rows if r["drogue"] == "uncertain"]}
    metrics = {g: [V.aggregate(rs, h) for h in horizons] for g, rs in groups.items() if rs}

    # -- calibration sweep -----------------------------------------------------------------------
    calibration = None
    if not args.no_calibration and results:
        grid = [float(x) for x in args.k_grid.split(",") if x.strip()]
        cal_h = tuple(h for h in (24, 48, 72) if h <= args.horizon)
        sweep = []
        for k in grid:
            flags = {h: [] for h in cal_h}
            flags50 = {h: [] for h in cal_h}
            cjobs = [(seg, str(cache), args.horizon, args.particles, k, args.seed, cal_h)
                     for seg, _fc, _scored, cache in results]
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for outcome in pool.map(_coverage_job, cjobs):
                    for h in cal_h:
                        flags[h].append(outcome[h][0])
                        flags50[h].append(outcome[h][1])
            entry = {"k_m2s": k}
            for h in cal_h:
                ci = V.wilson_interval(sum(flags[h]), len(flags[h]))
                entry[str(h)] = {"coverage_90": round(sum(flags[h]) / len(flags[h]), 3),
                                 "coverage_90_ci95": [round(ci[0], 3), round(ci[1], 3)],
                                 "coverage_50": round(sum(flags50[h]) / len(flags50[h]), 3), "n": len(flags[h])}
            sweep.append(entry)
            log(f"calibration K={k:g}: " + ", ".join(f"{h} h cov90={entry[str(h)]['coverage_90']}" for h in cal_h))
        per_h = {}
        for h in cal_h:
            ok = [e["k_m2s"] for e in sweep if e[str(h)]["coverage_90"] >= 0.9]
            per_h[str(h)] = min(ok) if ok else None
        needed = [v for v in per_h.values()]
        recommended = max(needed) if needed and all(v is not None for v in needed) else None
        calibration = {
            "method": ("every segment re-run with the same seed for each K on the grid; empirical 90% cone coverage at "
                       f"{', '.join(str(h) for h in cal_h)} h; smallest grid K reaching >= 0.90 per horizon; the "
                       "recommendation is the largest of those (so all listed horizons reach nominal coverage)"),
            "grid_m2s": grid, "sweep": sweep, "smallest_k_reaching_90pct_by_horizon": per_h,
            "recommended_k_surface_m2s": recommended,
            "current_default_k_surface_m2s": k_default,
            "applied": False,
            "note": ("NOT applied: config_geo.DRIFT_K_SURFACE_M2S is unchanged. A random walk grows as sqrt(t) while "
                     "model-error separation tends to grow faster, so one K cannot be exact at every horizon; "
                     "set GHOSTTRACE_DRIFT_K_SURFACE_M2S to try it."
                     if recommended is not None else
                     "no K on the grid reached nominal 90% coverage at every horizon; widen the grid"),
        }

    # -- verdict -----------------------------------------------------------------------------------
    headline = {}
    allm = {m["horizon_hours"]: m for m in metrics.get("all", [])}
    if 24 in allm:
        a24 = allm[24]
        headline = {
            "n_segments": len(seg_rows), "n_drifters": len({r["drifter_id"] for r in seg_rows}),
            "median_separation_24h_km": a24["separation_km_model"]["median"],
            "coverage90_24h": a24["coverage_90"]["fraction"],
            "coverage50_24h": a24["coverage_50"]["fraction"],
            "beats_persistence_24h_pct": a24["model_vs_persistence"]["median_separation_reduction_pct"],
        }
        if 72 in allm:
            a72 = allm[72]
            headline.update({
                "median_separation_72h_km": a72["separation_km_model"]["median"],
                "coverage90_72h": a72["coverage_90"]["fraction"],
                "beats_persistence_72h_pct": a72["model_vs_persistence"]["median_separation_reduction_pct"],
                "skill_72h_median": a72["liu_weisberg_skill_model"]["median"],
            })
    cov = [m["coverage_90"]["fraction"] for m in metrics.get("all", []) if m["horizon_hours"] >= 24
           and m["coverage_90"]["fraction"] is not None]
    if cov and max(cov) < 0.75:
        calib_verdict = ("OVERCONFIDENT: the 90% cone contained the real drifter far less than 90% of the time "
                         f"(between {min(cov):.0%} and {max(cov):.0%} at 24-72 h). The default diffusivity "
                         f"K = {k_default:g} m^2/s makes cones too narrow for real ocean drift; recalibrate K "
                         "(see calibration) before reading a cone as a 90% region.")
    elif cov and min(cov) < 0.85:
        calib_verdict = (f"somewhat overconfident: 90% cone coverage {min(cov):.0%}-{max(cov):.0%} at 24-72 h")
    elif cov:
        calib_verdict = f"approximately calibrated: 90% cone coverage {min(cov):.0%}-{max(cov):.0%} at 24-72 h"
    else:
        calib_verdict = "not assessed"

    summary = {
        "format": V.FORMAT,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "simulated": False,
        "title": "GhostTrace floating drift vs real NOAA Global Drifter Program drifters",
        "headline": headline,
        "calibration_verdict": calib_verdict,
        "region": BOX_NAME, "box": BOX,
        "config": {"mode": "floating", "horizon_hours": args.horizon, "horizons": list(horizons),
                   "n_particles": args.particles, "dt_minutes": 30, "k_surface_m2s": k_default, "seed": args.seed,
                   "integrator": cfg.DRIFT_INTEGRATOR, "per_drifter": args.per_drifter, "gap_days": args.gap_days,
                   "max_segments": args.max_segments, "windage": "none (no wind data in GhostTrace)"},
        "counts": {"segments": len(seg_rows), "drifters": len({r["drifter_id"] for r in seg_rows}),
                   "by_drogue": {g: len(rs) for g, rs in groups.items() if g != "all"},
                   "failed_segments": len(failures)},
        "metrics": metrics,
        "calibration": calibration,
        "segments": seg_rows,
        "failures": failures,
        "datasets": [
            {"role": "truth", "name": "NOAA Global Drifter Program, 6-hour interpolated QC drifter data",
             "dataset_id": GDP_DATASET, "server": ERDDAP, "url": gdp_meta["url"], "licence": GDP_LICENCE,
             "citation": GDP_CITATION, "accessed": gdp_meta["accessed"],
             "time_coverage_end": gdp_meta["time_coverage_end"], "rows_in_box": gdp_meta["rows"],
             "drifters_in_box": gdp_meta["drifters"]},
            {"role": "currents", "name": "HYCOM ESPC-D-V02 Global 1/12 deg analysis archive (NRL / FNMOC), surface u/v",
             "dataset_id": "ESPC-D-V02 u3z / v3z (yearly aggregations)", "url": HYCOM_OPENDAP.format(var="{u,v}", year="{year}"),
             "licence": HYCOM_LICENCE,
             "citation": "HYCOM consortium / US Naval Research Laboratory; ESPC-D V02, https://www.hycom.org/dataserver/espc-d-v02",
             "accessed": datetime.now(timezone.utc).date().isoformat(),
             "note": "per-segment subsets cached in data/ghosttrace/validation/currents/ (git-ignored)"},
        ],
        "method": [
            f"Segments: {args.horizon} h windows of unbroken 6-hourly GDP positions, <= {args.per_drifter} per drifter, "
            f">= {args.gap_days:g} days apart, round-robin across drifters, capped at {args.max_segments}.",
            "Forecast: ghosttrace.drift.drift_forecast(mode='floating') from the drifter's position and time, "
            f"{args.particles} particles, RK4, dt 30 min, K = {k_default:g} m^2/s (the engine default), "
            "no land layer (open ocean), seeded.",
            "Separation: great-circle distance between the ensemble mean position and the observed position.",
            "Coverage: fraction of segments whose observed position lies inside the 50% / 90% highest-density cone.",
            "Skill: " + V.LIU_WEISBERG_CITATION + " Tolerance n = 1.",
            "Baselines: persistence (no motion); HYCOM surface current at the start position and time held constant; "
            "the drifter's own GDP velocity (ve, vn) at the start held constant.",
            "Drogue status per segment from GDP drogue_lost_date: drogued (still attached at segment end), "
            "undrogued (lost before the start), uncertain (lost during the segment, or uncertain from deployment).",
        ],
        "caveats": [
            "Hindcast: HYCOM archive fields are the model's analysis for those dates, so forecast skill in real time "
            "(with forecast currents) will be lower than shown.",
            "No wind: GhostTrace has no windage term. Undrogued drifters and floating nets feel wind drag, so they are "
            "expected to be forecast worse; they are reported separately.",
            "1/12 deg (~9 km) currents do not resolve small eddies, fronts or nearshore flow; these segments are "
            "mostly offshore, so near-coast skill is not measured here.",
            "Drogued SVP drifters follow the current at ~15 m depth; they are a proxy for floating or midwater gear, "
            "not for a net lying on the seabed. The seabed mode is NOT validated by this test.",
            "Segments from the same drifter or the same weeks are not independent; the Wilson intervals assume "
            "independence and so understate the true uncertainty.",
            f"Region and season: {BOX_NAME}, the dates listed per segment; performance elsewhere is not measured.",
        ],
    }
    (OUT / V.SUMMARY_FILE).write_text(json.dumps(summary, indent=1, allow_nan=False), encoding="utf-8")
    (OUT / ".gitignore").write_text("# Written by tools/validate_drift_gdp.py: regenerable caches, not bundled.\n"
                                    "currents/\nraw/\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in [OUT / V.SUMMARY_FILE, *(OUT / V.SEGMENTS_DIR).glob("*.json")])
    log(f"wrote {OUT / V.SUMMARY_FILE} and {len(seg_rows)} segment files ({size / 1e6:.2f} MB bundled); "
        f"{len(failures)} failures")
    log(f"headline: {json.dumps(headline)}")
    log(f"verdict: {calib_verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
