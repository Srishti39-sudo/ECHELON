"""Ocean currents from the bundled model snapshot, sampled in space and time.

The drift forecast is only as honest as the currents under it, so this module
has one job and a strict boundary around it: read the real model output that
tools/fetch_ghosttrace_data.py bundled, and interpolate it without inventing
anything the model did not say.

WHAT IS BUNDLED
    data/ghosttrace/currents/hycom_espc_<region>.nc, one file per region: HYCOM
    ESPC-D-V02 (US Naval Research Laboratory / FNMOC, global 1/12 degree,
    3-hourly), subset from HYCOM.org's THREDDS NetCDF Subset Service. Two
    levels are kept:

        surface   the model's z = 0 m level
        bottom    in each grid cell, the deepest z-level with valid data, i.e.
                  a NEAR-bottom current up to one level spacing above the model
                  bed. In 12 m of water that is the 10 or 12 m level; in 900 m
                  of water it is the 800 m level. It is not a boundary-layer
                  velocity and it does not resolve the last metre where a net
                  actually lies.

    Each time step carries the model run that produced it (run_time), so a
    step can be told apart as analysis or forecast.

INTERPOLATION
    Space: bilinear on the regular lat/lon grid. Cells the model treats as land
    are NaN. Next to the land mask a sample uses the valid corners only, with
    the bilinear weights renormalised, when at least
    config_geo.CURRENT_MIN_VALID_CORNERS of the four are valid; otherwise NaN.
    Without that rule every coast would be a ~9 km wide dead band where no
    current exists, and particles would stop offshore instead of reaching the
    beach. With it, the velocity next to the coast is the nearest ocean cell's
    velocity, which is an extrapolation and is stated as such in drift output.

    Time: linear between the two bracketing steps. A time outside the field's
    range returns NaN; this module never clamps silently. drift.py decides what
    to do about a start time outside the window and says so in its output.

    Outside the grid: NaN. A caller can ask contains() first.

SYNTHETIC FIELDS
    AnalyticField exists ONLY for unit tests: a uniform or callable velocity
    over a box and a time range. It is labelled synthetic in describe(), in
    its source string, and in every drift forecast that uses it. Nothing in
    this package loads one by default, and load_default_field() never falls
    back to one: with no bundled file it returns None and the drift stage
    reports that no current data is available.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ghosttrace import config_geo as cfg

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def to_epoch_seconds(t: Any) -> float:
    """datetime (naive = UTC), ISO-8601 string, or epoch seconds -> epoch seconds."""
    if isinstance(t, (int, float, np.floating, np.integer)):
        return float(t)
    if isinstance(t, str):
        text = t.strip().replace("Z", "+00:00")
        t = datetime.fromisoformat(text)
    if isinstance(t, datetime):
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t.timestamp()
    if isinstance(t, np.datetime64):
        return float(t.astype("datetime64[ms]").astype("int64")) / 1000.0
    raise TypeError(f"cannot interpret {t!r} as a time")


def iso(seconds: float) -> str:
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc).isoformat().replace("+00:00", "Z")


class CurrentField:
    """A gridded current snapshot for one region: surface and near-bottom u/v."""

    synthetic = False

    def __init__(self, *, lat: np.ndarray, lon: np.ndarray, times_s: np.ndarray,
                 u: dict[str, np.ndarray], v: dict[str, np.ndarray], run_times_s: np.ndarray | None = None,
                 bottom_level_depth: np.ndarray | None = None, attrs: dict[str, Any] | None = None,
                 path: Path | None = None):
        self.lat = np.asarray(lat, dtype=float)
        self.lon = np.asarray(lon, dtype=float)
        self.times_s = np.asarray(times_s, dtype=float)
        if np.any(np.diff(self.lat) <= 0) or np.any(np.diff(self.lon) <= 0) or np.any(np.diff(self.times_s) <= 0):
            raise ValueError("lat, lon and time axes must be strictly increasing")
        self.u = {k: np.asarray(a, dtype=np.float32) for k, a in u.items()}
        self.v = {k: np.asarray(a, dtype=np.float32) for k, a in v.items()}
        self.run_times_s = None if run_times_s is None else np.asarray(run_times_s, dtype=float)
        self.bottom_level_depth = bottom_level_depth
        self.attrs = attrs or {}
        self.path = path

    # -- construction -----------------------------------------------------------------

    @classmethod
    def from_netcdf(cls, path: Path) -> "CurrentField":
        import netCDF4

        with netCDF4.Dataset(path) as ds:
            ds.set_auto_maskandscale(True)
            t = ds.variables["time"]
            scale = {"hours": 3600.0, "seconds": 1.0, "days": 86400.0}[t.units.split()[0]]
            base = netCDF4.num2date(0, t.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
            base_s = base.replace(tzinfo=timezone.utc).timestamp()
            times_s = base_s + np.asarray(t[:], dtype=float) * scale
            runs = None
            if "run_time" in ds.variables:
                runs = base_s + np.asarray(ds.variables["run_time"][:], dtype=float) * scale
            u, v = {}, {}
            for level in cfg.CURRENT_LEVELS:
                u[level] = np.ma.filled(ds.variables[f"u_{level}"][:].astype("float32"), np.nan)
                v[level] = np.ma.filled(ds.variables[f"v_{level}"][:].astype("float32"), np.nan)
            bld = None
            if "bottom_level_depth" in ds.variables:
                bld = np.ma.filled(ds.variables["bottom_level_depth"][:].astype("float32"), np.nan)
            attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
            lat = np.asarray(ds.variables["lat"][:], dtype=float)
            lon = np.asarray(ds.variables["lon"][:], dtype=float)
        return cls(lat=lat, lon=lon, times_s=times_s, u=u, v=v, run_times_s=runs,
                   bottom_level_depth=bld, attrs=attrs, path=Path(path))

    # -- description ------------------------------------------------------------------

    @property
    def region(self) -> str | None:
        return self.attrs.get("region")

    @property
    def time_range(self) -> tuple[datetime, datetime]:
        return (datetime.fromtimestamp(self.times_s[0], tz=timezone.utc),
                datetime.fromtimestamp(self.times_s[-1], tz=timezone.utc))

    @property
    def source(self) -> str:
        return str(self.attrs.get("title") or "bundled current field")

    @property
    def model_run(self) -> str | None:
        """The latest model run in the snapshot (the one its forecast steps come from)."""
        if self.run_times_s is None:
            return None
        return iso(float(np.max(self.run_times_s)))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return float(self.lat[0]), float(self.lon[0]), float(self.lat[-1]), float(self.lon[-1])

    def describe(self) -> dict[str, Any]:
        t0, t1 = self.time_range
        runs = sorted(set(self.attrs.get("model_runs", "").split(";")) - {""})
        forecast_from = None
        if self.run_times_s is not None:
            fc = self.times_s - self.run_times_s >= 24 * 3600
            if fc.any():
                forecast_from = iso(float(self.times_s[np.argmax(fc)]))
        return {
            "name": self.source,
            "synthetic": False,
            "region": self.region,
            "model": self.attrs.get("generating_model"),
            "institution": self.attrs.get("institution"),
            "url": self.attrs.get("source_url"),
            "accessed": self.attrs.get("accessed"),
            "time_start": t0.isoformat().replace("+00:00", "Z"),
            "time_end": t1.isoformat().replace("+00:00", "Z"),
            "time_step_hours": float(np.median(np.diff(self.times_s)) / 3600.0) if len(self.times_s) > 1 else None,
            "model_runs": runs,
            "latest_model_run": self.model_run,
            "forecast_steps_from": forecast_from,
            "grid_spacing_deg": {"lat": float(np.median(np.diff(self.lat))), "lon": float(np.median(np.diff(self.lon)))},
            "levels": {"surface": "z = 0 m",
                       "bottom": "deepest valid model z-level in each cell (near-bottom, not boundary layer)"},
            "file": self.path.name if self.path else None,
        }

    def contains(self, lat: Any, lon: Any) -> np.ndarray:
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        return (lat >= self.lat[0]) & (lat <= self.lat[-1]) & (lon >= self.lon[0]) & (lon <= self.lon[-1])

    def in_time(self, t: Any) -> bool:
        s = to_epoch_seconds(t)
        return bool(self.times_s[0] <= s <= self.times_s[-1])

    def bottom_depth_at(self, lat: float, lon: float) -> float | None:
        """Depth of the model level used as 'bottom' in the cell nearest a point."""
        if self.bottom_level_depth is None or not bool(self.contains(lat, lon)):
            return None
        i = int(np.argmin(np.abs(self.lat - lat)))
        j = int(np.argmin(np.abs(self.lon - lon)))
        val = float(self.bottom_level_depth[i, j])
        return val if np.isfinite(val) else None

    # -- sampling ---------------------------------------------------------------------

    def _spatial(self, grid: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Bilinear sample of a (Y, X) grid with valid-corner renormalisation."""
        ny, nx = len(self.lat), len(self.lon)
        fy = np.interp(lat, self.lat, np.arange(ny))
        fx = np.interp(lon, self.lon, np.arange(nx))
        i0 = np.clip(np.floor(fy).astype(int), 0, ny - 2)
        j0 = np.clip(np.floor(fx).astype(int), 0, nx - 2)
        wy = fy - i0
        wx = fx - j0
        corners = (grid[i0, j0], grid[i0, j0 + 1], grid[i0 + 1, j0], grid[i0 + 1, j0 + 1])
        weights = ((1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx)
        num = np.zeros(lat.shape, dtype=float)
        den = np.zeros(lat.shape, dtype=float)
        nvalid = np.zeros(lat.shape, dtype=int)
        for c, w in zip(corners, weights):
            ok = np.isfinite(c)
            num += np.where(ok, c, 0.0) * w
            den += np.where(ok, w, 0.0)
            nvalid += ok
        out = np.where((nvalid >= cfg.CURRENT_MIN_VALID_CORNERS) & (den > 1e-9), num / np.maximum(den, 1e-12), np.nan)
        out = np.where(self.contains(lat, lon), out, np.nan)
        return out

    def sample(self, lat: Any, lon: Any, t: Any, level: str = "surface") -> tuple[np.ndarray, np.ndarray]:
        """(u, v) in m/s, eastward and northward. NaN on land, outside the grid or the time range.

        lat and lon may be scalars or arrays of the same shape; t is one time for all points.
        """
        if level not in self.u:
            raise ValueError(f"level must be one of {tuple(self.u)}, not {level!r}")
        scalar = np.ndim(lat) == 0
        lat = np.atleast_1d(np.asarray(lat, dtype=float))
        lon = np.atleast_1d(np.asarray(lon, dtype=float))
        s = to_epoch_seconds(t)
        if not (self.times_s[0] <= s <= self.times_s[-1]):
            nan = np.full(lat.shape, np.nan)
            return (float(nan[0]), float(nan[0])) if scalar else (nan, nan.copy())
        k = int(np.clip(np.searchsorted(self.times_s, s, side="right") - 1, 0, len(self.times_s) - 2)) \
            if len(self.times_s) > 1 else 0
        if len(self.times_s) == 1:
            u = self._spatial(self.u[level][0], lat, lon)
            v = self._spatial(self.v[level][0], lat, lon)
        else:
            a = (s - self.times_s[k]) / (self.times_s[k + 1] - self.times_s[k])
            u0 = self._spatial(self.u[level][k], lat, lon)
            v0 = self._spatial(self.v[level][k], lat, lon)
            if a <= 0:
                u, v = u0, v0
            else:
                u1 = self._spatial(self.u[level][k + 1], lat, lon)
                v1 = self._spatial(self.v[level][k + 1], lat, lon)
                u = (1 - a) * u0 + a * u1
                v = (1 - a) * v0 + a * v1
        if scalar:
            return float(u[0]), float(v[0])
        return u, v

    def speed_at(self, lat: Any, lon: Any, t: Any, level: str = "surface") -> Any:
        u, v = self.sample(lat, lon, t, level)
        return np.hypot(u, v) if not np.isscalar(u) else float(np.hypot(u, v))

    def max_speed(self, lat: float, lon: float, t_start: Any, hours: float, level: str = "bottom") -> dict[str, Any]:
        """Maximum speed at a point over [t_start, t_start + hours], at the field's own time steps."""
        s0 = to_epoch_seconds(t_start)
        s1 = s0 + hours * 3600.0
        ts = [s0] + [float(x) for x in self.times_s if s0 < x < s1] + [s1]
        speeds = [(self.speed_at(lat, lon, t, level), t) for t in ts]
        finite = [(sp, t) for sp, t in speeds if np.isfinite(sp)]
        if not finite:
            return {"max_mps": None, "at": None, "samples": len(ts), "valid_samples": 0}
        sp, t = max(finite)
        return {"max_mps": round(float(sp), 3), "at": iso(t), "samples": len(ts), "valid_samples": len(finite)}


class FieldCollection:
    """Several regional fields behind one interface. A point is served by the field containing it."""

    synthetic = False

    def __init__(self, fields: Iterable[CurrentField]):
        self.fields = list(fields)
        if not self.fields:
            raise ValueError("FieldCollection needs at least one field")

    def field_for(self, lat: float, lon: float) -> CurrentField | None:
        for f in self.fields:
            if bool(f.contains(lat, lon)):
                return f
        return None

    def describe(self) -> dict[str, Any]:
        return {"name": "bundled current fields", "synthetic": any(f.synthetic for f in self.fields),
                "fields": [f.describe() for f in self.fields]}

    @property
    def source(self) -> str:
        return "; ".join(sorted({f.source for f in self.fields}))

    def sample(self, lat: Any, lon: Any, t: Any, level: str = "surface"):
        f = self.field_for(float(np.ravel(lat)[0]), float(np.ravel(lon)[0]))
        if f is None:
            if np.ndim(lat) == 0:
                return float("nan"), float("nan")
            nan = np.full(np.shape(lat), np.nan)
            return nan, nan.copy()
        return f.sample(lat, lon, t, level)

    def speed_at(self, lat: Any, lon: Any, t: Any, level: str = "surface"):
        u, v = self.sample(lat, lon, t, level)
        return np.hypot(u, v)


class AnalyticField(CurrentField):
    """SYNTHETIC analytic current field, for unit tests only.

    velocity(lat, lon, t_hours, level) -> (u, v) in m/s, or constants. Never
    loaded by default, and labelled synthetic everywhere it is described.
    """

    synthetic = True

    def __init__(self, *, bounds: tuple[float, float, float, float], t_start: Any, hours: float,
                 velocity: Callable[..., tuple[Any, Any]] | tuple[float, float] | None = None,
                 bottom_velocity: Callable[..., tuple[Any, Any]] | tuple[float, float] | None = None,
                 label: str = "SYNTHETIC analytic test field"):
        s, w, n, e = bounds
        self._bounds = bounds
        self._t0 = to_epoch_seconds(t_start)
        self._hours = float(hours)
        self._vel = velocity if velocity is not None else (0.0, 0.0)
        self._bvel = bottom_velocity if bottom_velocity is not None else self._vel
        self.label = label
        self.lat = np.array([s, n], dtype=float)
        self.lon = np.array([w, e], dtype=float)
        self.times_s = np.array([self._t0, self._t0 + self._hours * 3600.0])
        self.run_times_s = None
        self.bottom_level_depth = None
        self.attrs = {"title": label, "region": "synthetic", "synthetic": "true"}
        self.path = None
        self.u, self.v = {"surface": None, "bottom": None}, {"surface": None, "bottom": None}

    @property
    def source(self) -> str:
        return self.label

    def describe(self) -> dict[str, Any]:
        t0, t1 = self.time_range
        return {"name": self.label, "synthetic": True, "region": "synthetic",
                "time_start": t0.isoformat(), "time_end": t1.isoformat(),
                "note": "analytic field for unit tests; not ocean data"}

    def sample(self, lat: Any, lon: Any, t: Any, level: str = "surface"):
        scalar = np.ndim(lat) == 0
        lat = np.atleast_1d(np.asarray(lat, dtype=float))
        lon = np.atleast_1d(np.asarray(lon, dtype=float))
        s = to_epoch_seconds(t)
        vel = self._bvel if level == "bottom" else self._vel
        if not (self.times_s[0] <= s <= self.times_s[-1]):
            u = np.full(lat.shape, np.nan)
            v = u.copy()
        else:
            if callable(vel):
                u, v = vel(lat, lon, (s - self._t0) / 3600.0)
            else:
                u, v = vel
            u = np.broadcast_to(np.asarray(u, dtype=float), lat.shape).copy()
            v = np.broadcast_to(np.asarray(v, dtype=float), lat.shape).copy()
            inside = self.contains(lat, lon)
            u[~inside] = np.nan
            v[~inside] = np.nan
        if scalar:
            return float(u[0]), float(v[0])
        return u, v


_DEFAULT: FieldCollection | None = None
_LOADED = False
_LOCK = threading.Lock()


def load_default_field() -> FieldCollection | None:
    """Every bundled regional current file, or None when none has been fetched.

    Never returns a synthetic field.
    """
    global _DEFAULT, _LOADED
    with _LOCK:
        if not _LOADED:
            files = sorted(Path(cfg.CURRENTS_DIR).glob("*.nc")) if Path(cfg.CURRENTS_DIR).exists() else []
            fields = [CurrentField.from_netcdf(p) for p in files]
            _DEFAULT = FieldCollection(fields) if fields else None
            _LOADED = True
        return _DEFAULT
