"""Seabed coverage, blind spots, re-look lines and a mission replay.

    from survey_hazard_map.hazard_coverage import coverage_for_survey, replay_for_survey
    coverage = coverage_for_survey("data/surveys/<id>")     # cached as coverage.json
    replay   = replay_for_survey("data/surveys/<id>")

A survey that carries per-ping navigation (a deepecho-strip-nav/1 sidecar per
strip, recorded in manifest.json under survey.navigation.sidecars) knows, for
every image row, where the towfish was, which way it was heading, how high it
flew and whether the row is trustworthy. That is enough to say which patch of
seabed the sonar actually imaged, and which it did not. Everything below is
computed from those sidecar rows and nothing else. A survey without them (a
plain image with no navigation) gets {"available": false, "reason": ...}: there
is no footprint to compute and none is guessed.

SWATH, PER ROW
    Ground-range strips (sonar_ingest, mode "ping"). The ingest resampled every
    ping onto ground range with ground = sqrt(slant^2 - altitude^2) and wrote
    the widest ground range it produced, processing.slant_range_correction.
    ground_range_max_m, reached at the lowest altitude of the line. The far
    slant range is therefore F = sqrt(ground_range_max_m^2 + min(altitude)^2)
    and each row's half-width is sqrt(F^2 - altitude^2), capped at the image's
    own extent either side of nadir_col. On the Gulf of Mannar demo this
    matches the last non-zero pixel of every row checked to 0.1 m.

    Slant-range strips (import_geotag.py, mode "geotag_slant_range"). Each row
    records slant_range_m (the maximum slant range per side) and altitude_m,
    so the half-width is sqrt(slant_range_m^2 - altitude_m^2). A row with no
    altitude uses slant_range_m as an upper bound and the basis says so.

NADIR BLIND STRIP, PER ROW
    Directly under the towfish the first seabed return arrives at slant range
    = altitude. Every ground range between 0 and g0 = sqrt((altitude + ds)^2 -
    altitude^2), where ds is one native slant-range sample, is served by that
    ONE sample interval: a ground-range image shows pixels there, but they are
    interpolated, not independently resolved seabed. That band is reported as
    the nadir blind strip and excluded from the imaged area. ds is
    processing.slant_range_correction.native_slant_m_per_sample for an ingested
    strip and slant_range_m / samples_per_side for a geotag strip. This is a
    LOWER BOUND on the practical nadir gap: the transducer's vertical beam
    pattern, which usually widens it, is not recorded in the data.

DEGRADED ROWS
    Rows whose quality is not "ok" (dropout, attitude, interpolated, nav_jump)
    or that fall in the sidecar's degraded_rows ranges are excluded from the
    imaged area and reported as gaps, by reason, unless another row imaged the
    same seabed.

GEOMETRY
    Positions are projected into a local azimuthal equidistant projection
    (pyproj, WGS84) centred on the survey. Rows are sampled every
    SAMPLE_STEP_M along track; consecutive samples form a quad (port and
    starboard edges, perpendicular to heading), the quad's convex hull is its
    footprint, and shapely unions them. Consecutive samples further apart than
    the expected step by a wide margin are not joined, so a navigation jump
    never paints seabed that was not passed over.

RE-LOOK LINES (planning geometry, a HEURISTIC)
    (a) A gap of at least GAP_MIN_AREA_M2 gets short lines on the original
        heading, each running RUN_IN_M past the gap at both ends. With W the
        gap's across-track width, hw the half-width and g0 the nadir half-width
        at the nearest track row, and r_mid = (g0 + hw) / 2:
          W <= hw - g0   one line offset so the gap sits at mid-range (r_mid)
                         on its starboard side, never under the new nadir;
          W <= 2 hw + r_mid (and 3 g0 <= hw)
                         two lines: one that images the gap but its own nadir
                         band, and one r_mid further to starboard that images
                         that band at mid-range and reaches r_mid further;
          wider          ceil(W / (hw - g0)) lines, each imaging one band of
                         the gap at mid-range on its starboard side.
    (b) A contact that is low-confidence (confidence_pct < RELOOK_CONFIDENCE_PCT),
        of an unidentified class, filtered by verification while its severity
        tier is medium or critical, or whose engine action asks for expert
        identification gets a line PERPENDICULAR to the original track, so its
        shadow is seen from a second aspect. The line is offset so the contact
        is at mid-range rather than directly beneath it, because a perpendicular
        pass straight over the contact would put it in the nadir blind strip.

    Why perpendicular: an acoustic shadow is cast away from the sonar, so a
    contact imaged from one aspect shows one shadow outline; a second pass at a
    different aspect shows another, which is how an elongated object is told
    from a round one. NOAA's Hydrographic Surveys Specifications and
    Deliverables (version 2023.2.02) sizes a "holiday" by the inability to
    detect features of the claimed size and allows none spanning potentially
    significant features (section 8.3), and it requires 200% side scan sonar
    coverage before a charted feature may be disproved (section 7.5), delivered
    as a separate mosaic per 100% (section 8.5). It does NOT prescribe
    perpendicular re-look lines; that choice, the thresholds and
    the run-in length are this project's heuristic, not a navigation procedure,
    and every line must be checked against water depth, traffic and the
    vessel's turning circle before it is run.

WHAT THIS DOES NOT CORRECT
    Everything hazard_geo lists for ping navigation: layback, pitch and yaw of
    the footprint, refraction, seabed slope across the swath. Coverage is a
    footprint on a flat seabed, not a guarantee that an object of any size
    would have been detected in it.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("deepecho.coverage")

VERSION = "1.0.0"
CACHE_FILE = "coverage.json"
SIDECAR_FORMAT = "deepecho-strip-nav/1"
NAV_MODES = ("ping", "geotag_slant_range")

# Polygon sampling interval along track, metres. Two metres is far below the
# swath width and keeps a kilometre of line to a few hundred quads.
SAMPLE_STEP_M = float(os.environ.get("HAZARD_COVERAGE_STEP_M", "2.0"))
# Gaps smaller than this are reported but get no re-look line.
GAP_MIN_AREA_M2 = float(os.environ.get("HAZARD_COVERAGE_GAP_MIN_M2", "100"))
# Slivers left by polygon arithmetic below this are not reported as gaps at all.
GAP_REPORT_MIN_M2 = 1.0
RELOOK_CONFIDENCE_PCT = float(os.environ.get("HAZARD_RELOOK_CONFIDENCE_PCT", "60"))
RUN_IN_M = float(os.environ.get("HAZARD_RELOOK_RUN_IN_M", "50"))
REPLAY_MAX_POINTS = int(os.environ.get("HAZARD_REPLAY_MAX_POINTS", "1500"))
REPLAY_AREA_CHECKPOINTS = 60
TRACK_KEYS = ("t", "lat", "lon", "heading", "quality", "strip", "row", "half_width_port_m",
              "half_width_stbd_m", "nadir_half_width_m", "distance_km", "area_km2")
UNIDENTIFIED_CLASSES = {"unknown", "anomaly", "unknown-anomaly", "other", "unidentified",
                        "unclassified"}
RELOOK_TIERS_WHEN_FILTERED = {"critical", "medium"}
QUALITY_OK = "ok"

REFERENCES = [
    {"title": "NOAA Office of Coast Survey, Hydrographic Surveys Specifications and "
              "Deliverables, version 2023.2.02: section 8.3 (holidays), section 7.5 "
              "(200% side scan sonar coverage for feature disproval), section 8.5 (a "
              "separate mosaic per 100% of coverage)",
     "url": "https://nauticalcharts.noaa.gov/publications/docs/standards-and-requirements/"
            "specs/HSSD_2023-2-02.pdf",
     "used_for": "what a coverage gap (holiday) is and that full coverage twice over is "
                 "required to prove absence; not for the perpendicular re-look geometry, "
                 "which is this project's heuristic"},
]


class CoverageUnavailable(Exception):
    """The survey has no navigation the coverage can be computed from."""


# --- loading ------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_time(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.timestamp()


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass
class StripTrack:
    """One strip's per-row navigation, swath and nadir widths."""

    strip: str
    kind: str                       # "ground_range" or "slant_range"
    lat: Any
    lon: Any
    heading: Any
    altitude: Any
    time: Any                       # epoch seconds, NaN where unknown
    quality: list[str]
    port_w: Any
    stbd_w: Any
    nadir_w: Any
    m_per_px_along: float | None
    synthetic: bool
    basis: dict[str, str] = field(default_factory=dict)
    x: Any = None
    y: Any = None


def _strip_track(strip: str, sidecar: dict[str, Any]) -> StripTrack:
    import numpy as np

    rows = sidecar.get("rows") or []
    if not rows:
        raise CoverageUnavailable(f"strip {strip}: the navigation sidecar has no rows")
    n = len(rows)

    def column(key: str) -> Any:
        return np.array([np.nan if _num(r.get(key)) is None else float(r[key]) for r in rows])

    lat, lon, heading, altitude = (column("lat"), column("lon"), column("heading_deg"),
                                   column("altitude_m"))
    time = np.array([np.nan if _parse_time(r.get("time")) is None else _parse_time(r.get("time"))
                     for r in rows])
    quality = [str(r.get("quality") or QUALITY_OK) for r in rows]
    for entry in sidecar.get("degraded_rows") or []:
        try:
            start, end, reason = int(entry[0]), int(entry[1]), str(entry[2])
        except (TypeError, ValueError, IndexError):
            continue
        for i in range(max(0, start), min(n, end + 1)):
            if quality[i] == QUALITY_OK:
                quality[i] = reason

    basis: dict[str, str] = {}
    across = _num(sidecar.get("m_per_px_across"))
    slant_corrected = sidecar.get("slant_range_corrected")
    has_slant = any(_num(r.get("slant_range_m")) is not None for r in rows)

    if slant_corrected is False or (across is None and has_slant):
        if not has_slant:
            raise CoverageUnavailable(
                f"strip {strip}: slant-range strip without slant_range_m on its rows")
        kind = "slant_range"
        slant = column("slant_range_m")
        samples = column("samples_per_side")
        ds = np.where(np.isfinite(samples) & (samples > 0), slant / np.where(samples > 0, samples, 1),
                      _num(sidecar.get("slant_m_per_sample")) or np.nan)
        known = np.isfinite(altitude) & (altitude > 0)
        ground = np.where(known, np.sqrt(np.clip(slant ** 2 - np.where(known, altitude, 0) ** 2,
                                                 0, None)), slant)
        port_w = stbd_w = ground
        nadir_w = np.where(known & np.isfinite(ds),
                           np.sqrt(np.clip((np.where(known, altitude, 0) + ds) ** 2
                                           - np.where(known, altitude, 0) ** 2, 0, None)), np.nan)
        basis["half_width"] = ("slant-range strip: half-width = sqrt(slant_range_m^2 - "
                               "altitude_m^2) per row"
                               + ("" if known.all() else "; rows without altitude use "
                                  "slant_range_m as an upper bound"))
        basis["nadir"] = ("nadir blind strip g0 = sqrt((altitude + ds)^2 - altitude^2) per row, "
                          "ds = slant_range_m / samples_per_side (one slant sample); rows "
                          "without altitude have no nadir width")
    elif across is not None:
        kind = "ground_range"
        width = _num(sidecar.get("width"))
        nadir_col = _num(sidecar.get("nadir_col"))
        if width is None or nadir_col is None:
            raise CoverageUnavailable(f"strip {strip}: sidecar has no width or nadir_col")
        extent_port = nadir_col * across
        extent_stbd = (width - nadir_col) * across
        src = (sidecar.get("processing") or {}).get("slant_range_correction") or {}
        gmax = _num(src.get("ground_range_max_m"))
        ds = _num(src.get("native_slant_m_per_sample"))
        known = np.isfinite(altitude)
        if gmax is not None and known.any():
            far = math.hypot(gmax, float(np.nanmin(altitude)))
            ground = np.where(known, np.sqrt(np.clip(far ** 2 - np.where(known, altitude, 0) ** 2,
                                                     0, None)), gmax)
            port_w = np.minimum(ground, extent_port)
            stbd_w = np.minimum(ground, extent_stbd)
            basis["half_width"] = (
                f"ground-range strip: far slant range F = sqrt(ground_range_max_m^2 + "
                f"min altitude^2) = {far:.2f} m; half-width = sqrt(F^2 - altitude^2) per row, "
                f"capped at the image extent ({extent_port:.1f} m port, {extent_stbd:.1f} m "
                f"starboard); rows without altitude use ground_range_max_m")
        else:
            port_w = np.full(n, extent_port)
            stbd_w = np.full(n, extent_stbd)
            basis["half_width"] = ("ground-range strip without a recorded ground_range_max_m: "
                                   "half-width = the image extent either side of nadir_col")
        if ds is not None:
            nadir_w = np.where(known, np.sqrt(np.clip((np.where(known, altitude, 0) + ds) ** 2
                                                      - np.where(known, altitude, 0) ** 2,
                                                      0, None)), np.nan)
            basis["nadir"] = (f"nadir blind strip g0 = sqrt((altitude + ds)^2 - altitude^2) per "
                              f"row, ds = native_slant_m_per_sample = {ds:g} m: the ground band "
                              f"served by one slant-range sample, interpolated rather than "
                              f"resolved; a lower bound (beam pattern not recorded)")
        else:
            nadir_w = np.full(n, np.nan)
            basis["nadir"] = ("no native slant-range sample spacing recorded: nadir blind "
                              "strip not computed")
    else:
        raise CoverageUnavailable(
            f"strip {strip}: sidecar gives neither m_per_px_across nor slant ranges, so the "
            f"swath width is unknown")

    track = StripTrack(strip=strip, kind=kind, lat=lat, lon=lon, heading=heading,
                       altitude=altitude, time=time, quality=quality,
                       port_w=np.asarray(port_w, dtype=float),
                       stbd_w=np.asarray(stbd_w, dtype=float),
                       nadir_w=np.asarray(nadir_w, dtype=float),
                       m_per_px_along=_num(sidecar.get("m_per_px_along")),
                       synthetic=bool(sidecar.get("synthetic")), basis=basis)
    return track


def load_tracks(survey_dir: Any) -> tuple[list[StripTrack], dict[str, Any], dict[str, Any]]:
    """(tracks, export, manifest survey block). Raises CoverageUnavailable."""
    survey_dir = Path(survey_dir)
    export_path = survey_dir / "export.json"
    if not export_path.is_file():
        raise FileNotFoundError(f"no export.json in {survey_dir}")
    export = _read_json(export_path)
    manifest_path = survey_dir / "manifest.json"
    survey = {}
    if manifest_path.is_file():
        payload = _read_json(manifest_path)
        survey = payload.get("survey", {}) if isinstance(payload, dict) else {}
    navigation = survey.get("navigation") or {}
    mode = navigation.get("mode")
    sidecars = navigation.get("sidecars") or {}
    if mode not in NAV_MODES or not sidecars:
        raise CoverageUnavailable(
            "This survey has no per-ping navigation sidecar (navigation mode "
            f"{mode or 'none'!r}). Without the towfish track, heading and altitude there is "
            "no seabed footprint to compute, so none is shown.")

    root = survey_dir.resolve()
    tracks: list[StripTrack] = []
    problems: list[str] = []
    for strip, relative in sorted(sidecars.items()):
        path = (survey_dir / str(relative)).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            problems.append(f"{strip}: sidecar path escapes the survey directory")
            continue
        if not path.is_file():
            problems.append(f"{strip}: sidecar {Path(str(relative)).name} not found")
            continue
        try:
            sidecar = _read_json(path)
        except (OSError, ValueError) as exc:
            problems.append(f"{strip}: sidecar unreadable ({exc})")
            continue
        if not isinstance(sidecar, dict) or sidecar.get("format") != SIDECAR_FORMAT:
            problems.append(f"{strip}: sidecar is not {SIDECAR_FORMAT}")
            continue
        try:
            tracks.append(_strip_track(str(strip), sidecar))
        except CoverageUnavailable as exc:
            problems.append(str(exc))
    if not tracks:
        raise CoverageUnavailable("No usable navigation sidecar: " + "; ".join(problems))
    return tracks, export, survey


# --- geometry -----------------------------------------------------------------------


def _require_geometry() -> tuple[Any, Any]:
    try:
        import pyproj
        import shapely
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise CoverageUnavailable(
            f"coverage needs shapely and pyproj ({exc.name} is not installed; see "
            f"requirements-ghosttrace.txt)") from exc
    return pyproj, shapely


class _Projection:
    """Local azimuthal equidistant projection centred on the survey, metres."""

    def __init__(self, lat0: float, lon0: float) -> None:
        pyproj, _ = _require_geometry()
        self.definition = (f"+proj=aeqd +lat_0={lat0:.8f} +lon_0={lon0:.8f} +datum=WGS84 "
                           f"+units=m +no_defs")
        crs = pyproj.CRS.from_proj4(self.definition)
        self.lat0, self.lon0 = lat0, lon0
        self.forward = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        self.inverse = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    def to_xy(self, lon: Any, lat: Any) -> tuple[Any, Any]:
        return self.forward.transform(lon, lat)

    def to_lonlat(self, x: Any, y: Any) -> tuple[Any, Any]:
        return self.inverse.transform(x, y)


def _fill_heading(track: StripTrack) -> None:
    """Heading where the sidecar has none: course over ground from the projected track."""
    import numpy as np

    h = track.heading
    missing = ~np.isfinite(h)
    valid = np.isfinite(track.x) & np.isfinite(track.y)
    if not missing.any() or valid.sum() < 2:
        return
    idx = np.flatnonzero(valid)
    dx = np.gradient(track.x[idx])
    dy = np.gradient(track.y[idx])
    course = np.degrees(np.arctan2(dx, dy)) % 360.0
    full = np.full(len(h), np.nan)
    full[idx] = course
    h[missing] = full[missing]
    track.basis["heading"] = ("heading_deg from the sidecar; rows without one use course over "
                              "ground from the track")


@dataclass
class Quad:
    strip: str
    row_a: int
    row_b: int
    quality: str
    t_end: float
    swath: Any
    nadir: Any
    imaged: Any


def _edges(track: StripTrack, i: int, width: Any) -> tuple[float, float, float, float]:
    """(port x, port y, starboard x, starboard y) for row i at (port, starboard) widths."""
    h = math.radians(float(track.heading[i]))
    sx, sy = math.cos(h), -math.sin(h)            # starboard unit vector
    port, stbd = width
    x, y = float(track.x[i]), float(track.y[i])
    return x - port * sx, y - port * sy, x + stbd * sx, y + stbd * sy


def _runs(track: StripTrack) -> list[tuple[int, int, str]]:
    """Contiguous runs of rows with a position and the same quality: (start, end, quality)."""
    import numpy as np

    valid = np.isfinite(track.x) & np.isfinite(track.y) & np.isfinite(track.heading)
    runs: list[tuple[int, int, str]] = []
    start = None
    for i in range(len(track.quality)):
        if not valid[i]:
            if start is not None:
                runs.append((start, i - 1, track.quality[start]))
                start = None
            continue
        if start is None:
            start = i
        elif track.quality[i] != track.quality[start]:
            runs.append((start, i - 1, track.quality[start]))
            start = i
    if start is not None:
        runs.append((start, len(track.quality) - 1, track.quality[start]))
    return runs


def _row_step(track: StripTrack, target_m: float, rows: int) -> int:
    if track.m_per_px_along:
        return max(1, int(round(target_m / track.m_per_px_along)))
    return max(1, rows // 2000)


def _quads(track: StripTrack, clock: Any) -> list[Quad]:
    import numpy as np
    from shapely.geometry import MultiPoint

    n = len(track.quality)
    step = _row_step(track, SAMPLE_STEP_M, n)
    expected = step * (track.m_per_px_along or SAMPLE_STEP_M)
    max_jump = max(expected * 5.0, 25.0)
    valid = np.isfinite(track.x) & np.isfinite(track.y) & np.isfinite(track.heading)
    quads: list[Quad] = []
    for start, end, quality in _runs(track):
        samples = list(range(start, end + 1, step))
        if samples[-1] != end:
            samples.append(end)
        # Share the boundary row with the next run so neighbouring footprints meet.
        if end + 1 < n and valid[end + 1]:
            samples.append(end + 1)
        for a, b in zip(samples, samples[1:]):
            if math.hypot(track.x[b] - track.x[a], track.y[b] - track.y[a]) > max_jump:
                continue
            widths_a = (track.port_w[a], track.stbd_w[a])
            widths_b = (track.port_w[b], track.stbd_w[b])
            if not all(np.isfinite(widths_a + widths_b)):
                continue
            pa, pb = _edges(track, a, widths_a), _edges(track, b, widths_b)
            swath = MultiPoint([pa[:2], pb[:2], pb[2:], pa[2:]]).convex_hull
            if swath.geom_type != "Polygon" or swath.area <= 0:
                continue
            nadir = None
            if np.isfinite(track.nadir_w[a]) and np.isfinite(track.nadir_w[b]):
                na = _edges(track, a, (track.nadir_w[a], track.nadir_w[a]))
                nb = _edges(track, b, (track.nadir_w[b], track.nadir_w[b]))
                hull = MultiPoint([na[:2], nb[:2], nb[2:], na[2:]]).convex_hull
                nadir = hull if hull.geom_type == "Polygon" and hull.area > 0 else None
            imaged = swath.difference(nadir) if nadir is not None else swath
            quads.append(Quad(strip=track.strip, row_a=a, row_b=b, quality=quality,
                              t_end=float(clock[b]), swath=swath, nadir=nadir, imaged=imaged))
    return quads


def _clock(tracks: list[StripTrack]) -> tuple[list[Any], float | None, str]:
    """Seconds from survey start per row of every strip, and how that clock was made."""
    import numpy as np

    starts = [float(np.nanmin(t.time)) for t in tracks if np.isfinite(t.time).any()]
    if len(starts) == len(tracks):
        origin = min(starts)
        clocks = []
        for track in tracks:
            t = track.time.copy()
            finite = np.isfinite(t)
            if not finite.all():
                idx = np.arange(len(t))
                t = np.interp(idx, idx[finite], t[finite])
            clocks.append(t - origin)
        return clocks, origin, "ping times recorded in the navigation sidecar rows"
    clocks, offset = [], 0.0
    for track in tracks:
        clocks.append(np.arange(len(track.quality), dtype=float) + offset)
        offset += len(track.quality)
    return clocks, None, ("no ping times in the navigation sidecar: the replay clock is row "
                          "order, one row per second, and is not elapsed time")


@dataclass
class Analysis:
    survey_id: str
    export: dict[str, Any]
    survey: dict[str, Any]
    tracks: list[StripTrack]
    clocks: list[Any]
    origin: float | None
    clock_basis: str
    projection: _Projection
    quads: list[Quad]


def analyse(survey_dir: Any) -> Analysis:
    """Load the tracks, project them and build every footprint quad."""
    import numpy as np

    _require_geometry()
    survey_dir = Path(survey_dir)
    tracks, export, survey = load_tracks(survey_dir)
    lats = np.concatenate([t.lat[np.isfinite(t.lat)] for t in tracks])
    lons = np.concatenate([t.lon[np.isfinite(t.lon)] for t in tracks])
    if not len(lats):
        raise CoverageUnavailable("the navigation sidecars record no position on any row")
    projection = _Projection(float(np.mean(lats)), float(np.mean(lons)))
    for track in tracks:
        x, y = projection.to_xy(track.lon, track.lat)
        track.x, track.y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        track.x[~np.isfinite(track.lat) | ~np.isfinite(track.lon)] = np.nan
        track.y[~np.isfinite(track.lat) | ~np.isfinite(track.lon)] = np.nan
        _fill_heading(track)
    clocks, origin, clock_basis = _clock(tracks)
    quads: list[Quad] = []
    for track, clock in zip(tracks, clocks):
        quads.extend(_quads(track, clock))
    if not quads:
        raise CoverageUnavailable("the navigation sidecars yield no swath footprint")
    survey_id = str((export.get("metadata") or {}).get("survey_id") or survey_dir.name)
    return Analysis(survey_id=survey_id, export=export, survey=survey, tracks=tracks,
                    clocks=clocks, origin=origin, clock_basis=clock_basis,
                    projection=projection, quads=quads)


# --- GeoJSON helpers ----------------------------------------------------------------


def _ring_lonlat(projection: _Projection, coords: Any) -> list[list[float]]:
    import numpy as np

    array = np.asarray(coords, dtype=float)
    lon, lat = projection.to_lonlat(array[:, 0], array[:, 1])
    return [[round(float(a), 7), round(float(b), 7)] for a, b in zip(lon, lat)]


def _polygons(geometry: Any) -> list[Any]:
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if hasattr(geometry, "geoms"):
        out = []
        for part in geometry.geoms:
            out.extend(_polygons(part))
        return out
    return []


def _geojson_geometry(projection: _Projection, geometry: Any,
                      simplify_m: float = 0.2) -> dict[str, Any] | None:
    if geometry is None or geometry.is_empty:
        return None
    simplified = geometry.simplify(simplify_m, preserve_topology=True) if simplify_m else geometry
    polys = []
    for poly in _polygons(simplified):
        rings = [_ring_lonlat(projection, poly.exterior.coords)]
        rings += [_ring_lonlat(projection, ring.coords) for ring in poly.interiors]
        polys.append(rings)
    if not polys:
        return None
    if len(polys) == 1:
        return {"type": "Polygon", "coordinates": polys[0]}
    return {"type": "MultiPolygon", "coordinates": polys}


def _feature(geometry: dict[str, Any] | None, properties: dict[str, Any]) -> dict[str, Any] | None:
    if geometry is None:
        return None
    return {"type": "Feature", "geometry": geometry, "properties": properties}


def _collection(features: list[Any]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": [f for f in features if f]}


# --- coverage -----------------------------------------------------------------------


def _nearest_sample(analysis: Analysis, x: float, y: float,
                    strip: str | None = None) -> tuple[StripTrack, int] | None:
    """Nearest row with a position and heading, preferring rows with a nadir width.

    Rows inside a dropout carry no altitude, so their swath and nadir widths
    are fallbacks; the nearest row that has an altitude describes the geometry
    a re-look line will actually fly.
    """
    for need_nadir in (True, False):
        found = _nearest(analysis, x, y, strip, need_nadir)
        if found is not None:
            return found
    return None


def _nearest(analysis: Analysis, x: float, y: float, strip: str | None,
             need_nadir: bool) -> tuple[StripTrack, int] | None:
    import numpy as np

    best = None
    for track in analysis.tracks:
        if strip is not None and track.strip != strip:
            continue
        ok = np.isfinite(track.x) & np.isfinite(track.y) & np.isfinite(track.heading)
        if need_nadir:
            ok &= np.isfinite(track.nadir_w)
        if not ok.any():
            continue
        d = np.where(ok, (track.x - x) ** 2 + (track.y - y) ** 2, np.inf)
        i = int(np.argmin(d))
        if best is None or d[i] < best[0]:
            best = (float(d[i]), track, i)
    return None if best is None else (best[1], best[2])


def _line(analysis: Analysis, start: tuple[float, float], end: tuple[float, float],
          properties: dict[str, Any]) -> dict[str, Any]:
    lon, lat = analysis.projection.to_lonlat([start[0], end[0]], [start[1], end[1]])
    coords = [[round(float(lon[0]), 7), round(float(lat[0]), 7)],
              [round(float(lon[1]), 7), round(float(lat[1]), 7)]]
    heading = math.degrees(math.atan2(end[0] - start[0], end[1] - start[1])) % 360.0
    props = dict(properties)
    props.update({
        "heading_deg": round(heading, 1),
        "length_m": round(math.hypot(end[0] - start[0], end[1] - start[1]), 1),
        "start_lat": coords[0][1], "start_lon": coords[0][0],
        "end_lat": coords[1][1], "end_lon": coords[1][0],
    })
    return {"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
            "properties": props}


def _gap_lines(analysis: Analysis, gap: Any, gap_id: str, reason: str) -> list[dict[str, Any]]:
    import numpy as np

    centroid = gap.centroid
    found = _nearest_sample(analysis, centroid.x, centroid.y)
    if found is None:
        return []
    track, i = found
    h = math.radians(float(track.heading[i]))
    u = np.array([math.sin(h), math.cos(h)])       # along track
    s = np.array([math.cos(h), -math.sin(h)])      # starboard
    hw = float(min(track.port_w[i], track.stbd_w[i]))
    g0 = float(track.nadir_w[i]) if np.isfinite(track.nadir_w[i]) else 0.0
    usable = hw - g0
    if not math.isfinite(hw) or usable <= 1.0:
        return []
    coords = np.asarray(gap.convex_hull.exterior.coords if gap.geom_type != "Point"
                        else [(centroid.x, centroid.y)], dtype=float)
    c = np.array([centroid.x, centroid.y])
    rel = coords - c
    along, across = rel @ u, rel @ s
    a0, a1 = float(along.min()), float(along.max())
    c0, c1 = float(across.min()), float(across.max())
    r_mid = (g0 + hw) / 2.0
    width = c1 - c0
    centre_across = (c0 + c1) / 2.0
    # (across offset of the line from the gap centroid, what it is for)
    if width <= usable:
        plan = [(centre_across - r_mid, "gap at mid-range on the starboard side")]
    elif width <= 2.0 * hw + r_mid and 3.0 * g0 <= hw:
        # Line A images [A - hw, A + hw] but its own nadir band; line B, r_mid
        # further to starboard, images that band at mid-range on its port side
        # and extends the coverage to A + r_mid + hw.
        first = centre_across if width <= 2.0 * hw else c0 + hw
        plan = [(first, "images the gap except its own nadir band"),
                (first + r_mid, "offset by mid-range so the first line's nadir band is "
                                "imaged at mid-range on the port side")]
    else:
        count = math.ceil(width / usable - 1e-9)
        plan = [(c0 + (j + 0.5) * width / count - r_mid,
                 f"band {j + 1} of {count} at mid-range on the starboard side")
                for j in range(count)]
    lines = []
    for j, (offset, purpose) in enumerate(plan):
        centre = c + s * offset
        start = centre + u * (a0 - RUN_IN_M)
        end = centre + u * (a1 + RUN_IN_M)
        lines.append(_line(analysis, (float(start[0]), float(start[1])),
                           (float(end[0]), float(end[1])), {
            "kind": "coverage_gap",
            "target_id": gap_id,
            "target_class": None,
            "reason": (f"coverage gap {gap_id} ({reason}, {gap.area:.0f} m², {width:.0f} m "
                       f"across)" + (f", line {j + 1} of {len(plan)}" if len(plan) > 1 else "")),
            "target_offset_m": round(abs(offset), 1),
            "basis": (f"original heading; {purpose}; half-width {hw:.1f} m, nadir half-width "
                      f"{g0:.1f} m; {RUN_IN_M:g} m run-in and run-out; heuristic"),
        }))
    return lines


def _contact_reasons(detection: dict[str, Any]) -> list[str]:
    reasons = []
    pct = _num(detection.get("confidence_pct"))
    klass = str(detection.get("class_normalized") or detection.get("object_class") or "").lower()
    tier = str(detection.get("severity_tier") or "")
    if pct is not None and pct < RELOOK_CONFIDENCE_PCT:
        reasons.append(f"low confidence ({pct:.1f}% < {RELOOK_CONFIDENCE_PCT:g}%)")
    if klass in UNIDENTIFIED_CLASSES:
        reasons.append(f"unidentified class '{klass}'")
    if detection.get("suppressed") is True and tier in RELOOK_TIERS_WHEN_FILTERED:
        reasons.append(f"filtered by verification while severity tier is {tier}: a second "
                       f"aspect tests the filter")
    action = str(detection.get("recommended_action") or "")
    if detection.get("suppressed") is not True and "identification" in action.lower():
        reasons.append(f"engine action asks for identification ('{action}')")
    return reasons


def _contact_line(analysis: Analysis, detection: dict[str, Any], reasons: list[str]
                  ) -> dict[str, Any] | None:
    import numpy as np

    lat, lon = _num(detection.get("latitude")), _num(detection.get("longitude"))
    if lat is None or lon is None:
        return None
    x, y = analysis.projection.to_xy(lon, lat)
    strip = (detection.get("provenance") or {}).get("strip")
    track, i = None, None
    for candidate in analysis.tracks:
        if candidate.strip == strip and _num(detection.get("global_y")) is not None:
            row = int(min(max(round(float(detection["global_y"]) - 0.5), 0),
                          len(candidate.quality) - 1))
            if np.isfinite(candidate.heading[row]):
                track, i = candidate, row
            break
    if track is None:
        found = _nearest_sample(analysis, x, y)
        if found is None:
            return None
        track, i = found
    h2 = math.radians(float(track.heading[i]) + 90.0)
    u = np.array([math.sin(h2), math.cos(h2)])
    s = np.array([math.cos(h2), -math.sin(h2)])
    hw = float(min(track.port_w[i], track.stbd_w[i]))
    g0 = float(track.nadir_w[i]) if np.isfinite(track.nadir_w[i]) else 0.0
    if not math.isfinite(hw):
        return None
    r_mid = (g0 + hw) / 2.0
    half = max(hw, RUN_IN_M)
    centre = np.array([x, y]) - s * r_mid
    start, end = centre - u * half, centre + u * half
    return _line(analysis, (float(start[0]), float(start[1])), (float(end[0]), float(end[1])), {
        "kind": "contact",
        "target_id": detection.get("id"),
        "target_class": detection.get("object_class"),
        "target_lat": lat, "target_lon": lon,
        "confidence_pct": _num(detection.get("confidence_pct")),
        "severity_tier": detection.get("severity_tier"),
        "suppressed": detection.get("suppressed"),
        "reason": "; ".join(reasons),
        "target_offset_m": round(r_mid, 1),
        "basis": ("perpendicular to the original track at the contact (second aspect); the "
                  f"contact passes abeam at mid-range ({r_mid:.1f} m, starboard) rather than "
                  "under the nadir blind strip; heuristic, not a cited procedure"),
    })


def _metric_round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def compute_coverage(survey_dir: Any, analysis: Analysis | None = None) -> dict[str, Any]:
    """The coverage document for one survey. Raises CoverageUnavailable."""
    import numpy as np
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    analysis = analysis or analyse(survey_dir)
    quads = analysis.quads
    good = [q for q in quads if q.quality == QUALITY_OK]
    imaged = unary_union([q.imaged for q in good]) if good else Polygon()
    nadir_all = unary_union([q.nadir for q in good if q.nadir is not None])
    nadir_blind = nadir_all.difference(imaged) if not nadir_all.is_empty else nadir_all
    footprint = unary_union([q.swath for q in quads])
    hull = footprint.convex_hull

    by_reason: dict[str, Any] = {}
    for reason in sorted({q.quality for q in quads if q.quality != QUALITY_OK}):
        merged = unary_union([q.swath for q in quads if q.quality == reason])
        by_reason[reason] = merged.difference(imaged)

    gaps: list[tuple[str, Any]] = []
    for reason, geometry in by_reason.items():
        for part in _polygons(geometry):
            if part.area >= GAP_REPORT_MIN_M2:
                gaps.append((f"degraded rows: {reason}", part))
    holes_area = 0.0
    for poly in _polygons(footprint):
        for ring in poly.interiors:
            hole = Polygon(ring)
            if hole.area >= GAP_REPORT_MIN_M2:
                holes_area += hole.area
                gaps.append(("unsurveyed hole inside the survey footprint", hole))
    gaps.sort(key=lambda item: -item[1].area)

    gap_features, relook = [], []
    for k, (reason, geometry) in enumerate(gaps, start=1):
        gap_id = f"G{k:02d}"
        lines = _gap_lines(analysis, geometry, gap_id, reason) \
            if geometry.area >= GAP_MIN_AREA_M2 else []
        relook.extend(lines)
        centroid = geometry.centroid
        clon, clat = analysis.projection.to_lonlat(centroid.x, centroid.y)
        gap_features.append(_feature(_geojson_geometry(analysis.projection, geometry), {
            "id": gap_id, "reason": reason, "area_m2": round(geometry.area, 1),
            "centroid_lat": round(float(clat), 7), "centroid_lon": round(float(clon), 7),
            "relook": bool(lines),
            "relook_note": (None if lines or geometry.area >= GAP_MIN_AREA_M2 else
                            f"below the {GAP_MIN_AREA_M2:g} m² re-look threshold"),
        }))

    contacts = []
    detections = analysis.export.get("detections") or []
    for detection in detections:
        reasons = _contact_reasons(detection)
        if not reasons:
            continue
        line = _contact_line(analysis, detection, reasons)
        if line is not None:
            contacts.append((float(detection.get("severity") or 0.0), line))
    contacts.sort(key=lambda item: -item[0])
    relook.extend(line for _, line in contacts)
    for number, line in enumerate(relook, start=1):
        line["properties"] = {"id": f"RL{number:02d}", "number": number, **line["properties"]}

    # Track length and altitude, over rows with a position.
    length = 0.0
    rows_total = rows_degraded = 0
    altitudes = []
    for track in analysis.tracks:
        ok = np.isfinite(track.x) & np.isfinite(track.y)
        idx = np.flatnonzero(ok)
        if len(idx) > 1:
            length += float(np.sum(np.hypot(np.diff(track.x[idx]), np.diff(track.y[idx]))))
        rows_total += len(track.quality)
        rows_degraded += sum(1 for q in track.quality if q != QUALITY_OK)
        altitudes.extend(track.altitude[np.isfinite(track.altitude)].tolist())
    widths = np.concatenate([t.port_w + t.stbd_w for t in analysis.tracks])
    nadirs = np.concatenate([2 * t.nadir_w for t in analysis.tracks])

    degraded_total = sum(g.area for g in by_reason.values())
    metrics = {
        "track_length_km": _metric_round(length / 1000.0, 4),
        "swath_footprint_km2": _metric_round(footprint.area / 1e6),
        "imaged_km2": _metric_round(imaged.area / 1e6),
        "nadir_blind_km2": _metric_round(nadir_blind.area / 1e6),
        "degraded_gap_km2": _metric_round(degraded_total / 1e6),
        "degraded_gap_by_reason_km2": {k: _metric_round(v.area / 1e6) for k, v in by_reason.items()},
        "unsurveyed_holes_km2": _metric_round(holes_area / 1e6),
        "hull_km2": _metric_round(hull.area / 1e6),
        "imaged_pct_of_hull": round(100.0 * imaged.area / hull.area, 2) if hull.area else None,
        "imaged_pct_of_footprint": (round(100.0 * imaged.area / footprint.area, 2)
                                    if footprint.area else None),
        "mean_swath_width_m": (round(float(np.nanmean(widths)), 2)
                               if np.isfinite(widths).any() else None),
        "mean_nadir_strip_width_m": (round(float(np.nanmean(nadirs)), 2)
                                     if np.isfinite(nadirs).any() else None),
        "altitude_m": ({"min": round(min(altitudes), 2), "median": round(float(np.median(altitudes)), 2),
                        "max": round(max(altitudes), 2)} if altitudes else None),
        "strips": len(analysis.tracks),
        "rows": rows_total,
        "rows_degraded": rows_degraded,
        "gaps": len(gap_features),
        "gaps_needing_relook": sum(1 for g in gap_features if g and g["properties"]["relook"]),
        "relook_lines": len(relook),
        "relook_lines_gaps": sum(1 for l in relook if l["properties"]["kind"] == "coverage_gap"),
        "relook_lines_contacts": sum(1 for l in relook if l["properties"]["kind"] == "contact"),
    }

    metadata = analysis.export.get("metadata") or {}
    synthetic = bool(metadata.get("demo")) or any(t.synthetic for t in analysis.tracks)
    polygons = _collection([
        _feature(_geojson_geometry(analysis.projection, imaged),
                 {"kind": "imaged", "area_m2": round(imaged.area, 1),
                  "label": "Seabed imaged (good rows, nadir strip excluded)"}),
        _feature(_geojson_geometry(analysis.projection, nadir_blind),
                 {"kind": "nadir_blind", "area_m2": round(nadir_blind.area, 1),
                  "label": "Nadir blind strip"}),
        _feature(_geojson_geometry(analysis.projection, hull, simplify_m=0),
                 {"kind": "hull", "area_m2": round(hull.area, 1),
                  "label": "Convex hull of the survey footprint"}),
    ])
    return {
        "available": True,
        "format": "deepecho-coverage/1",
        "version": VERSION,
        "survey_id": analysis.survey_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "export_processed_at": metadata.get("processed_at"),
        "demo": bool(metadata.get("demo")),
        "synthetic": synthetic,
        "synthetic_note": ("SYNTHETIC DATA: the navigation and detections this coverage is "
                           "computed from were generated, not recorded." if synthetic else None),
        "metrics": metrics,
        "polygons": polygons,
        "gaps": _collection(gap_features),
        "relook_lines": _collection(relook),
        "thresholds": {"gap_min_area_m2": GAP_MIN_AREA_M2, "gap_report_min_m2": GAP_REPORT_MIN_M2,
                       "relook_confidence_pct": RELOOK_CONFIDENCE_PCT, "run_in_m": RUN_IN_M,
                       "sample_step_m": SAMPLE_STEP_M,
                       "unidentified_classes": sorted(UNIDENTIFIED_CLASSES),
                       "relook_tiers_when_filtered": sorted(RELOOK_TIERS_WHEN_FILTERED)},
        "method": {
            "projection": f"local azimuthal equidistant, WGS84 ({analysis.projection.definition})",
            "footprint": ("per-row port/starboard swath edges perpendicular to heading, sampled "
                          f"every {SAMPLE_STEP_M:g} m along track; consecutive samples form a "
                          "quad (convex hull), unioned with shapely"),
            "imaged": ("union over rows of quality 'ok' of each quad minus its nadir band; "
                       "degraded rows are excluded"),
            "percent_of_hull": "imaged area / area of the convex hull of the whole footprint",
            "gaps": ("footprint of degraded rows not imaged by any other row, by reason, plus "
                     "holes inside the survey footprint"),
            "strips": [{"strip": t.strip, "geometry": t.kind, **t.basis} for t in analysis.tracks],
            "relook": ("heuristic planning geometry; see hazard_coverage.py RE-LOOK LINES. Not "
                       "a navigation procedure: check depth, traffic and turning circle."),
        },
        "limitations": [
            "Flat seabed: no layback, pitch/yaw, refraction or across-swath slope correction.",
            "The nadir strip is the ground band served by one slant-range sample, a lower bound; "
            "the transducer beam pattern is not in the data.",
            "Imaged means the footprint on a flat seabed, not that an object of any given size "
            "would have been detected there.",
        ],
        "references": REFERENCES,
    }


# --- replay -------------------------------------------------------------------------


def compute_replay(survey_dir: Any, analysis: Analysis | None = None) -> dict[str, Any]:
    """A compact, time-ordered replay of the survey. Raises CoverageUnavailable."""
    import numpy as np
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    analysis = analysis or analyse(survey_dir)
    tracks, clocks = analysis.tracks, analysis.clocks

    total_rows = sum(int(np.sum(np.isfinite(t.x))) for t in tracks)
    step = max(1, math.ceil(total_rows / max(REPLAY_MAX_POINTS, 2)))

    points: list[dict[str, Any]] = []
    degraded: list[dict[str, Any]] = []
    strips_out: list[dict[str, Any]] = []
    order = sorted(range(len(tracks)), key=lambda k: float(np.nanmin(clocks[k])))
    for k in order:
        track, clock = tracks[k], clocks[k]
        runs = _runs(track)
        chosen: set[int] = set()
        for start, end, quality in runs:
            chosen.update(range(start, end + 1, step))
            chosen.update((start, end))
            if quality != QUALITY_OK:
                degraded.append({"strip": track.strip, "reason": quality, "rows": [start, end],
                                 "t_start": round(float(clock[start]), 2),
                                 "t_end": round(float(clock[min(end + 1, len(clock) - 1)]), 2)})
        idx = sorted(chosen)
        distance = 0.0
        previous = None
        for i in idx:
            if previous is not None:
                distance += math.hypot(track.x[i] - track.x[previous], track.y[i] - track.y[previous])
            previous = i
            points.append({
                "t": round(float(clock[i]), 2),
                "lat": round(float(track.lat[i]), 7), "lon": round(float(track.lon[i]), 7),
                "heading": round(float(track.heading[i]), 1),
                "quality": track.quality[i],
                "strip": k,
                "row": int(i),
                "half_width_port_m": round(float(track.port_w[i]), 2),
                "half_width_stbd_m": round(float(track.stbd_w[i]), 2),
                "nadir_half_width_m": (round(float(track.nadir_w[i]), 2)
                                       if np.isfinite(track.nadir_w[i]) else None),
                "_distance_m": distance,
            })
        strips_out.append({"index": k, "strip": track.strip, "geometry": track.kind,
                           "t_start": round(float(np.nanmin(clock)), 2),
                           "t_end": round(float(np.nanmax(clock)), 2),
                           "rows": len(track.quality), **track.basis})
    points.sort(key=lambda p: p["t"])
    strips_out.sort(key=lambda entry: entry["index"])

    # Distance accumulates strip after strip in time order.
    offsets: dict[str, float] = {}
    running, last_strip = 0.0, None
    for p in points:
        if p["strip"] != last_strip:
            offsets.setdefault(p["strip"], running)
            last_strip = p["strip"]
        p["distance_km"] = round((offsets[p["strip"]] + p.pop("_distance_m")) / 1000.0, 5)
        running = max(running, p["distance_km"] * 1000.0)

    # Imaged area so far: exact unions at checkpoints, linear in between.
    good = sorted((q for q in analysis.quads if q.quality == QUALITY_OK), key=lambda q: q.t_end)
    duration = max((p["t"] for p in points), default=0.0)
    checkpoints = np.linspace(0.0, duration, REPLAY_AREA_CHECKPOINTS + 1) if duration > 0 \
        else np.array([0.0])
    # The last checkpoint takes every quad: point times are rounded to 0.01 s.
    last_quad = max((q.t_end for q in good), default=0.0)
    areas, current, j = [], Polygon(), 0
    for T in checkpoints:
        batch = []
        limit = max(T, last_quad) if T == checkpoints[-1] else T + 1e-9
        while j < len(good) and good[j].t_end <= limit:
            batch.append(good[j].imaged)
            j += 1
        if batch:
            current = unary_union([current, *batch])
        areas.append(current.area / 1e6)
    for p in points:
        p["area_km2"] = round(float(np.interp(p["t"], checkpoints, areas)), 6)

    detections = []
    for detection in analysis.export.get("detections") or []:
        strip = (detection.get("provenance") or {}).get("strip")
        t = None
        for track, clock in zip(tracks, clocks):
            if track.strip == strip and _num(detection.get("global_y")) is not None:
                row = int(min(max(round(float(detection["global_y"]) - 0.5), 0), len(clock) - 1))
                t = round(float(clock[row]), 2)
                break
        if t is None:
            continue
        detections.append({
            "id": detection.get("id"), "t": t, "strip": strip,
            "row": _num(detection.get("global_y")),
            "lat": _num(detection.get("latitude")), "lon": _num(detection.get("longitude")),
            "class": detection.get("object_class"),
            "confidence_pct": _num(detection.get("confidence_pct")),
            "confidence": _num(detection.get("confidence")),
            "suppressed": detection.get("suppressed"),
            "tier": detection.get("severity_tier"),
            "severity": _num(detection.get("severity")),
            "recommended_action": detection.get("recommended_action"),
        })
    detections.sort(key=lambda d: d["t"])

    metadata = analysis.export.get("metadata") or {}
    synthetic = bool(metadata.get("demo")) or any(t.synthetic for t in tracks)
    return {
        "available": True,
        "format": "deepecho-replay/1",
        "version": VERSION,
        "survey_id": analysis.survey_id,
        "demo": bool(metadata.get("demo")),
        "synthetic": synthetic,
        "start_time": (datetime.fromtimestamp(analysis.origin, timezone.utc).isoformat(
            timespec="seconds") if analysis.origin is not None else None),
        "duration_s": round(duration, 2),
        "time_basis": analysis.clock_basis,
        "track": {key: [p[key] for p in points] for key in TRACK_KEYS},
        "track_layout": ("columnar: track[key][i] is point i; strip is an index into strips; "
                         "half widths and nadir half-width in metres; distance_km and area_km2 "
                         "are cumulative"),
        "detections": detections,
        "degraded": sorted(degraded, key=lambda d: d["t_start"]),
        "strips": strips_out,
        "totals": {
            "points": len(points),
            "distance_km": points[-1]["distance_km"] if points else 0.0,
            "area_km2": round(areas[-1], 6) if areas else 0.0,
            "contacts": sum(1 for d in detections if d["suppressed"] is not True),
            "filtered": sum(1 for d in detections if d["suppressed"] is True),
        },
        "basis": (
            f"Track points are navigation sidecar rows, one every {step} row(s) plus every "
            "quality boundary. Swath half-widths per point follow hazard_coverage.py (SWATH, "
            "PER ROW). Distance is along the projected track. Area covered is the union of "
            "imaged footprint (good rows, nadir strip excluded) computed exactly at "
            f"{REPLAY_AREA_CHECKPOINTS} checkpoints and interpolated between them. Detections "
            "appear at the ping time of the row they were found on. " + analysis.clock_basis
            + "."),
    }


# --- cache and exports ----------------------------------------------------------------

_LOCK = threading.Lock()
_MEMO: dict[tuple[str, str], tuple[tuple[float, ...], dict[str, Any]]] = {}


def _stamp(survey_dir: Path) -> tuple[float, ...]:
    files = [survey_dir / "export.json", survey_dir / "manifest.json"]
    return tuple(p.stat().st_mtime if p.is_file() else 0.0 for p in files)


def _unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason, "version": VERSION}


def coverage_for_survey(survey_dir: Any, *, use_cache: bool = True,
                        write_cache: bool = True) -> dict[str, Any]:
    """Coverage for a survey directory, cached in coverage.json beside export.json.

    The cache is reused only when it is at least as new as export.json and was
    written by this VERSION; otherwise it is recomputed and rewritten. A
    directory that cannot be written still gets its answer.
    """
    survey_dir = Path(survey_dir)
    cache = survey_dir / CACHE_FILE
    export = survey_dir / "export.json"
    if use_cache and cache.is_file() and export.is_file() \
            and cache.stat().st_mtime >= export.stat().st_mtime:
        try:
            cached = _read_json(cache)
            if isinstance(cached, dict) and cached.get("version") == VERSION:
                return cached
        except (OSError, ValueError):
            pass
    try:
        result = compute_coverage(survey_dir)
    except CoverageUnavailable as exc:
        result = _unavailable(str(exc))
    if write_cache:
        try:
            cache.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
        except OSError as exc:
            log.warning("coverage cache for %s not written: %s", survey_dir.name, exc)
    return result


def replay_for_survey(survey_dir: Any) -> dict[str, Any]:
    """The replay document, memoised in process against export/manifest mtimes."""
    survey_dir = Path(survey_dir)
    key = (str(survey_dir.resolve()), "replay")
    stamp = _stamp(survey_dir)
    with _LOCK:
        hit = _MEMO.get(key)
        if hit and hit[0] == stamp:
            return hit[1]
    try:
        result = compute_replay(survey_dir)
    except CoverageUnavailable as exc:
        result = _unavailable(str(exc))
    with _LOCK:
        _MEMO[key] = (stamp, result)
    return result


def relook_gpx(coverage: dict[str, Any]) -> str:
    """GPX 1.1: a start and end waypoint per line, each contact, and one route per line."""
    ns = "http://www.topografix.com/GPX/1/1"
    ET.register_namespace("", ns)
    gpx = ET.Element(f"{{{ns}}}gpx", {"version": "1.1",
                                      "creator": f"DeepEcho hazard_coverage {VERSION}"})
    meta = ET.SubElement(gpx, f"{{{ns}}}metadata")
    ET.SubElement(meta, f"{{{ns}}}name").text = f"{coverage.get('survey_id')} re-look lines"
    desc = ("Heuristic re-look planning lines from DeepEcho coverage analysis. Not a navigation "
            "procedure: check depth, traffic and turning circle before running any line.")
    if coverage.get("synthetic"):
        desc = "SYNTHETIC DATA. " + desc
    ET.SubElement(meta, f"{{{ns}}}desc").text = desc
    ET.SubElement(meta, f"{{{ns}}}time").text = str(coverage.get("generated_at") or "")

    lines = (coverage.get("relook_lines") or {}).get("features") or []

    def wpt(lat: float, lon: float, name: str, text: str, kind: str) -> None:
        point = ET.SubElement(gpx, f"{{{ns}}}wpt", {"lat": f"{lat:.7f}", "lon": f"{lon:.7f}"})
        ET.SubElement(point, f"{{{ns}}}name").text = name
        ET.SubElement(point, f"{{{ns}}}desc").text = text
        ET.SubElement(point, f"{{{ns}}}type").text = kind

    seen_targets = set()
    for line in lines:
        p = line["properties"]
        wpt(p["start_lat"], p["start_lon"], f"{p['id']}-S", f"Start of {p['id']}: {p['reason']}",
            "relook-start")
        wpt(p["end_lat"], p["end_lon"], f"{p['id']}-E", f"End of {p['id']}", "relook-end")
        if p.get("kind") == "contact" and p.get("target_id") not in seen_targets \
                and p.get("target_lat") is not None:
            seen_targets.add(p.get("target_id"))
            wpt(p["target_lat"], p["target_lon"], f"{p['id']}-C",
                f"Contact {p.get('target_id')} ({p.get('target_class')})", "contact")
    for line in lines:
        p = line["properties"]
        route = ET.SubElement(gpx, f"{{{ns}}}rte")
        ET.SubElement(route, f"{{{ns}}}name").text = p["id"]
        ET.SubElement(route, f"{{{ns}}}desc").text = f"{p['reason']} | {p['basis']}"
        for suffix, lat, lon in (("S", p["start_lat"], p["start_lon"]),
                                 ("E", p["end_lat"], p["end_lon"])):
            rtept = ET.SubElement(route, f"{{{ns}}}rtept", {"lat": f"{lat:.7f}", "lon": f"{lon:.7f}"})
            ET.SubElement(rtept, f"{{{ns}}}name").text = f"{p['id']}-{suffix}"
    body = ET.tostring(gpx, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body + "\n"


RELOOK_CSV_COLUMNS = ("line_id", "kind", "reason", "target_id", "target_class", "start_lat",
                      "start_lon", "end_lat", "end_lon", "heading_deg", "length_m",
                      "target_offset_m", "basis", "synthetic")


def relook_csv(coverage: dict[str, Any]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(RELOOK_CSV_COLUMNS)
    for line in (coverage.get("relook_lines") or {}).get("features") or []:
        p = line["properties"]
        writer.writerow([p.get("id"), p.get("kind"), p.get("reason"), p.get("target_id"),
                         p.get("target_class") or "", p.get("start_lat"), p.get("start_lon"),
                         p.get("end_lat"), p.get("end_lon"), p.get("heading_deg"),
                         p.get("length_m"), p.get("target_offset_m"), p.get("basis"),
                         "yes" if coverage.get("synthetic") else "no"])
    return buffer.getvalue()


if __name__ == "__main__":  # pragma: no cover - manual inspection
    import sys

    for arg in sys.argv[1:]:
        doc = coverage_for_survey(arg, write_cache=False, use_cache=False)
        print(arg, json.dumps(doc.get("metrics") or doc.get("reason"), indent=2))
