#!/usr/bin/env python3
"""The teammate's geotag.py output, as a DeepEcho survey directory.

    python import_geotag.py report/hazards.json data/surveys/<id> --xtf survey.xtf
    python import_geotag.py report/hazards.json data/surveys/<id> --nav-csv nav.csv
    python import_geotag.py report/hazards.json data/surveys/<id>      # no navigation

    from survey_hazard_map.import_geotag import import_geotag
    export = import_geotag("report/hazards.json", "data/surveys/line01", xtf="survey.xtf")

models/sonar_pipeline.py (teammate) runs the trained detector, the shadow
check, the anomaly channel and geotag.py, and writes hazards.json. This module
turns that file into exactly the directory build_hazard_map() would have
written -- manifest, tiles, export.json, actions.csv, report.csv/.geojson,
map.html, a navigation sidecar and a water column -- so the Survey Hazard Map,
GhostTrace and the Assistant consume it with no other change. The contract,
and why each field is read, is in docs/INTEGRATION_GEOTAG.md.

WHAT IS TAKEN AS GIVEN, AND WHAT IS NOT
    latitude/longitude   geotag.py's, copied unchanged. Never recomputed here.
    length_m/width_m     geotag.py's (edge ground-range difference, windowed
                         along-track spacing). With a navigation table they are
                         cross-checked independently; a disagreement over 25%
                         is recorded as a warning, never applied.
    confidence_pct       sonar_pipeline.py's (already calibrated, and fused
                         with shadow_check.py when that ran).
    severity, action,    this engine's: the same floor, score, deduplication
    hotspots, reports    and ranking as build_hazard_map.

THE WATERFALL IS SLANT RANGE
    geotag.read_xtf renders slant-range samples, not ground range. The sidecar
    written here therefore says slant_range_corrected: false, records
    slant_m_per_sample, leaves m_per_px_across null, and is registered under
    navigation mode "geotag_slant_range", which hazard_geo.references_for_survey
    does not use: nothing downstream re-locates a pixel from it. GhostTrace
    reads its ping times, track, altitude and water column.

VERIFICATION, ONE PATH ONLY
    theirs  shadow_score / verdict / vetoed from hazards.json
    ours    hazard_verify over the waterfall, nadir from the navigation table,
            no metric heights from slant geometry
    none    confidence_pct as supplied, marked unverified
    auto    theirs if any record carries shadow_score or verdict, else ours
            if a waterfall exists, else none. Never both.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from survey_hazard_map import hazard_config as cfg  # noqa: E402
from survey_hazard_map.hazard_coords import to_global  # noqa: E402
from survey_hazard_map.hazard_dedup import deduplicate  # noqa: E402
from survey_hazard_map.hazard_export import (build_export, build_summary, run_configuration,  # noqa: E402
                           safe_model_reference, utc_now, write_actions, write_export,
                           write_reports)
from survey_hazard_map.hazard_hotspots import build_hotspots  # noqa: E402
from survey_hazard_map.hazard_severity import apply_confidence_floor, normalize_class, score_detection  # noqa: E402

log = logging.getLogger("deepecho.import_geotag")

IMPORTER = "import_geotag.py"
IMPORT_VERSION = "1.0.0"
SIDECAR_FORMAT = "deepecho-strip-nav/1"
# Deliberately not "ping": hazard_geo.references_for_survey only builds a
# per-row Georeference for mode "ping", and this strip is slant range.
NAV_MODE = "geotag_slant_range"
VERIFY_CHOICES = ("auto", "theirs", "ours", "none")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

DIMENSION_BASIS = "geotag.py (edge ground-range difference; windowed along-track spacing)"
DIMENSION_BASIS_UNCHECKED = (DIMENSION_BASIS + "; as supplied in hazards.json, not "
                             "cross-checked: no navigation table was available")
DIMENSION_BASIS_NONE = ("not measured: hazards.json carries no size for this detection "
                        "(sonar_pipeline.py ran without navigation)")
CROSS_CHECK_TOLERANCE = 0.25      # relative disagreement recorded as a warning
CROSS_CHECK_MIN_ABS_M = 0.1       # below this a disagreement is pixel quantisation

SHADOW_SOURCE = "shadow_check.py (teammate)"

# Used only when hazards.json has a shadow verdict but no `vetoed` flag. The
# teammate brief's hard veto is "verdict == no-shadow AND conf < 0.6", so a
# confident detection is never removed on a missing shadow alone.
SHADOW_VETO_BELOW_CONFIDENCE = float(os.environ.get("GEOTAG_SHADOW_VETO_BELOW", "0.6"))
ANOMALY_SOURCE = "anomaly.py (teammate)"
UNKNOWN_ANOMALY = "unknown-anomaly"


# --- small helpers ----------------------------------------------------------------


def _is_abs_path(text: str) -> bool:
    return os.path.isabs(text) or text.startswith("~") or bool(re.match(r"^[A-Za-z]:[\\/]", text))


def _scrub(value: Any) -> Any:
    """Absolute filesystem paths reduced to file names. export.json gets emailed."""
    if isinstance(value, str):
        if ("/" in value or "\\" in value) and _is_abs_path(value.strip()):
            text = value.strip()
            return PureWindowsPath(text).name if "\\" in text else Path(text).name
        return value
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    return value


def _file_name(value: Any) -> Any:
    """A path-like source (nav.source, meta.source) as its file name; other text as is."""
    if isinstance(value, str) and ("/" in value or "\\" in value) and " " not in value.strip():
        text = value.strip()
        return PureWindowsPath(text).name if "\\" in text else Path(text).name
    return value


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _iso_z(value: Any) -> str | None:
    """An ISO time in UTC with a Z. A naive time is taken as UTC (XTF clocks are)."""
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _ranges(flags: list[bool], reason: str) -> list[list[Any]]:
    """[[start, end_inclusive, reason], ...] over runs of True."""
    out, start = [], None
    for i, flag in enumerate(list(flags) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append([start, i - 1, reason])
            start = None
    return out


# --- the teammate's module --------------------------------------------------------


def _module_dirs(modules: Any) -> list[Path]:
    dirs = [Path(modules)] if modules else []
    if os.environ.get("DEEPECHO_GEOTAG_MODULES"):
        dirs.append(Path(os.environ["DEEPECHO_GEOTAG_MODULES"]))
    dirs += [ROOT / "models" / "marine", ROOT / "models"]
    return dirs


def load_geotag(modules: Any = None):
    """Import geotag.py from the teammate's folder, read-only.

    Bytecode writing is switched off for the import so nothing is written into
    a folder this project does not own, and the folder is taken off sys.path
    again afterwards so its other files cannot shadow this repository's.
    """
    if "geotag" in sys.modules and hasattr(sys.modules["geotag"], "NavTable"):
        return sys.modules["geotag"]
    tried = []
    for folder in _module_dirs(modules):
        tried.append(str(folder))
        if not (folder / "geotag.py").is_file():
            continue
        entry = str(folder.resolve())
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        sys.path.insert(0, entry)
        try:
            return importlib.import_module("geotag")
        finally:
            sys.dont_write_bytecode = previous
            if entry in sys.path:
                sys.path.remove(entry)
    raise ImportError("geotag.py not found (looked in " + ", ".join(tried)
                      + "); pass modules= / --modules")


# --- inputs -----------------------------------------------------------------------


def load_hazards(path: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(meta, hazard records) from geotag.write_report's hazards.json."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {}, data
    if not isinstance(data, dict) or not isinstance(data.get("hazards"), list):
        raise ValueError(f"{path}: not a geotag hazards.json (expected {{meta, hazards: [...]}})")
    return dict(data.get("meta") or {}), list(data["hazards"])


def find_waterfall(hazards_path: Path, records: list[dict[str, Any]]) -> Path | None:
    """<hazards dir>/<image stem>_waterfall.png, else <image> itself, beside hazards.json.

    The annotated PNG write_report draws boxes on is never taken: its pixels
    are not the sonar record.
    """
    base = hazards_path.parent
    for image in dict.fromkeys(str(r.get("image")) for r in records if r.get("image")):
        name = Path(image)
        stem = name.stem if name.suffix.lower() in IMAGE_SUFFIXES else name.name
        for candidate in (base / f"{stem}_waterfall.png", base / name.name,
                          base / f"{name.name}.png"):
            if candidate.is_file() and not candidate.stem.endswith("_annotated"):
                return candidate
    return None


def read_nav_csv(geotag: Any, path: Any, nadir_col: int, simulated: bool = False):
    """A NavTable from NavTable.to_csv's columns, plus the optional ones read_xtf adds.

    roll_deg, pitch_deg, heave_m and gap_before are read when present and are
    null / false otherwise.
    """
    rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for index, record in enumerate(csv.DictReader(handle)):
            def opt(key: str) -> float | None:
                return _number(record.get(key)) if record.get(key) not in (None, "") else None
            rows.append({
                "ping": int(float(record.get("ping") or index)),
                "time": record.get("time") or "",
                "lat": float(record["lat"]), "lon": float(record["lon"]),
                "heading_deg": float(record["heading_deg"]),
                "altitude_m": float(record["altitude_m"]),
                "slant_range_m": float(record["slant_range_m"]),
                "samples_per_side": int(float(record["samples_per_side"])),
                "roll_deg": opt("roll_deg"), "pitch_deg": opt("pitch_deg"),
                "heave_m": opt("heave_m"),
                "gap_before": str(record.get("gap_before") or "").strip().lower()
                in {"1", "true", "yes"},
            })
    if not rows:
        raise ValueError(f"{path}: navigation CSV has no rows")
    nav = geotag.NavTable(rows, nadir_col, Path(path).name, simulated)
    nav.notes = []
    return nav


def _prepare_out_dir(out_dir: Path, inputs: list[Path | None], overwrite: bool) -> None:
    if out_dir.exists() and not out_dir.is_dir():
        raise FileExistsError(f"{out_dir} exists and is not a directory")
    if out_dir.is_dir() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out_dir} already holds files; pass overwrite=True "
                                  "(--overwrite) to replace it")
        resolved = out_dir.resolve()
        for item in inputs:
            if item is not None and resolved in Path(item).resolve().parents:
                raise ValueError(f"refusing to overwrite {out_dir}: it holds the input {item.name}")
        if not any((out_dir / marker).exists() for marker in
                   ("export.json", "manifest.json", "manifest.csv", cfg.TILES_DIRNAME)):
            raise FileExistsError(f"{out_dir} does not look like a survey directory; "
                                  "refusing to delete it")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


# --- navigation derived products --------------------------------------------------


def _track(geotag: Any, nav: Any, n: int) -> tuple[Any, Any, bool]:
    """Per-row lat/lon with repeated fixes interpolated over the ping sequence.

    A 1 Hz GPS under a 10 Hz sonar repeats each fix for several pings; spacing
    measured between repeats is zero. Rows between two distinct fixes are
    interpolated by row index; rows after the last change keep the last fix.
    """
    import numpy as np

    lat = np.array([float(r["lat"]) for r in nav.rows[:n]], dtype=float)
    lon = np.array([float(r["lon"]) for r in nav.rows[:n]], dtype=float)
    change = [0] + [i for i in range(1, n) if lat[i] != lat[i - 1] or lon[i] != lon[i - 1]]
    repeated = 2 <= len(change) < 0.9 * n
    if repeated:
        for a, b in zip(change[:-1], change[1:]):
            if b - a > 1:
                f = (np.arange(a, b) - a) / float(b - a)
                lat[a:b] = lat[a] + f * (lat[b] - lat[a])
                lon[a:b] = lon[a] + f * (lon[b] - lon[a])
    return lat, lon, repeated


def _spacing(geotag: Any, lat: Any, lon: Any) -> float | None:
    import numpy as np

    if len(lat) < 2:
        return None
    _, _, dist = geotag.GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    value = float(np.median(np.asarray(dist, dtype=float)))
    return round(value, 6) if value > 0 and math.isfinite(value) else None


def build_sidecar(geotag: Any, nav: Any, *, strip: str, width: int, height: int,
                  source_file: str, source_format: str, simulated: bool,
                  sensor_depth_m: float | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """(deepecho-strip-nav/1 sidecar, derived values) for one slant-range strip."""
    import numpy as np

    n = min(len(nav), height)
    lat, lon, repeated = _track(geotag, nav, n)
    along = _spacing(geotag, lat, lon)
    altitudes = [_number(r.get("altitude_m")) for r in nav.rows[:n]]
    # read_xtf writes 0 when the file recorded no altitude; a whole line of 0 is
    # "unknown", not "sitting on the seabed".
    no_altitude = all(a is None or a <= 0 for a in altitudes)
    m_per_sample = [float(r["slant_range_m"]) / max(int(r["samples_per_side"]), 1)
                    for r in nav.rows[:n]]

    rows = []
    for k in range(n):
        r = nav.rows[k]
        alt = None if no_altitude else altitudes[k]
        depth = None if sensor_depth_m is None else float(sensor_depth_m)
        rows.append({
            "row": k,
            "ping": r.get("ping"),
            "time": _iso_z(r.get("time")),
            "lat": round(float(lat[k]), 8),
            "lon": round(float(lon[k]), 8),
            "heading_deg": _number(r.get("heading_deg")),
            "altitude_m": alt,
            "depth_m": depth,
            "seabed_depth_m": (round(depth + alt, 3)
                               if depth is not None and alt is not None else None),
            "roll_deg": _number(r.get("roll_deg")),
            "pitch_deg": _number(r.get("pitch_deg")),
            "heave_m": _number(r.get("heave_m")),
            "slant_range_m": _number(r.get("slant_range_m")),
            "samples_per_side": r.get("samples_per_side"),
            "quality": "dropout" if r.get("gap_before") else "ok",
        })
    degraded = _ranges([row["quality"] == "dropout" for row in rows], "dropout")
    notes = [str(x) for x in (getattr(nav, "notes", None) or [])]
    if repeated:
        notes.append("repeated GPS fixes interpolated over the ping sequence for the "
                     "sidecar track and m_per_px_along (detection positions untouched)")
    if len(nav) != height:
        notes.append(f"navigation has {len(nav)} rows and the waterfall {height}; the sidecar "
                     f"covers the first {n}")
    sidecar = {
        "format": SIDECAR_FORMAT,
        "producer": f"{IMPORTER} {IMPORT_VERSION}",
        "source_file": source_file,
        "source_format": source_format,
        "synthetic": bool(simulated),
        "width": int(width),
        "height": int(height),
        "nadir_col": float(nav.nadir_col),
        # Null on purpose: a pixel of this image is a SLANT-range sample. A
        # ground-range scale here would let a consumer place pixels with it.
        "m_per_px_across": None,
        "m_per_px_along": along,
        "slant_range_corrected": False,
        "slant_m_per_sample": round(float(np.median(m_per_sample)), 6) if m_per_sample else None,
        "port_is_left": True,
        "coordinate_units": "degrees (WGS-84), from geotag.py navigation",
        "rows": rows,
        "degraded_rows": degraded,
        "processing": {
            "geometry": ("slant-range waterfall (geotag.read_xtf): column offset from nadir_col "
                         "is a slant-range sample index, NOT ground range. Positions come from "
                         "geotag.py per detection; this sidecar is not used to locate pixels."),
            "along_track": "median geodesic spacing between consecutive rows",
            "quality": "dropout on rows geotag flagged gap_before (time gap > 2.5x median); "
                       "every other row ok",
            "depth": ("depth_m is the sensor depth supplied to the import"
                      if sensor_depth_m is not None else
                      "no sensor depth in geotag output and none supplied; depth_m null"),
            "altitude": ("altitude_m null: every row recorded 0 (no altitude in the source)"
                         if no_altitude else "altitude_m from geotag navigation"),
            "notes": notes,
        },
    }
    return sidecar, {"lat": lat, "lon": lon, "rows": n, "repeated_fixes": repeated,
                     "no_altitude": no_altitude}


def build_water_column(grey: Any, nav: Any, sidecar: dict[str, Any]
                       ) -> tuple[dict[str, Any] | None, str]:
    """(arrays for <strip>.wc.npz, note) or (None, reason) from the slant waterfall.

    Samples nearer than the row's altitude (slant < altitude) never reach the
    seabed: they are the water column. Bin j of each side is the slant range
    (j + 0.5) * m_per_bin from the transducer, NaN at or beyond the altitude.
    """
    import numpy as np

    rows = sidecar["rows"]
    height, width = grey.shape
    altitude = np.array([np.nan if r["altitude_m"] is None else float(r["altitude_m"])
                         for r in rows], dtype=float)
    if not rows or not np.isfinite(altitude).any() or np.nanmax(altitude) <= 0:
        return None, "no altitude in the navigation, so the water column cannot be separated"
    nadir = int(round(float(sidecar["nadir_col"])))
    mps = np.array([float(r["slant_range_m"]) / max(int(r["samples_per_side"]), 1)
                    for r in rows], dtype=float)
    m_per_bin = float(np.median(mps))
    bins = int(math.ceil(float(np.nanmax(altitude)) / m_per_bin))
    if bins < 1:
        return None, "altitude shorter than one sample"
    centres = (np.arange(bins) + 0.5) * m_per_bin
    index = np.floor(centres[None, :] / mps[:, None]).astype(int)          # rows x bins
    valid = (centres[None, :] < np.where(np.isfinite(altitude), altitude, -1.0)[:, None])
    port = np.full((height, bins), np.nan, dtype=np.float32)
    starboard = np.full((height, bins), np.nan, dtype=np.float32)
    n = len(rows)
    image = np.asarray(grey, dtype=np.float32)
    port_ok = valid & (index < nadir)
    stbd_ok = valid & (nadir + index < width)
    r_idx = np.repeat(np.arange(n)[:, None], bins, axis=1)
    port_cols = np.clip(nadir - 1 - index, 0, width - 1)
    stbd_cols = np.clip(nadir + index, 0, width - 1)
    port[:n] = np.where(port_ok, image[r_idx, port_cols], np.nan)
    starboard[:n] = np.where(stbd_ok, image[r_idx, stbd_cols], np.nan)
    bottom = np.full(height, np.nan, dtype=np.float32)
    bottom[:n] = altitude
    return {"port": port, "starboard": starboard, "bottom_range_m": bottom,
            "m_per_bin": np.float32(m_per_bin), "bins": bins,
            "max_range_m": float(np.nanmax(altitude))}, "ok"


# --- detections -------------------------------------------------------------------


def _tile_for(rows: list[dict[str, Any]], cx: float, cy: float) -> dict[str, Any] | None:
    """The manifest tile containing the box centre, most central first."""
    best, best_d = None, None
    for row in rows:
        x, y = int(row["x"]), int(row["y"])
        tw, th = int(row.get("tile_width") or cfg.TILE), int(row.get("tile_height") or cfg.TILE)
        if x <= cx < x + tw and y <= cy < y + th:
            d = math.hypot(cx - (x + tw / 2.0), cy - (y + th / 2.0))
            if best_d is None or d < best_d:
                best, best_d = row, d
    return best


def _cross_check(geotag: Any, nav: Any, lat: Any, lon: Any, box: list[float],
                 width_m: Any, length_m: Any) -> dict[str, Any] | None:
    """Width and length recomputed independently of geotag.geotag, for comparison.

    width   ground range sqrt(slant^2 - altitude^2) at both box edges on the box's
            centre row; the two ranges add when the box straddles nadir
    length  geodesic distance between the track positions of rows y1 and y2,
            repeated fixes interpolated
    """
    x1, y1, x2, y2 = box
    n = len(lat)
    row = nav.rows[min(max(int((y1 + y2) / 2.0), 0), len(nav) - 1)]
    mps = float(row["slant_range_m"]) / max(int(row["samples_per_side"]), 1)
    alt = float(row["altitude_m"] or 0.0)

    def ground(col: float) -> tuple[float, str]:
        side, idx = (("port", nav.nadir_col - 1 - col) if col < nav.nadir_col
                     else ("stbd", col - nav.nadir_col))
        slant = (idx + 0.5) * mps
        return math.sqrt(max(slant * slant - alt * alt, 0.0)), side

    g1, s1 = ground(x1)
    g2, s2 = ground(x2)
    width = abs(g2 - g1) if s1 == s2 else g1 + g2
    a, b = min(max(int(y1), 0), n - 1), min(max(int(y2), 0), n - 1)
    _, _, length = geotag.GEOD.inv(lon[a], lat[a], lon[b], lat[b])
    out = {"width_m": round(width, 3), "length_m": round(float(length), 3),
           "geotag_width_m": width_m, "geotag_length_m": length_m, "disagreements": []}
    for key, mine, theirs in (("width_m", width, _number(width_m)),
                              ("length_m", float(length), _number(length_m))):
        if theirs is None:
            continue
        diff = abs(mine - theirs)
        rel = diff / max(abs(mine), abs(theirs), 1e-9)
        if rel > CROSS_CHECK_TOLERANCE and diff > CROSS_CHECK_MIN_ABS_M:
            out["disagreements"].append({"field": key, "geotag": theirs,
                                         "recomputed": round(mine, 3),
                                         "relative_difference": round(rel, 3)})
    return out


def _confidence_pct(record: dict[str, Any]) -> tuple[float | None, str]:
    pct = _number(record.get("confidence_pct"))
    anomaly = _number(record.get("anomaly_score"))
    is_anomaly = normalize_class(record.get("classification") or record.get("label")) == UNKNOWN_ANOMALY
    if is_anomaly and anomaly is not None:
        return round(100.0 * anomaly, 1), (
            f"anomaly_score x 100 from {ANOMALY_SOURCE}: an open-set novelty score against "
            "empty seabed, not a calibrated probability of any class")
    if pct is None:
        return None, "hazards.json records no confidence_pct for this detection"
    if record.get("shadow_score") is not None or record.get("verdict") is not None:
        return pct, ("sonar_pipeline.py confidence_pct: calibrated detector probability x "
                     f"(0.6 + 0.4 x shadow_score) from {SHADOW_SOURCE}; copied, not recomputed")
    return pct, ("sonar_pipeline.py confidence_pct: calibrated detector probability (no shadow "
                 "check ran on it); copied, not recomputed")


def build_records(records: list[dict[str, Any]], *, strip: str, rows: list[dict[str, Any]],
                  model_label: str, located_by_nav: bool, geotag: Any = None, nav: Any = None,
                  track: tuple[Any, Any] | None = None) -> tuple[list[dict[str, Any]], list[dict]]:
    """hazards.json records into the engine's raw detection records (pre-scoring)."""
    detections: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for index, record in enumerate(records):
        box = record.get("box_xyxy_px")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"hazard {record.get('id', index)} has no box_xyxy_px")
        x1, y1, x2, y2 = (float(v) for v in box)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        gid = record.get("id", index + 1)
        ident = f"{strip}_g{gid}"
        if ident in seen:
            seen[ident] += 1
            ident = f"{ident}_{seen[ident]}"
        else:
            seen[ident] = 1

        raw_class = str(record.get("classification") or record.get("label") or "unknown")
        cls = cfg.DOWNGRADE_LABEL if normalize_class(raw_class) == UNKNOWN_ANOMALY else raw_class
        pct, pct_basis = _confidence_pct(record)
        confidence = 0.0 if pct is None else min(max(pct / 100.0, 0.0), 1.0)

        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        tile = _tile_for(rows, cx, cy)
        tx, ty = (int(tile["x"]), int(tile["y"])) if tile else (0, 0)

        length_m, width_m = _number(record.get("length_m")), _number(record.get("width_m"))
        height_m = _number(record.get("height_m"))
        sized = length_m is not None or width_m is not None
        basis = (DIMENSION_BASIS if sized and located_by_nav else
                 DIMENSION_BASIS_UNCHECKED if sized else DIMENSION_BASIS_NONE)
        dimensions = {
            "length_m": length_m, "width_m": width_m, "height_m": height_m,
            "length_px": round(y2 - y1, 2), "width_px": round(x2 - x1, 2),
            "m_per_px_across": None, "m_per_px_along": None,
            "basis": basis,
            "height_basis": (f"{SHADOW_SOURCE} flat-seabed shadow geometry, via hazards.json"
                             if height_m is not None else
                             "not measured: hazards.json carries no height (shadow_check.py "
                             "did not run, or had no altitude / resolution)"),
        }

        if geotag is not None and nav is not None and track is not None and sized:
            check = _cross_check(geotag, nav, track[0], track[1], [x1, y1, x2, y2],
                                 width_m, length_m)
            if check and check["disagreements"]:
                checks.append({"detection_id": ident, **check})

        detection = {
            "id": ident,
            "class": cls,
            "confidence": round(confidence, 4),
            "detector_model": model_label,
            "strip": strip,
            "tile": tile["tile"] if tile else None,
            "tile_x": tx,
            "tile_y": ty,
            "bbox_tile": [round(x1 - tx, 2), round(y1 - ty, 2), round(x2 - tx, 2), round(y2 - ty, 2)],
            "latitude": _number(record.get("lat")),
            "longitude": _number(record.get("lon")),
            "dimensions": dimensions,
            "_pct": pct,
            "_pct_basis": pct_basis,
            "_hazard": record,
            "_geotag": _scrub({
                "geotag_id": gid,
                "classification": raw_class,
                "image": record.get("image"),
                "box_xyxy_px": [x1, y1, x2, y2],
                "side": record.get("side"),
                "ground_range_m": record.get("ground_range_m"),
                "slant_range_m": record.get("slant_range_m"),
                "ping": record.get("ping"),
                "ping_time": _iso_z(record.get("ping_time")) or record.get("ping_time"),
                "vehicle_altitude_m": record.get("vehicle_altitude_m"),
                "heading_deg": record.get("heading_deg"),
                "lat_dms": record.get("lat_dms"),
                "lon_dms": record.get("lon_dms"),
                "man_made": record.get("man_made"),
                "navigation": _file_name(record.get("navigation")),
                "representative_tile_rule": ("manifest tile containing the box centre, the "
                                             "most central when tiles overlap"
                                             if tile else "no tiles: hazards.json was imported "
                                                          "without a waterfall image"),
            }),
        }
        detections.append(detection)
    to_global(detections)
    return detections, checks


# --- verification -----------------------------------------------------------------


def choose_verification(verify: str, records: list[dict[str, Any]], has_waterfall: bool
                        ) -> tuple[str, str]:
    """(path, why). Exactly one path; see the module docstring."""
    if verify not in VERIFY_CHOICES:
        raise ValueError(f"verify must be one of {', '.join(VERIFY_CHOICES)}, not {verify!r}")
    theirs = any(r.get("shadow_score") is not None or r.get("verdict") is not None
                 for r in records)
    if verify != "auto":
        if verify == "ours" and not has_waterfall:
            raise ValueError("verify='ours' needs the waterfall image, and none was found")
        return verify, "chosen by the caller"
    if theirs:
        return "theirs", "auto: at least one record carries shadow_score or verdict"
    if has_waterfall:
        return "ours", "auto: no shadow_check.py fields in hazards.json, waterfall available"
    return "none", "auto: no shadow_check.py fields and no waterfall image"


def _anomaly_fields(detection: dict[str, Any]) -> dict[str, Any]:
    score = _number(detection["_hazard"].get("anomaly_score"))
    return {} if score is None else {"anomaly_score": score, "anomaly_source": ANOMALY_SOURCE}


def verify_theirs(detections: list[dict[str, Any]]) -> dict[str, Any]:
    checked = not_checked = suppressed = 0
    for det in detections:
        h = det["_hazard"]
        score, verdict = _number(h.get("shadow_score")), h.get("verdict")
        ran = score is not None or verdict is not None
        block: dict[str, Any] = {
            "version": IMPORT_VERSION, "path": "theirs", "source": SHADOW_SOURCE,
            "detector_confidence": det["confidence"], "confidence_pct": det["_pct"],
            "heuristic": True, "hard_reasons": [], "reasons": [], "notes": [],
            **_anomaly_fields(det),
        }
        is_suppressed = False
        if ran:
            checked += 1
            vetoed = h.get("vetoed")
            if vetoed is not None:
                is_suppressed, rule = bool(vetoed), "the vetoed flag sonar_pipeline.py set"
            else:
                # The teammate brief's own hard rule: a missing shadow vetoes only
                # a weak detection. A confident call with no shadow (a flush pipe,
                # a draped net) is kept and the missing shadow is recorded.
                is_suppressed = (verdict == "no-shadow"
                                 and float(det["confidence"]) < SHADOW_VETO_BELOW_CONFIDENCE)
                rule = ("no vetoed flag in hazards.json, so the brief's rule is applied: "
                        f"verdict 'no-shadow' AND confidence < {SHADOW_VETO_BELOW_CONFIDENCE:.2f}")
            text = f"{SHADOW_SOURCE} verdict '{verdict}'" + (
                f" (shadow_score {score:.3f})" if score is not None else "")
            if is_suppressed:
                block["hard_reasons"] = ["no_acoustic_shadow"]
                block["reasons"] = [text + ": no acoustic shadow behind the box, vetoed as a "
                                           "likely false positive"]
            else:
                block["reasons"] = [text]
            block.update({"status": "checked", "shadow_score": score, "verdict": verdict,
                          "vetoed": vetoed, "suppression_basis": rule})
        else:
            not_checked += 1
            block.update({"status": "not_checked",
                          "reason": "shadow_check.py did not run on this detection (no "
                                    "shadow_score or verdict in hazards.json)"})
        block["suppressed"] = is_suppressed
        suppressed += is_suppressed
        det["verification"] = block
        det["suppressed"] = is_suppressed
        det["confidence_pct"] = det["_pct"]
        det["confidence_pct_basis"] = det["_pct_basis"]
    return {"ran": True, "path": "theirs", "source": f"{SHADOW_SOURCE} via hazards.json",
            "checked": checked, "not_checked": not_checked, "suppressed": suppressed,
            "suppression_rule": "suppressed = vetoed when hazards.json carries it, else "
                                f"verdict == 'no-shadow' and confidence < "
                                f"{SHADOW_VETO_BELOW_CONFIDENCE:.2f}",
            "confidence_rule": "confidence_pct copied from sonar_pipeline.py, never recomputed"}


def verify_ours(detections: list[dict[str, Any]], grey: Any, strip: str,
                sidecar: dict[str, Any] | None) -> dict[str, Any]:
    from survey_hazard_map.hazard_verify import configuration, strip_context, verify_survey

    ctx_sidecar = None
    if sidecar is not None:
        # No across-track scale: this strip is slant range, and a height from
        # slant pixels is not something this path is allowed to produce.
        ctx_sidecar = {k: v for k, v in sidecar.items() if k != "m_per_px_across"}
    contexts = {strip: strip_context(None, ctx_sidecar, strip=strip, grey=grey)}
    # sonar_detector.py already calibrated its scores against the teammate's
    # calibration.json; this repository's calibration would be applied twice.
    summary = verify_survey(detections, contexts, None)
    for det in detections:
        block = det["verification"]
        block.update({"path": "ours", "source": "hazard_verify.py (DeepEcho)",
                      **_anomaly_fields(det)})
        block["calibration"] = {"method": "identity",
                                "basis": "sonar_detector.py scores are already calibrated "
                                         "upstream; not recalibrated here"}
        theirs_height = _number(det["_hazard"].get("height_m"))
        dims = det["dimensions"]
        if theirs_height is not None:
            dims["height_m"] = theirs_height
            dims["height_basis"] = (f"{SHADOW_SOURCE} flat-seabed shadow geometry, via "
                                    "hazards.json; hazard_verify computes no height on a "
                                    "slant-range strip")
        else:
            dims["height_basis"] = ("not measured: no height in hazards.json, and hazard_verify "
                                    "computes none on a slant-range strip")
        if det["_hazard"].get("anomaly_score") is not None:
            det["confidence_pct_basis"] = ("hazard_verify fusion over " + det["_pct_basis"])
    return {"ran": True, "path": "ours", **summary,
            "calibration": "none applied: sonar_detector.py output is calibrated upstream",
            "nadir": "from the navigation table" if sidecar else "estimated from the image",
            "heights": "not computed from slant geometry (no m_per_px_across passed)",
            "configuration": configuration()}


def verify_none(detections: list[dict[str, Any]]) -> dict[str, Any]:
    for det in detections:
        det["confidence_pct"] = det["_pct"] if det["_pct"] is not None else round(
            det["confidence"] * 100.0, 1)
        det["confidence_pct_basis"] = det["_pct_basis"] + "; not verified"
        det["suppressed"] = False
        extra = _anomaly_fields(det)
        if extra:
            det["verification"] = {"version": IMPORT_VERSION, "path": "none",
                                   "status": "not_checked",
                                   "reason": "verification not run (verify='none')",
                                   "suppressed": False, **extra}
    return {"ran": False, "path": "none",
            "note": "confidence_pct as supplied by sonar_pipeline.py; nothing verified"}


# --- the manifest -----------------------------------------------------------------


def _write_manifest(out_dir: Path, rows: list[dict[str, Any]], survey: dict[str, Any]) -> None:
    columns = list(cfg.MANIFEST_REQUIRED_COLUMNS) + [
        c for c in (rows[0] if rows else {}) if c not in cfg.MANIFEST_REQUIRED_COLUMNS]
    with (out_dir / cfg.MANIFEST_CSV).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{k: ("" if v is None else v) for k, v in row.items()} for row in rows])
    (out_dir / cfg.MANIFEST_JSON).write_text(
        json.dumps({"survey": survey, "tiles": rows}, indent=2), encoding="utf-8")


# --- the import -------------------------------------------------------------------


def import_geotag(hazards_json: Any, out_dir: Any, *, waterfall: Any = None, nav_csv: Any = None,
                  nadir_col: int | None = None, xtf: Any = None, survey_id: str | None = None,
                  title: str | None = None, verify: str = "auto",
                  sensor_depth_m: float | None = None, overwrite: bool = False,
                  modules: Any = None, weights: str | None = None,
                  altitude_m: float | None = None, ghosttrace: bool = True,
                  render: bool = True) -> dict[str, Any]:
    """Build a DeepEcho survey directory from geotag.py's hazards.json. Returns the export.

    hazards_json    geotag.write_report's hazards.json
    out_dir         the survey directory to write (refused if non-empty unless overwrite)
    waterfall       the slant-range waterfall the detector ran on; found beside
                    hazards.json as <image>_waterfall.png when omitted
    xtf             re-read navigation exactly with geotag.read_xtf (preferred)
    nav_csv         NavTable.to_csv navigation, when there is no XTF
    nadir_col       nadir column for nav_csv (default: waterfall width // 2)
    verify          "auto" | "theirs" | "ours" | "none"; see the module docstring
    sensor_depth_m  towfish depth, written to every sidecar row (geotag has none)
    modules         folder holding geotag.py (default ../models)
    weights         detector checkpoint name, for the model reference
    altitude_m      altitude override passed to read_xtf when the file records 0
    ghosttrace      run GhostTrace over the result (surveys_root = out_dir.parent)
    """
    import numpy as np

    started = time.perf_counter()
    hazards_path = Path(hazards_json)
    out_dir = Path(out_dir)
    meta, records = load_hazards(hazards_path)
    images = sorted({str(r.get("image")) for r in records if r.get("image")})
    if len(images) > 1:
        raise ValueError(f"hazards.json names {len(images)} images ({', '.join(images)}); "
                         "import one strip per hazards.json")

    waterfall_path = Path(waterfall) if waterfall else find_waterfall(hazards_path, records)
    if waterfall_path is not None and not waterfall_path.is_file():
        raise FileNotFoundError(f"waterfall not found: {waterfall_path}")
    _prepare_out_dir(out_dir, [hazards_path, waterfall_path,
                               Path(xtf) if xtf else None, Path(nav_csv) if nav_csv else None],
                     overwrite)

    simulated = bool(meta.get("simulated_navigation")) or any(
        "SIMULATED" in str(r.get("navigation") or "").upper() for r in records)
    warnings: list[str] = []

    # --- navigation ------------------------------------------------------------
    geotag = nav = grey = None
    source_file, source_format, nav_input = None, None, "hazards.json only"
    if xtf is not None:
        geotag = load_geotag(modules)
        image, nav = geotag.read_xtf(Path(xtf), altitude_override=altitude_m)
        source_file, source_format, nav_input = Path(xtf).name, "xtf", "xtf (geotag.read_xtf)"
        if waterfall_path is not None:
            from PIL import Image
            with Image.open(waterfall_path) as handle:
                grey = np.asarray(handle.convert("L"))
            if grey.shape != image.shape:
                raise ValueError(f"{waterfall_path.name} is {grey.shape[1]}x{grey.shape[0]} but "
                                 f"{Path(xtf).name} renders {image.shape[1]}x{image.shape[0]}; "
                                 "the boxes cannot be matched to this navigation")
        else:
            grey = np.asarray(image)
    elif nav_csv is not None:
        geotag = load_geotag(modules)
        if waterfall_path is not None:
            from PIL import Image
            with Image.open(waterfall_path) as handle:
                grey = np.asarray(handle.convert("L"))
        if nadir_col is None:
            if grey is None:
                raise ValueError("nav_csv without a waterfall needs nadir_col")
            nadir_col = grey.shape[1] // 2
        nav = read_nav_csv(geotag, nav_csv, int(nadir_col), simulated)
        source_file, source_format, nav_input = Path(nav_csv).name, "nav_csv", "nav_csv (NavTable)"
    elif waterfall_path is not None:
        from PIL import Image
        with Image.open(waterfall_path) as handle:
            grey = np.asarray(handle.convert("L"))
    simulated = simulated or bool(getattr(nav, "simulated", False))
    notes = [str(n) for n in (getattr(nav, "notes", None) or meta.get("notes") or [])]
    for note in notes:
        if "WARNING" in note.upper():
            warnings.append(f"geotag.py: {note}")

    # --- strip, tiles, manifest -----------------------------------------------
    if waterfall_path is not None:
        strip = waterfall_path.stem
    elif xtf is not None:
        strip = f"{Path(xtf).stem}_waterfall"
    else:
        strip = Path(images[0]).stem if images else "geotag"
    strip = re.sub(r"[^A-Za-z0-9._-]+", "_", strip).strip("._") or "geotag"

    rows: list[dict[str, Any]] = []
    survey_meta: dict[str, Any] = {}
    tiles_dir = out_dir / cfg.TILES_DIRNAME
    if grey is not None:
        from PIL import Image

        from survey_hazard_map.survey_preparation import STRIPS_DIRNAME, load_manifest, prepare_survey
        strips_dir = out_dir / STRIPS_DIRNAME
        strips_dir.mkdir(parents=True, exist_ok=True)
        strip_png = strips_dir / f"{strip}.png"
        Image.fromarray(np.asarray(grey, dtype=np.uint8), mode="L").save(strip_png, "PNG")
        prepare_survey([strip_png], out_dir)
        rows, survey_meta = load_manifest(out_dir / cfg.MANIFEST_JSON)
    else:
        warnings.append("no waterfall image: no tiles, no imagery, no verification from pixels")
        survey_meta = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "engine": cfg.ENGINE_NAME, "processing_version": cfg.PROCESSING_VERSION,
            "strips": [{"strip": strip, "source_image": images[0] if images else None,
                        "width": None, "height": None, "m_per_px_across": None,
                        "m_per_px_along": None, "resolution_basis": None}],
            "tiles_written": 0, "tiles_skipped_low_content": 0, "strips_unreadable": [],
            "tiling": {"tile": cfg.TILE, "stride": cfg.STRIDE, "overlap": cfg.TILE - cfg.STRIDE,
                       "min_content": cfg.MIN_CONTENT, "denoise": cfg.DENOISE,
                       "note": "no waterfall image was imported, so nothing was tiled"},
        }
    height, width = (grey.shape if grey is not None else (0, 0))

    sidecar = derived = None
    navigation: dict[str, Any] = {
        "mode": NAV_MODE if nav is not None else "geotag_positions",
        "source": "geotag.py latitude/longitude per detection (teammate), copied unchanged",
        "navigation_input": nav_input,
        "position_rule": ("every detection position is geotag.py's; nothing in this survey "
                          "re-locates a pixel. The strip is slant range, so no ground-range "
                          "transform is registered for it."),
        "across_track_resolved": True,
    }
    wc_reason = "no navigation table (no xtf or nav_csv), so no altitude to separate it"
    if nav is not None:
        if grey is None:
            height, width = len(nav), 2 * int(nav.nadir_col)
        sidecar, derived = build_sidecar(
            geotag, nav, strip=strip, width=width, height=height, source_file=source_file,
            source_format=source_format, simulated=simulated, sensor_depth_m=sensor_depth_m)
        nav_dir = out_dir / "nav"
        nav_dir.mkdir(parents=True, exist_ok=True)
        wc = None
        if grey is not None:
            wc, wc_reason = build_water_column(grey, nav, sidecar)
        else:
            wc_reason = "no waterfall image, so no water-column samples"
        if wc is not None:
            wc_name = f"{strip}.wc.npz"
            np.savez_compressed(nav_dir / wc_name, port=wc["port"], starboard=wc["starboard"],
                                bottom_range_m=wc["bottom_range_m"], m_per_bin=wc["m_per_bin"])
            sidecar["water_column"] = {
                "path": wc_name, "m_per_bin": round(float(wc["m_per_bin"]), 6),
                "max_range_m": round(wc["max_range_m"], 3), "bins": wc["bins"],
                "rows": height,
                "arrays": {"port": "rows x bins float32", "starboard": "rows x bins float32",
                           "bottom_range_m": "rows float32, the row's altitude (flat seabed), "
                                             "NaN if unknown"},
                "note": ("derived from the slant-range waterfall: samples with slant range < "
                         "altitude. Values are the waterfall's 8-bit log-scaled grey levels "
                         "used AS IS (not converted back to linear intensity); the activity "
                         "method's per-bin median/MAD background is insensitive to that "
                         "monotonic scaling. Not a recorded water-column channel."),
                "bin_order": "bin 0 at the transducer on both channels (port is NOT mirrored)",
                "row_alignment": "row i is row i of the waterfall and of the sidecar rows",
            }
            navigation["water_columns"] = {strip: f"nav/{wc_name}"}
        else:
            sidecar["water_column"] = None
            sidecar["processing"]["water_column"] = f"not written: {wc_reason}"
            warnings.append(f"no water column: {wc_reason}")
        (nav_dir / f"{strip}.nav.json").write_text(json.dumps(sidecar, separators=(",", ":")),
                                                   encoding="utf-8")
        navigation["sidecars"] = {strip: f"nav/{strip}.nav.json"}
        navigation["slant_range_corrected"] = False
        navigation["slant_m_per_sample"] = sidecar["slant_m_per_sample"]
        navigation["m_per_px_along"] = sidecar["m_per_px_along"]
        navigation["sidecar_use"] = ("ping times, track, altitude and water column for "
                                     "GhostTrace; not a pixel-locating transform")
        # Tile centres through geotag's own geometry, so tile rows carry real
        # positions (GhostTrace coverage, map imagery placement for display).
        for row in rows:
            g = nav.pixel_to_latlon(float(row["center_x"]), float(row["center_y"]))
            row["lat"], row["lon"] = round(float(g["lat"]), 8), round(float(g["lon"]), 8)
        if rows:
            navigation["tile_positions"] = ("tile centres located with geotag.NavTable."
                                            "pixel_to_latlon (slant to ground, flat seabed)")
    else:
        navigation["note"] = ("no navigation table: positions exist only where hazards.json "
                              "records them; no sidecar, no water column")

    # --- detections --------------------------------------------------------------
    model_name = Path(weights).name if weights else "sonar_detector"
    model_label = f"geotag:{Path(model_name).stem}"
    detections, checks = build_records(
        records, strip=strip, rows=rows, model_label=model_label,
        located_by_nav=nav is not None, geotag=geotag, nav=nav,
        track=(derived["lat"], derived["lon"]) if derived else None)
    raw_count = len(detections)
    withheld = 0
    for det in detections:
        apply_confidence_floor(det)
        withheld += "class_withheld" in det
        score_detection(det)
    detections = deduplicate(detections)

    path, why = choose_verification(verify, records, grey is not None)
    if path == "theirs":
        verification = verify_theirs(detections)
    elif path == "ours":
        verification = verify_ours(detections, np.asarray(grey, dtype=np.float32), strip, sidecar)
    else:
        verification = verify_none(detections)
    verification["choice"] = why
    verification["choice_rule"] = ("auto = theirs if any record carries shadow_score or verdict, "
                                   "else ours if a waterfall exists, else none; never both")
    for det in detections:
        det.setdefault("suppressed", False)

    located = any(d.get("latitude") is not None for d in detections) or any(
        r.get("lat") is not None for r in rows)
    georeferenced = any(d.get("latitude") is not None for d in detections)
    coordinate_mode = cfg.COORD_MODE_GEO if georeferenced else cfg.COORD_MODE_RELATIVE
    reportable = [d for d in detections if not d.get("suppressed")]
    hotspots = build_hotspots(reportable, None)

    for check in checks:
        for item in check["disagreements"]:
            warnings.append(f"{check['detection_id']}: geotag {item['field']} {item['geotag']} "
                            f"vs independent {item['recomputed']} "
                            f"({item['relative_difference']:.0%} apart); geotag's value kept")
    if simulated:
        warnings.insert(0, "navigation was SIMULATED by geotag.py: no position or along-track "
                           "size in this survey is real")

    # --- manifest survey block -------------------------------------------------
    survey_id = survey_id or out_dir.resolve().name
    title = title or f"{strip} (geotag import)"
    survey_meta.update({
        "survey_id": survey_id, "title": title,
        "coordinate_mode": cfg.COORD_MODE_GEO if located else cfg.COORD_MODE_RELATIVE,
        "navigation": navigation,
        "imported_by": f"{IMPORTER} {IMPORT_VERSION}",
    })
    if sidecar is not None and survey_meta.get("strips"):
        for entry in survey_meta["strips"]:
            if entry.get("strip") == strip:
                entry.update({"sidecar": navigation["sidecars"][strip],
                              "water_column": (navigation.get("water_columns") or {}).get(strip),
                              "nadir_col": sidecar["nadir_col"], "source_file": source_file,
                              "source_format": source_format, "synthetic": simulated,
                              "slant_range_corrected": False,
                              "slant_m_per_sample": sidecar["slant_m_per_sample"],
                              "degraded_rows": sidecar["degraded_rows"]})
    if simulated:
        survey_meta["synthetic_strips"] = [strip]
        survey_meta["synthetic_warning"] = ("SIMULATED NAVIGATION: geotag.py generated this "
                                            "track; no position on it is real.")
    _write_manifest(out_dir, rows, _scrub(survey_meta))

    # --- export ------------------------------------------------------------------
    summary = build_summary(raw_count=raw_count, detections=detections, hotspots=hotspots,
                            strips=[strip], tiles=len(rows), georeferenced=georeferenced,
                            coordinate_mode=coordinate_mode)
    summary["model_confidence_threshold"] = None

    model_reference = safe_model_reference(model_name)
    configuration = run_configuration()
    configuration["detection"] = {
        "CONF_THRESH": None, "imgsz": None,
        "note": "detection ran upstream in models/sonar_pipeline.py (sonar_detector.py); "
                "its thresholds are not recorded in hazards.json"}
    configuration["import"] = {"importer": IMPORTER, "version": IMPORT_VERSION,
                               "verify_requested": verify, "verify_path": path,
                               "cross_check_tolerance": CROSS_CHECK_TOLERANCE}

    metadata = {
        "engine": cfg.ENGINE_NAME,
        "processing_version": cfg.PROCESSING_VERSION,
        "processed_at": utc_now(),
        "processing_seconds": round(time.perf_counter() - started, 3),
        "survey_id": survey_id,
        "title": title,
        "coordinate_mode": coordinate_mode,
        "confidence_threshold": None,
        **model_reference,
        "detector_classes": sorted({str(d["_geotag"]["classification"]) for d in detections}),
        "detector": "sonar_detector.py via sonar_pipeline.py and geotag.py (teammate)",
        "demo": simulated,
        "imported_from": "geotag.py hazards.json",
        "importer": f"{IMPORTER} {IMPORT_VERSION}",
        "warnings": warnings,
    }
    if simulated:
        metadata["data_source"] = "SYNTHETIC DEMO DATA: SIMULATED NAVIGATION"
        metadata["demo_warning"] = (
            "The navigation for this survey was SIMULATED by geotag.py (--simulate-nav). "
            "The detections come from the sonar image, but every latitude, longitude and "
            "along-track size rests on an invented track, so no position here is real "
            "and nothing here is evidence of where anything is.")

    provenance = {
        "demo": simulated,
        "processing_version": cfg.PROCESSING_VERSION,
        "severity_policy_version": cfg.SEVERITY_POLICY_VERSION,
        "coordinate_mode": coordinate_mode,
        "coordinate_note": (
            "global_x and global_y are pixel offsets within the slant-range waterfall and are "
            "present for every detection. Latitude and longitude are geotag.py's, copied "
            "unchanged, and null wherever hazards.json has none."),
        "navigation": navigation,
        "source_strips": survey_meta.get("strips", [{"strip": strip}]),
        "tile_count": len(rows),
        "tiles_in_manifest": len(rows),
        "tiles_processed": len(rows),
        "tiles_failed": [],
        "manifest": cfg.MANIFEST_JSON,
        "tiling": survey_meta.get("tiling", {}),
        "model": model_reference,
        "confidence_threshold": None,
        "severity_formula": "severity = class_weight * confidence",
        "confidence_input": ("confidence = sonar_pipeline.py confidence_pct / 100 (or "
                             "anomaly_score for unknown_anomaly)"),
        "class_confidence_floors": dict(cfg.CLASS_CONFIDENCE_FLOOR),
        "class_floor_rule": (
            "a detection below its class's floor is relabelled "
            f"'{cfg.DOWNGRADE_LABEL}' and never dropped; the original call is kept in the "
            "detection's downgraded_from."),
        "class_floors_withheld": withheld,
        "ranking_rule": "hotspots sorted by total_severity descending; H001 is the highest "
                        "priority",
        "audit_note": ("Every hotspot carries a `rationale`; every detection carries its "
                       "geotag.py record under provenance.geotag."),
        "verification": verification,
        "suppression_rule": (
            "a detection marked suppressed is kept in `detections` with its reasons and "
            "excluded from hotspots and the action list; nothing is deleted"),
        "strip_resolutions_m_per_px": {},
        "import": _scrub({
            "importer": f"{IMPORTER} {IMPORT_VERSION}",
            "hazards_json": hazards_path.name,
            "hazards_meta": {k: (_file_name(v) if k == "source" else v) for k, v in meta.items()},
            "records_read": len(records),
            "waterfall": waterfall_path.name if waterfall_path else None,
            "xtf": Path(xtf).name if xtf else None,
            "nav_csv": Path(nav_csv).name if nav_csv else None,
            "sensor_depth_m": sensor_depth_m,
            "id_rule": "detection id = <strip>_g<geotag id>; geotag ids are run-local, so an id "
                       "is stable for a given hazards.json only",
            "class_rule": "classification copied; unknown_anomaly becomes 'unknown'",
            "dimension_rule": DIMENSION_BASIS + " copied; see dimension_cross_check",
            "dimension_cross_check": {
                "method": ("width from sqrt(slant^2 - altitude^2) at the box edges on the centre "
                           "row (ranges added across nadir); length as the geodesic distance "
                           "between the track positions of rows y1 and y2"),
                "tolerance": CROSS_CHECK_TOLERANCE,
                "ran": nav is not None,
                "disagreements": checks,
            },
            "water_column": ("written" if navigation.get("water_columns") else
                             f"not written: {wc_reason}"),
        }),
        "geotag_notes": _scrub(notes),
        "geotag_limitations": _scrub(list(meta.get("limitations") or [])),
    }

    export = build_export(metadata=metadata, summary=summary, detections=detections,
                          hotspots=hotspots, provenance=provenance, configuration=configuration)
    by_id = {d["id"]: d for d in detections}
    for exported in export["detections"]:
        exported["provenance"]["geotag"] = by_id[exported["id"]]["_geotag"]
    export = _scrub(export)
    write_export(out_dir, export)
    write_actions(out_dir, hotspots)
    write_reports(out_dir, export)

    if render:
        from survey_hazard_map.hazard_mapview import render_map
        render_map(export, out_dir, tiles_dir if rows else None, out_dir / cfg.MANIFEST_JSON,
                   title=title, demo=simulated,
                   # With no imagery to draw, a basemap is the only thing under the
                   # dots. It stays off when the map opens and needs a network.
                   basemap=georeferenced and not rows)

    if ghosttrace:
        from ghosttrace.engine import run_ghosttrace
        run_ghosttrace(out_dir, surveys_root=out_dir.parent)

    log.info("imported %d hazard(s) into %s: %d detection(s), %d suppressed, %d hotspot(s), "
             "verification %s, %s", len(records), out_dir, len(detections),
             summary["suppressed_detections"], len(hotspots), path, coordinate_mode)
    return export


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hazards_json", help="hazards.json written by geotag.py / sonar_pipeline.py")
    parser.add_argument("out_dir", help="survey directory to write, e.g. data/surveys/<id>")
    parser.add_argument("--waterfall", help="the waterfall image the detector ran on")
    parser.add_argument("--xtf", help="the raw .xtf: navigation re-read with geotag.read_xtf")
    parser.add_argument("--nav-csv", help="navigation CSV (NavTable.to_csv columns)")
    parser.add_argument("--nadir-col", type=int, help="nadir column for --nav-csv")
    parser.add_argument("--survey-id")
    parser.add_argument("--title")
    parser.add_argument("--verify", choices=VERIFY_CHOICES, default="auto")
    parser.add_argument("--sensor-depth", type=float, dest="sensor_depth_m",
                        help="towfish depth in metres (geotag output has none)")
    parser.add_argument("--altitude", type=float, dest="altitude_m",
                        help="altitude override for read_xtf when the file records 0")
    parser.add_argument("--modules", help="folder holding geotag.py (default ../models)")
    parser.add_argument("--weights", help="detector checkpoint name for the model reference")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-ghosttrace", action="store_true")
    parser.add_argument("--no-map", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)-7s %(message)s")
    try:
        export = import_geotag(
            args.hazards_json, args.out_dir, waterfall=args.waterfall, nav_csv=args.nav_csv,
            nadir_col=args.nadir_col, xtf=args.xtf, survey_id=args.survey_id, title=args.title,
            verify=args.verify, sensor_depth_m=args.sensor_depth_m, overwrite=args.overwrite,
            modules=args.modules, weights=args.weights, altitude_m=args.altitude_m,
            ghosttrace=not args.no_ghosttrace, render=not args.no_map)
    except (FileExistsError, FileNotFoundError, ValueError, ImportError) as exc:
        print(f"import_geotag: {exc}", file=sys.stderr)
        return 1
    summary = export["survey_summary"]
    print(f"{args.out_dir}: {summary['total_deduplicated_detections']} detection(s), "
          f"{summary['suppressed_detections']} suppressed, {summary['total_hotspots']} "
          f"hotspot(s), {summary['coordinate_mode']}, verification "
          f"{export['provenance']['verification']['path']}")
    for warning in export["metadata"].get("warnings") or []:
        print(f"  WARNING {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
