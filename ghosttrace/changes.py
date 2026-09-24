"""Has it moved, is it new, is it gone? Target change across repeat surveys.

Why this matters for a rescue decision: a net that has MOVED since the last
survey is mobile gear, still sweeping new seabed and still catching along its
path. A net that is NEW arrived since the last look over the same spot. A net
that is PERSISTENT is a known, stationary problem. And a net that has
disappeared from a spot the current survey actually covered was removed,
buried, or drifted off - all worth knowing, none of them provable from sonar
alone, which is why `removed_since_previous` carries a basis on every record.

WHAT IS COMPARED
    Every other survey directory under the surveys root that
      * has ghosttrace.json (preferred: targets already selected) or
        export.json with georeferenced detections,
      * is earlier than the current survey (survey time from the navigation
        sidecar rows when both have it, else export metadata.processed_at -
        the basis records which), and
      * overlaps the current survey's lat/lon bounding box grown by
        EXTENT_MARGIN_M.

    When several earlier surveys overlap, the most recent observation of a
    place wins: a prior target is only eligible if no NEWER prior survey's
    coverage contains it. Otherwise a net recovered between survey 1 and
    survey 2 would be reported "removed" again by survey 3.

MATCHING
    Hungarian assignment (scipy.optimize.linear_sum_assignment) on geodesic
    (haversine) distance between current and eligible prior targets. A pair is
    gated out (not assignable) when
      * distance > MATCH_RADIUS_M (tens of metres of navigation disagreement
        between two surveys is normal, so the gate is well above it),
      * the class families differ (gear vs debris; see CLASS_FAMILIES),
      * both have plan dimensions and their area ratio > SIZE_RATIO_MAX.
    Within the gate the cost is distance + SIZE_PENALTY_M * ln(area ratio).

STATUS
    persistent          matched, displacement <= STATIONARY_M
    moved               matched, displacement > STATIONARY_M (within the gate)
    new                 unmatched, and inside an earlier survey's coverage
    first_survey        unmatched, and no earlier survey covered the position
                        (or there is no earlier overlapping survey at all)
    unmatched_no_prior  the target has no position (survey not georeferenced),
                        so it cannot be compared with anything
    removed             a prior eligible target inside the CURRENT survey's
                        coverage with no match. Coverage is the convex hull of
                        the current survey's located tile centres, located
                        detections and sampled track rows, all of which lie
                        inside the swath, so the hull under-states coverage and
                        an object near the swath edge is not called removed.
    removed: confirmed recovered
                        a removed prior target for which a field device
                        (net_finder, see ghosttrace/telemetry.py) reported a
                        recovery between the prior survey and this one. The
                        recovery record (time, device, simulated flag,
                        distance from the target) is attached. A recovery
                        reported AFTER this survey does not explain an absence
                        in it, so that record stays plain "removed" with a note.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ghosttrace import config_core as cfg

_SEP = re.compile(r"[\s_/]+")


# --- class helpers (used by engine and alerts as well) --------------------------

def normalize_class(name: Any) -> str:
    text = _SEP.sub("-", str(name or "").strip().lower())
    return text.strip("-")


def class_matches(name: Any, key: str) -> bool:
    """True when `key` appears in `name` as whole hyphen-delimited token(s)."""
    norm = normalize_class(name)
    if not norm:
        return False
    if norm == key:
        return True
    parts = key.split("-")
    raw = [t for t in norm.split("-") if t]
    # Plurals: "nets" -> "net". Only for tokens longer than three letters so
    # "gas" or "bus" are not mangled into something they are not.
    singular = [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in raw]
    for seq in (raw, singular):
        for i in range(len(seq) - len(parts) + 1):
            if seq[i:i + len(parts)] == parts:
                return True
    return False


def class_family(name: Any) -> str | None:
    for family, keys in cfg.CLASS_FAMILIES.items():
        if any(class_matches(name, k) for k in keys):
            return family
    return None


def is_target_class(name: Any, include_unknown: bool | None = None) -> bool:
    include_unknown = cfg.INCLUDE_UNKNOWN if include_unknown is None else include_unknown
    if any(class_matches(name, k) for k in cfg.GHOSTTRACE_CLASSES):
        return True
    return bool(include_unknown) and class_family(name) == "unknown"


# --- geometry -----------------------------------------------------------------------

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(h)))


def _enu(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    r = 6371008.8
    return (math.radians(lon - lon0) * r * math.cos(math.radians(lat0)),
            math.radians(lat - lat0) * r)


class Coverage:
    """Convex hull of positions known to lie inside a survey's swath."""

    def __init__(self, points: list[tuple[float, float]]):
        self.points = [(float(a), float(b)) for a, b in points
                       if a is not None and b is not None]
        self.hull = None
        self.reason = None
        if len(self.points) < 3:
            self.reason = f"only {len(self.points)} located positions; no area can be formed"
            return
        self.lat0 = sum(p[0] for p in self.points) / len(self.points)
        self.lon0 = sum(p[1] for p in self.points) / len(self.points)
        try:
            import numpy as np
            from scipy.spatial import Delaunay
            xy = np.array([_enu(self.lat0, self.lon0, a, b) for a, b in self.points])
            self.hull = Delaunay(xy)
        except Exception as exc:  # collinear points (QhullError) and friends
            self.reason = f"positions do not enclose an area ({type(exc).__name__})"

    def contains(self, lat: float | None, lon: float | None) -> bool:
        if self.hull is None or lat is None or lon is None:
            return False
        x, y = _enu(self.lat0, self.lon0, float(lat), float(lon))
        return bool(self.hull.find_simplex([[x, y]])[0] >= 0)

    def bbox(self) -> tuple[float, float, float, float] | None:
        if not self.points:
            return None
        lats = [p[0] for p in self.points]
        lons = [p[1] for p in self.points]
        return min(lats), min(lons), max(lats), max(lons)


# --- survey loading -----------------------------------------------------------------

def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            from datetime import timezone
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


def nav_sidecars(survey_dir: Path, manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Every readable navigation sidecar of a survey, keyed by strip."""
    out: dict[str, dict[str, Any]] = {}
    navigation = ((manifest or {}).get("survey") or {}).get("navigation") or {}
    for strip, rel in (navigation.get("sidecars") or {}).items():
        p = Path(str(rel))
        p = p if p.is_absolute() else survey_dir / p
        data = _load(p) if p.is_file() else None
        if isinstance(data, dict):
            out[strip] = data
    nav_dir = survey_dir / "nav"
    if nav_dir.is_dir():
        for p in sorted(nav_dir.glob("*.nav.json")):
            strip = p.name[:-len(".nav.json")]
            if strip not in out:
                data = _load(p)
                if isinstance(data, dict):
                    out[strip] = data
    return out


def _row_time(row: dict[str, Any]) -> Any:
    for key in ("time", "time_utc", "timestamp", "datetime"):
        if row.get(key):
            return row[key]
    return None


def survey_time(survey_dir: Any, export: dict[str, Any] | None = None,
                manifest: dict[str, Any] | None = None) -> tuple[str | None, str]:
    """(ISO time, basis). Earliest sidecar row time, else export processed_at."""
    survey_dir = Path(survey_dir)
    export = export if export is not None else (_load(survey_dir / "export.json") or {})
    manifest = manifest if manifest is not None else (_load(survey_dir / "manifest.json") or {})
    times = []
    for sidecar in nav_sidecars(survey_dir, manifest).values():
        for row in sidecar.get("rows") or []:
            t = _parse_time(_row_time(row))
            if t:
                times.append(t)
                break
    if times:
        return min(times).isoformat(), "earliest ping time in the navigation sidecars"
    processed = (export.get("metadata") or {}).get("processed_at")
    if processed:
        return str(processed), ("export metadata.processed_at (no ping times recorded; this "
                                "is when the survey was PROCESSED, not when it was recorded)")
    return None, "no survey time recorded"


def coverage_points(survey_dir: Path, export: dict[str, Any] | None,
                    manifest: dict[str, Any] | None) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for tile in (manifest or {}).get("tiles") or []:
        if tile.get("lat") not in (None, "") and tile.get("lon") not in (None, ""):
            pts.append((float(tile["lat"]), float(tile["lon"])))
    for det in (export or {}).get("detections") or []:
        if det.get("latitude") is not None and det.get("longitude") is not None:
            pts.append((float(det["latitude"]), float(det["longitude"])))
    for sidecar in nav_sidecars(survey_dir, manifest).values():
        rows = sidecar.get("rows") or []
        for row in rows[::max(1, cfg.COVERAGE_TRACK_STEP)]:
            if row.get("lat") is not None and row.get("lon") is not None:
                pts.append((float(row["lat"]), float(row["lon"])))
    return pts


def _plan_area(dims: Any) -> float | None:
    if not isinstance(dims, dict):
        return None
    length, width = dims.get("length_m"), dims.get("width_m")
    if length is None or width is None:
        return None
    try:
        area = float(length) * float(width)
    except (TypeError, ValueError):
        return None
    return area if area > 0 else None


def prior_targets_of(survey_dir: Path) -> tuple[list[dict[str, Any]], str]:
    """Georeferenced GhostTrace-class targets of a processed survey, and the source file."""
    gt = _load(survey_dir / cfg.OUTPUT_JSON)
    if isinstance(gt, dict) and isinstance(gt.get("targets"), list):
        out = []
        for t in gt["targets"]:
            if t.get("latitude") is None or t.get("longitude") is None or t.get("suppressed"):
                continue
            out.append({"detection_id": t.get("detection_id"), "object_class": t.get("object_class"),
                        "latitude": float(t["latitude"]), "longitude": float(t["longitude"]),
                        "dimensions": t.get("dimensions")})
        return out, cfg.OUTPUT_JSON
    export = _load(survey_dir / "export.json") or {}
    out = []
    for d in export.get("detections") or []:
        if d.get("latitude") is None or d.get("longitude") is None or d.get("suppressed"):
            continue
        if not is_target_class(d.get("class_normalized") or d.get("object_class")):
            continue
        out.append({"detection_id": d.get("id"), "object_class": d.get("object_class"),
                    "latitude": float(d["latitude"]), "longitude": float(d["longitude"]),
                    "dimensions": d.get("dimensions")})
    return out, "export.json"


def _bbox_overlap(a, b, margin_m: float) -> bool:
    if a is None or b is None:
        return False
    lat_pad = margin_m / 111_320.0
    mid = math.radians((a[0] + a[2]) / 2.0)
    lon_pad = margin_m / (111_320.0 * max(math.cos(mid), 1e-6))
    return not (a[2] + lat_pad < b[0] or b[2] + lat_pad < a[0]
                or a[3] + lon_pad < b[1] or b[3] + lon_pad < a[1])


# --- the comparison -------------------------------------------------------------------

REMOVED_STATUS = "removed"
RECOVERED_STATUS = "removed: confirmed recovered"


def _default_recoveries() -> dict[str, dict[str, Any]]:
    try:
        from ghosttrace.telemetry import load_recoveries
    except Exception:  # the overlay is optional; change tracking never depends on it
        return {}
    return load_recoveries()


def apply_recoveries(removed: list[dict[str, Any]], recoveries: dict[str, dict[str, Any]] | None,
                     current_time: datetime | None) -> int:
    """Mark removed records that a field device reported recovered. Mutates `removed`; returns the count.

    recoveries: {"survey_id|detection_id": overlay record} (ghosttrace.telemetry.load_recoveries).
    A recovery counts only when reported at or before the current survey's time.
    """
    confirmed = 0
    for rec in removed:
        rec["status"] = REMOVED_STATUS
        overlay = (recoveries or {}).get(f"{rec.get('previous_survey_id')}|{rec.get('previous_detection_id')}")
        if not overlay or overlay.get("status") != "recovered":
            continue
        when = _parse_time(overlay.get("recovered_at"))
        summary = {k: overlay.get(k) for k in ("recovered_at", "device_id", "simulated", "distance_to_target_m",
                                                "flags", "basis")}
        if current_time is not None and when is not None and when > current_time:
            rec["recovery_reported_later"] = summary
            rec["basis"] = str(rec.get("basis") or "") + (
                f" A recovery was reported by device {overlay.get('device_id')} at "
                f"{overlay.get('recovered_at')}, after this survey, so it does not explain this absence.")
            continue
        rec["status"] = RECOVERED_STATUS
        rec["recovered"] = True
        rec["recovery"] = summary
        sim = " (SIMULATED DEVICE)" if overlay.get("simulated") else ""
        rec["basis"] = (f"not seen again inside this survey's coverage, and field device {overlay.get('device_id')}"
                        f"{sim} reported it recovered at {overlay.get('recovered_at')}, before this survey. "
                        "The recovery is a device report, not an independent verification.")
        confirmed += 1
    return confirmed


def compare_with_previous(survey_dir: Any, targets: list[dict[str, Any]], *,
                          surveys_root: Any = None, export: dict[str, Any] | None = None,
                          manifest: dict[str, Any] | None = None,
                          recoveries: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Change status per target, removed prior targets, and a summary.

    targets: dicts with detection_id, object_class, latitude, longitude, dimensions.
    Returns {"changes": {detection_id: change block}, "removed_since_previous": [...],
             "change_summary": {...}, "notes": [...]}.
    recoveries: the field-device recovery overlay; None loads the default
    (data/telemetry/recoveries.json), {} ignores recoveries.
    """
    survey_dir = Path(survey_dir).resolve()
    export = export if export is not None else (_load(survey_dir / "export.json") or {})
    manifest = manifest if manifest is not None else (_load(survey_dir / "manifest.json") or {})
    root = Path(surveys_root).resolve() if surveys_root else survey_dir.parent
    notes: list[str] = []

    def block(status: str, basis: str, prev_survey=None, prev_id=None, moved=None,
              prev_lat=None, prev_lon=None) -> dict:
        # previous_latitude / previous_longitude: where the matched prior target
        # was seen (persistent and moved only), so a map can draw the movement.
        return {"status": status, "previous_survey_id": prev_survey,
                "previous_detection_id": prev_id,
                "previous_latitude": None if prev_lat is None else float(prev_lat),
                "previous_longitude": None if prev_lon is None else float(prev_lon),
                "moved_m": None if moved is None else round(moved, 1), "basis": basis}

    located = [t for t in targets if t.get("latitude") is not None and t.get("longitude") is not None]
    changes: dict[str, dict[str, Any]] = {}
    for t in targets:
        if t not in located:
            changes[t["detection_id"]] = block(
                "unmatched_no_prior",
                "target has no geographic position (survey not georeferenced), so it cannot be "
                "compared with any earlier survey")

    cur_time_text, cur_time_basis = survey_time(survey_dir, export, manifest)
    cur_time = _parse_time(cur_time_text)
    cur_points = coverage_points(survey_dir, export, manifest)
    current_cov = Coverage(cur_points)
    summary = {"compared_with": [], "new": 0, "moved": 0, "persistent": 0, "removed": 0,
               "confirmed_recovered": 0}

    if not located:
        if targets:
            notes.append("no target is georeferenced; change detection was not possible")
        return {"changes": changes, "removed_since_previous": [], "change_summary": summary,
                "notes": notes}

    # Candidate prior surveys.
    priors = []
    cur_bbox = current_cov.bbox()
    if root.is_dir():
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if d.resolve() == survey_dir:
                continue
            if not ((d / cfg.OUTPUT_JSON).is_file() or (d / "export.json").is_file()):
                continue
            p_export = _load(d / "export.json") or {}
            p_manifest = _load(d / "manifest.json") or {}
            p_targets, source = prior_targets_of(d)
            p_points = coverage_points(d, p_export, p_manifest) + \
                [(t["latitude"], t["longitude"]) for t in p_targets]
            if not p_points:
                continue
            p_cov = Coverage(p_points)
            if not _bbox_overlap(cur_bbox, p_cov.bbox(), cfg.EXTENT_MARGIN_M):
                continue
            p_time_text, p_basis = survey_time(d, p_export, p_manifest)
            p_time = _parse_time(p_time_text)
            if cur_time is None or p_time is None:
                notes.append(f"survey {d.name} overlaps but its order in time is unknown; skipped")
                continue
            if not p_time < cur_time:
                continue  # same time or later: not a previous survey
            sid = (p_export.get("metadata") or {}).get("survey_id") or d.name
            priors.append({"id": sid, "time": p_time, "time_basis": p_basis, "targets": p_targets,
                           "coverage": p_cov, "source": source})

    if not priors:
        for t in located:
            changes[t["detection_id"]] = block(
                "first_survey", "no earlier processed survey with georeferenced targets overlaps "
                                "this survey's extent")
        return {"changes": changes, "removed_since_previous": [], "change_summary": summary,
                "notes": notes}

    priors.sort(key=lambda p: p["time"], reverse=True)
    pool = []
    for i, prior in enumerate(priors):
        for pt in prior["targets"]:
            if any(newer["coverage"].contains(pt["latitude"], pt["longitude"])
                   for newer in priors[:i]):
                continue  # a newer survey observed this spot; that observation wins
            pool.append({**pt, "survey_id": prior["id"], "time_basis": prior["time_basis"],
                         "source": prior["source"]})
    summary["compared_with"] = [p["id"] for p in priors]

    matches: dict[int, tuple[int, float]] = {}
    if pool:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        big = 1e9
        cost = np.full((len(located), len(pool)), big)
        dist = np.full((len(located), len(pool)), np.inf)
        for i, t in enumerate(located):
            fam = class_family(t.get("object_class"))
            area_t = _plan_area(t.get("dimensions"))
            for j, p in enumerate(pool):
                d = haversine_m(t["latitude"], t["longitude"], p["latitude"], p["longitude"])
                dist[i, j] = d
                if d > cfg.MATCH_RADIUS_M or fam != class_family(p.get("object_class")):
                    continue
                c = d
                area_p = _plan_area(p.get("dimensions"))
                if area_t and area_p:
                    ratio = max(area_t, area_p) / min(area_t, area_p)
                    if ratio > cfg.SIZE_RATIO_MAX:
                        continue
                    c += cfg.SIZE_PENALTY_M * math.log(ratio)
                cost[i, j] = c
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if cost[i, j] < big:
                matches[i] = (j, float(dist[i, j]))

    matched_pool = set()
    for i, t in enumerate(located):
        if i in matches:
            j, d = matches[i]
            p = pool[j]
            matched_pool.add(j)
            status = "persistent" if d <= cfg.STATIONARY_M else "moved"
            summary[status] += 1
            changes[t["detection_id"]] = block(
                status,
                f"matched by Hungarian assignment to {p['detection_id']} in survey "
                f"{p['survey_id']} at {d:.1f} m (gate {cfg.MATCH_RADIUS_M:g} m, stationary "
                f"<= {cfg.STATIONARY_M:g} m, same class family); this survey's time from "
                f"{cur_time_basis}; previous survey's time from {p['time_basis']}",
                p["survey_id"], p["detection_id"], d, p["latitude"], p["longitude"])
        else:
            covering = [pr["id"] for pr in priors
                        if pr["coverage"].contains(t["latitude"], t["longitude"])]
            if covering:
                summary["new"] += 1
                changes[t["detection_id"]] = block(
                    "new", f"inside the coverage of earlier survey {covering[0]} with no "
                           f"same-family target within {cfg.MATCH_RADIUS_M:g} m",
                    covering[0], None, None)
            else:
                changes[t["detection_id"]] = block(
                    "first_survey", "no earlier overlapping survey covered this position")

    removed = []
    if current_cov.hull is None:
        notes.append(f"current survey coverage could not be formed ({current_cov.reason}); "
                     "no prior target is called removed")
    else:
        for j, p in enumerate(pool):
            if j in matched_pool or not current_cov.contains(p["latitude"], p["longitude"]):
                continue
            removed.append({
                "previous_survey_id": p["survey_id"], "previous_detection_id": p["detection_id"],
                "object_class": p.get("object_class"), "latitude": p["latitude"],
                "longitude": p["longitude"],
                "basis": (f"inside this survey's coverage hull (located tiles, detections and "
                          f"track) with no same-family target within {cfg.MATCH_RADIUS_M:g} m. "
                          "It may have been recovered, buried, moved beyond the gate, or "
                          "missed by the detector this time; sonar alone cannot tell which."),
            })
    summary["removed"] = len(removed)
    overlay = _default_recoveries() if recoveries is None else recoveries
    summary["confirmed_recovered"] = apply_recoveries(removed, overlay, cur_time)
    # A prior target reported recovered but matched again here: say so rather than hide it.
    for i, t in enumerate(located):
        if i not in matches:
            continue
        p = pool[matches[i][0]]
        rec = (overlay or {}).get(f"{p['survey_id']}|{p['detection_id']}")
        if rec and rec.get("status") == "recovered":
            changes[t["detection_id"]]["recovery_reported"] = {k: rec.get(k) for k in (
                "recovered_at", "device_id", "simulated", "distance_to_target_m")}
            notes.append(f"{p['detection_id']} (survey {p['survey_id']}) was reported recovered by "
                         f"{rec.get('device_id')} but {t['detection_id']} matches it here: verify")
    return {"changes": changes, "removed_since_previous": removed, "change_summary": summary,
            "notes": notes}
