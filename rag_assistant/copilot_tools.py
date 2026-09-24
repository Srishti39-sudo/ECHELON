"""Mission Copilot data tools: deterministic, read-only lookups over data/surveys.

Every tool reads the files the survey pipeline and GhostTrace already wrote
(export.json, ghosttrace.json, and coverage.json where it exists) and returns a
compact result. Nothing is written, nothing is fetched, and no model is
involved: the same call on the same files returns the same records.

A tool returns a ToolOutput: a one-line summary for the interface ("Looked up
GhostTrace targets across 2 surveys") and a list of records. The first record
of every call is its `query_result`: what was searched, with what filters, and
how many matched. It is what an answer cites for a count, and for "nothing
matched", which is a fact too. The records after it are the matches.

Records are numbered [D1], [D2], ... by a Ledger, across every call in a turn,
in the order the calls ran. That numbering is what the answer prompt shows and
what the citation panel resolves, so the Ledger is the only thing that assigns it.

Numbers are copied from the files. The only figures a tool computes are counts
of the records it returns and, for a `near` search, the great-circle distance to
the point asked about; both are labelled as the tool's.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from backend import config


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def surveys_dir() -> Path:
    """Read at call time, so a test or a second deployment can point elsewhere."""
    return Path(os.environ.get("DEEPECHO_SURVEYS_DIR", str(config.SURVEYS_DIR)))


_CACHE: dict[str, tuple[float, int, Any]] = {}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,120}$")


def _load(path: Path) -> Any | None:
    """JSON from disk, cached on (mtime, size) so a rewritten file is re-read."""
    try:
        stat = path.stat()
    except OSError:
        return None
    key = str(path)
    cached = _CACHE.get(key)
    if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
        return cached[2]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    _CACHE[key] = (stat.st_mtime, stat.st_size, data)
    return data


def survey_ids() -> list[str]:
    root = surveys_dir()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and (p / "export.json").is_file() and _SAFE_ID.match(p.name))


def _survey_path(survey_id: str) -> Path | None:
    if not isinstance(survey_id, str) or not _SAFE_ID.match(survey_id):
        return None
    path = surveys_dir() / survey_id
    return path if (path / "export.json").is_file() else None


def load_export(survey_id: str) -> dict | None:
    path = _survey_path(survey_id)
    data = _load(path / "export.json") if path else None
    return data if isinstance(data, dict) else None


def load_ghosttrace(survey_id: str) -> dict | None:
    path = _survey_path(survey_id)
    data = _load(path / "ghosttrace.json") if path else None
    return data if isinstance(data, dict) else None


def load_coverage(survey_id: str) -> dict | None:
    """coverage.json is optional and written by another stage; read only if present."""
    path = _survey_path(survey_id)
    data = _load(path / "coverage.json") if path else None
    return data if isinstance(data, dict) else None


def is_synthetic(export: dict | None, ghosttrace: dict | None = None) -> bool:
    meta = (export or {}).get("metadata") or {}
    if meta.get("demo") is True:
        return True
    if "synthetic" in str(meta.get("data_source") or "").lower():
        return True
    return bool((ghosttrace or {}).get("synthetic_inputs") or (ghosttrace or {}).get("demo"))


def survey_title(survey_id: str) -> str | None:
    meta = (load_export(survey_id) or {}).get("metadata") or {}
    return meta.get("title")


def survey_link(survey_id: str, kind: str = "map") -> str:
    return f"/ghosttrace/{survey_id}" if kind == "ghosttrace" else f"/map?survey={survey_id}"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class ToolOutput:
    name: str
    args: dict
    summary: str
    records: list[dict] = field(default_factory=list)
    error: str | None = None
    planned_by: str = "keywords"

    def as_call(self) -> dict:
        """The tool call as the API reports it."""
        return {"name": self.name, "args": self.args, "summary": self.summary,
                "record_count": len(self.records),
                "citations": [r["n"] for r in self.records if "n" in r],
                "error": self.error, "planned_by": self.planned_by}


def _record(kind: str, survey_id: str | None, source_file: str, record_id: str | None,
            label: str, data: dict, synthetic: bool | None, link: str | None) -> dict:
    return {"kind": kind, "survey_id": survey_id, "source_file": source_file,
            "record_id": record_id, "label": label, "synthetic": synthetic,
            "link": link, "data": data}


def _query_result(tool: str, label: str, data: dict, synthetic: bool | None = None,
                  survey_id: str | None = None, link: str | None = None) -> dict:
    return _record("query_result", survey_id, "tool result", None,
                   f"{tool} result: {label}", data, synthetic, link)


class Ledger:
    """Numbers records [D1], [D2], ... across the calls of one turn."""

    def __init__(self, max_records: int | None = None):
        self.outputs: list[ToolOutput] = []
        self.max_records = max_records or config.COPILOT_MAX_DATA_RECORDS
        self.count = 0
        self.dropped = 0

    def add(self, output: ToolOutput) -> ToolOutput:
        kept = []
        for index, record in enumerate(output.records):
            # The query_result of a call is always kept: it carries the counts.
            if self.count >= self.max_records and index > 0:
                self.dropped += 1
                continue
            self.count += 1
            record["n"] = self.count
            kept.append(record)
        if len(kept) < len(output.records):
            output.summary += f" (showing {len(kept) - 1} of {len(output.records) - 1})"
        output.records = kept
        self.outputs.append(output)
        return output

    @property
    def records(self) -> list[dict]:
        return [r for o in self.outputs for r in o.records]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _singular(word: str) -> str:
    word = word.strip()
    for suffix, repl in (("ies", "y"), ("es", ""), ("s", "")):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            candidate = word[: -len(suffix)] + repl
            if suffix == "es" and not candidate.endswith(("sh", "ch", "x", "s")):
                continue
            return candidate
    return word


def class_query(object_class: str | None) -> tuple[str | None, set[str]]:
    """(family, the normalised class names a query matches).

    "mine-like objects" -> ("mine", {mine, uxo, ...}). An unrecognised word is
    matched as itself, so "cylinder" still finds cylinders.
    """
    if not object_class:
        return None, set()
    words = [_singular(w) for w in _norm(object_class).split()]
    words = [w for w in words if w not in {"object", "contact", "thing", "item", "target"}]
    phrase = " ".join(words)
    for candidate in (phrase, *words):
        family = config.COPILOT_CLASS_WORDS.get(candidate)
        if family:
            return family, {_norm(c) for c in config.COPILOT_CLASS_FAMILIES[family]}
    return None, {phrase} if phrase else set()


def _class_matches(detection: dict, wanted: set[str]) -> bool:
    if not wanted:
        return True
    names = {_norm(detection.get("object_class")), _norm(detection.get("class_normalized"))}
    names.discard("")
    for name in names:
        if name in wanted:
            return True
        # "ghost net" matches a query for "net"; "net" does not match "cabinet".
        if any(re.search(rf"\b{re.escape(w)}\b", name) for w in wanted):
            return True
    return False


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _position(item: dict, strip: str | None = None) -> dict:
    lat, lon = item.get("latitude"), item.get("longitude")
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        return {"latitude": lat, "longitude": lon}
    out: dict[str, Any] = {"latitude": None, "longitude": None, "georeferenced": False}
    if item.get("global_x") is not None:
        out["pixel_x"] = item.get("global_x")
        out["pixel_y"] = item.get("global_y")
        if strip:
            out["strip"] = strip
    return out


def _num(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _surveys_for(survey_id: Any) -> tuple[list[str], list[str]]:
    """(surveys to read, ids asked for that do not exist)."""
    known = survey_ids()
    if survey_id in (None, "", [], "all", "*"):
        return known, []
    wanted = survey_id if isinstance(survey_id, (list, tuple)) else str(survey_id).split(",")
    wanted = [str(w).strip() for w in wanted if str(w).strip()]
    return [w for w in wanted if w in known], [w for w in wanted if w not in known]


def _unknown(tool: str, args: dict, missing: list[str]) -> ToolOutput:
    known = survey_ids()
    record = _query_result(tool, "unknown survey", {
        "error": "no processed survey with this id",
        "asked_for": missing, "known_surveys": known})
    return ToolOutput(tool, args, f"No processed survey named {', '.join(missing)}",
                      [record], error="unknown survey id")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def list_surveys() -> ToolOutput:
    ids = survey_ids()
    records = [_query_result("list_surveys", "processed surveys", {
        "surveys_dir": "data/surveys", "survey_count": len(ids), "survey_ids": ids})]
    for sid in ids:
        export = load_export(sid) or {}
        gt = load_ghosttrace(sid)
        meta = export.get("metadata") or {}
        summary = export.get("survey_summary") or {}
        data = {
            "survey_id": sid,
            "title": meta.get("title"),
            "synthetic": is_synthetic(export, gt),
            "coordinate_mode": summary.get("coordinate_mode") or meta.get("coordinate_mode"),
            "detector": meta.get("detector"),
            "processed_at": meta.get("processed_at"),
            "detections": summary.get("total_deduplicated_detections"),
            "filtered_as_false_positive": summary.get("suppressed_detections"),
            "hotspots": summary.get("total_hotspots"),
            "ghosttrace": gt is not None,
            "ghosttrace_targets": (gt or {}).get("summary", {}).get("targets") if gt else None,
            "compared_with": ((gt or {}).get("change_summary") or {}).get("compared_with") if gt else None,
        }
        records.append(_record("survey", sid, f"export.json of {sid}", sid,
                               f"export.json of {sid}, survey metadata", data,
                               data["synthetic"], survey_link(sid)))
    return ToolOutput("list_surveys", {}, f"Listed {_plural(len(ids), 'processed survey')}", records)


def survey_summary(survey_id: str) -> ToolOutput:
    args = {"survey_id": survey_id}
    surveys, missing = _surveys_for(survey_id)
    if missing or len(surveys) != 1:
        return _unknown("survey_summary", args, missing or [str(survey_id)])
    sid = surveys[0]
    export = load_export(sid) or {}
    gt = load_ghosttrace(sid)
    meta = export.get("metadata") or {}
    s = export.get("survey_summary") or {}
    top = s.get("highest_priority_hotspot") or {}
    synthetic = is_synthetic(export, gt)
    data: dict[str, Any] = {
        "survey_id": sid,
        "title": meta.get("title"),
        "synthetic": synthetic,
        "demo_warning": meta.get("demo_warning"),
        "detector": meta.get("detector"),
        "processed_at": meta.get("processed_at"),
        "coordinate_mode": s.get("coordinate_mode"),
        "georeferenced": s.get("georeferenced"),
        "strips_processed": s.get("strips_processed"),
        "tiles_processed": s.get("tiles_processed"),
        "raw_detections": s.get("total_raw_detections"),
        "detections_after_deduplication": s.get("total_deduplicated_detections"),
        "duplicates_merged": s.get("duplicates_removed"),
        "filtered_as_false_positive": s.get("suppressed_detections"),
        "class_distribution": s.get("class_distribution"),
        "detections_by_tier": s.get("detections_by_tier"),
        "hotspots": s.get("total_hotspots"),
        "hotspots_by_tier": s.get("hotspots_by_tier"),
        "highest_priority_hotspot": {
            "hotspot_id": top.get("hotspot_id"), "dominant_class": top.get("dominant_class"),
            "risk_score": top.get("risk_score"), "detection_count": top.get("detection_count"),
            "recommended_action": top.get("recommended_action")} if top else None,
        "severity_note": "severity = class_weight x confidence, a configurable heuristic",
    }
    if gt:
        gsum = gt.get("summary") or {}
        data["ghosttrace"] = {
            "targets": gsum.get("targets"), "urgent": gsum.get("urgent"), "high": gsum.get("high"),
            "propeller_hazards": gsum.get("propeller_hazards"),
            "change_summary": gt.get("change_summary")}
    coverage = load_coverage(sid)
    if coverage:
        data["coverage"] = {k: v for k, v in list(coverage.items())[:40]
                            if isinstance(v, (str, int, float, bool)) or v is None}
    records = [
        _query_result("survey_summary", sid, {"survey_id": sid, "found": True}, synthetic, sid),
        _record("survey_summary", sid, f"export.json of {sid}", sid,
                f"export.json of {sid}, survey_summary", data, synthetic, survey_link(sid)),
    ]
    return ToolOutput("survey_summary", args, f"Read the survey summary of {sid}", records)


def find_detections(survey_id: str | None = None, object_class: str | None = None,
                    min_confidence: float | None = None, include_filtered: bool = False,
                    tier: str | None = None, near: Any = None, limit: int = 20) -> ToolOutput:
    args = {k: v for k, v in {"survey_id": survey_id, "object_class": object_class,
                              "min_confidence": min_confidence,
                              "include_filtered": include_filtered or None,
                              "tier": tier, "near": near, "limit": limit}.items()
            if v not in (None, "")}
    surveys, missing = _surveys_for(survey_id)
    if missing:
        return _unknown("find_detections", args, missing)
    family, wanted = class_query(object_class)
    floor = _num(min_confidence)
    if floor is not None and floor > 1:
        floor = floor / 100.0
    tier_wanted = _norm(tier) or None
    point = _parse_near(near)
    limit = max(1, min(int(_num(limit, 20) or 20), 50))

    matched: list[tuple[dict, str, dict]] = []
    excluded_filtered = 0
    unlocated = 0
    by_survey: dict[str, int] = {}
    related: list[dict] = []
    any_synthetic = False
    for sid in surveys:
        export = load_export(sid) or {}
        synthetic = is_synthetic(export)
        any_synthetic = any_synthetic or synthetic
        by_survey.setdefault(sid, 0)
        for det in export.get("detections") or []:
            if not _class_matches(det, wanted):
                if family == "mine" and re.search(r"ordnance|\beod\b|explosive",
                                                  str(det.get("recommended_action") or ""), re.I):
                    related.append({"survey_id": sid, "detection_id": det.get("id"),
                                    "object_class": det.get("object_class"),
                                    "recommended_action": det.get("recommended_action"),
                                    "filtered_as_false_positive": bool(det.get("suppressed"))})
                continue
            if floor is not None and (_num(det.get("confidence"), -1) or -1) < floor:
                continue
            if tier_wanted and _norm(det.get("severity_tier")) != tier_wanted:
                continue
            extra: dict[str, Any] = {}
            if point:
                lat, lon = det.get("latitude"), det.get("longitude")
                if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
                    unlocated += 1
                    continue
                distance = _haversine_m(point[0], point[1], lat, lon)
                if distance > point[2]:
                    continue
                extra["distance_m_computed_by_tool"] = round(distance, 1)
            if det.get("suppressed") and not include_filtered:
                excluded_filtered += 1
                continue
            by_survey[sid] += 1
            matched.append((det, sid, extra))

    matched.sort(key=lambda m: (-(_num(m[0].get("severity"), 0) or 0),
                                -(_num(m[0].get("confidence"), 0) or 0), str(m[0].get("id"))))
    shown = matched[:limit]
    query: dict[str, Any] = {
        "surveys_searched": surveys,
        "filters": {"object_class": object_class, "class_family": family,
                    "classes_matched": sorted(wanted) if wanted else "any",
                    "min_detector_confidence": floor, "severity_tier": tier,
                    "near": {"latitude": point[0], "longitude": point[1], "radius_m": point[2]}
                    if point else None,
                    "include_filtered": bool(include_filtered)},
        "matched": len(matched),
        "returned": len(shown),
        "matched_by_survey": by_survey,
        "filtered_false_positives_excluded": excluded_filtered,
        "order": "severity descending, then detector confidence",
    }
    if point:
        query["not_georeferenced_skipped"] = unlocated
    if family == "mine":
        query["note"] = ("mine family = detector classes " + ", ".join(sorted(wanted)) +
                         "; a class is detector output, not a confirmed identification")
        if related:
            query["other_detections_whose_recommended_action_mentions_ordnance"] = related[:10]
    records = [_query_result("find_detections", _describe_filters(object_class, surveys, tier),
                             query, any_synthetic or None)]
    for det, sid, extra in shown:
        export = load_export(sid) or {}
        strip = _strip_of(export, det)
        data = {
            "survey_id": sid,
            "detection_id": det.get("id"),
            "object_class": det.get("object_class"),
            "detector_confidence": det.get("confidence"),
            "confidence_pct": det.get("confidence_pct"),
            "severity": det.get("severity"),
            "severity_tier": det.get("severity_tier"),
            "recommended_action": det.get("recommended_action"),
            **_position(det, strip),
            "filtered_as_false_positive": bool(det.get("suppressed")),
            "synthetic": is_synthetic(export),
            **extra,
        }
        dims = det.get("dimensions") or {}
        if dims.get("length_m") is not None:
            data["length_m"] = dims.get("length_m")
            data["width_m"] = dims.get("width_m")
        records.append(_record("detection", sid, f"export.json of {sid}", det.get("id"),
                               f"export.json of {sid}, detection {det.get('id')}", data,
                               data["synthetic"], survey_link(sid)))
    where = "all surveys" if survey_id in (None, "") else ", ".join(surveys)
    what = object_class or "detections"
    summary = f"Searched {what} in {where}: {len(matched)} matched"
    return ToolOutput("find_detections", args, summary, records)


def _describe_filters(object_class: str | None, surveys: list[str], tier: str | None) -> str:
    parts = [object_class or "all classes"]
    if tier:
        parts.append(f"tier {tier}")
    parts.append("in " + (", ".join(surveys) if len(surveys) <= 2 else f"{len(surveys)} surveys"))
    return " ".join(parts)


def _strip_of(export: dict, det: dict) -> str | None:
    strips = (export.get("survey_summary") or {}).get("strips") or []
    ident = str(det.get("id") or "")
    for strip in strips:
        if ident.startswith(str(strip)):
            return strip
    return strips[0] if len(strips) == 1 else None


def _parse_near(near: Any) -> tuple[float, float, float] | None:
    if near in (None, "", [], {}):
        return None
    if isinstance(near, dict):
        values = [near.get("latitude", near.get("lat")), near.get("longitude", near.get("lon")),
                  near.get("radius_m", near.get("radius"))]
    elif isinstance(near, (list, tuple)):
        values = list(near) + [None] * (3 - len(near))
    else:
        return None
    lat, lon, radius = (_num(v) for v in values[:3])
    if lat is None or lon is None or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return None
    return (lat, lon, radius if radius and radius > 0 else 1000.0)


def top_hotspots(survey_id: str | None = None, n: int = 5) -> ToolOutput:
    args = {k: v for k, v in {"survey_id": survey_id, "n": n}.items() if v not in (None, "")}
    surveys, missing = _surveys_for(survey_id)
    if missing:
        return _unknown("top_hotspots", args, missing)
    n = max(1, min(int(_num(n, 5) or 5), 20))
    rows: list[tuple[dict, str, bool]] = []
    for sid in surveys:
        export = load_export(sid) or {}
        for hotspot in export.get("hotspots") or []:
            rows.append((hotspot, sid, is_synthetic(export)))
    rows.sort(key=lambda r: (-(_num(r[0].get("risk_score"), 0) or 0), r[1],
                             int(_num(r[0].get("priority_rank"), 0) or 0)))
    shown = rows[:n]
    query = {"surveys_searched": surveys, "hotspots_total": len(rows), "returned": len(shown),
             "order": ("risk_score descending; priority_rank is each survey's own ranking"
                       if len(surveys) > 1 else "risk_score descending (the survey's priority_rank)"),
             "risk_score_formula": {sid: ((load_export(sid) or {}).get("configuration") or {})
                                    .get("hotspots", {}).get("risk_score") for sid in surveys},
             "heuristic": "severity and risk scores are configurable heuristics"}
    records = [_query_result("top_hotspots", "hotspots in " + (
        ", ".join(surveys) if len(surveys) <= 2 else f"{len(surveys)} surveys"), query,
        any(r[2] for r in shown) or None)]
    for hotspot, sid, synthetic in shown:
        centroid = hotspot.get("centroid") or {}
        data = {
            "survey_id": sid,
            "hotspot_id": hotspot.get("hotspot_id"),
            "priority_rank_in_survey": hotspot.get("priority_rank"),
            "dominant_class": hotspot.get("dominant_class"),
            "severity_tier": hotspot.get("severity_tier"),
            "risk_score": hotspot.get("risk_score"),
            "max_severity": hotspot.get("max_severity"),
            "detection_count": hotspot.get("detection_count"),
            "recommended_action": hotspot.get("recommended_action"),
            **_position({"latitude": centroid.get("latitude"), "longitude": centroid.get("longitude"),
                         "global_x": centroid.get("global_x"), "global_y": centroid.get("global_y")},
                        hotspot.get("strip")),
            "rationale": hotspot.get("rationale"),
            "synthetic": synthetic,
        }
        records.append(_record("hotspot", sid, f"export.json of {sid}", hotspot.get("hotspot_id"),
                               f"export.json of {sid}, hotspot {hotspot.get('hotspot_id')}",
                               data, synthetic, survey_link(sid)))
    where = "all surveys" if survey_id in (None, "") else ", ".join(surveys)
    return ToolOutput("top_hotspots", args,
                      f"Ranked hotspots in {where}: top {len(shown)} of {len(rows)}", records)


_TIER_ORDER = {"urgent": 0, "high": 1, "routine": 2}


def _supersession(ids: list[str]) -> dict[tuple[str, str], dict]:
    """(survey, detection) -> what a later survey says became of it.

    A later GhostTrace run that matched a target names it as its
    previous_detection_id; one that looked and did not see it lists it in
    removed_since_previous. Either way the older record is no longer the
    latest observation of that object.
    """
    out: dict[tuple[str, str], dict] = {}
    for sid in ids:
        gt = load_ghosttrace(sid)
        if not gt:
            continue
        for target in gt.get("targets") or []:
            change = target.get("change") or {}
            prev_s, prev_d = change.get("previous_survey_id"), change.get("previous_detection_id")
            if prev_s and prev_d:
                out[(prev_s, prev_d)] = {"survey_id": sid, "detection_id": target.get("detection_id"),
                                         "status": change.get("status"), "moved_m": change.get("moved_m")}
        for removed in gt.get("removed_since_previous") or []:
            key = (removed.get("previous_survey_id"), removed.get("previous_detection_id"))
            if all(key):
                out.setdefault(key, {"survey_id": sid, "detection_id": None,
                                     "status": "not seen again (removed_since_previous)",
                                     "moved_m": None})
    return out


def ghosttrace_targets(survey_id: str | None = None, tier: str | None = None,
                       limit: int = 10) -> ToolOutput:
    args = {k: v for k, v in {"survey_id": survey_id, "tier": tier, "limit": limit}.items()
            if v not in (None, "")}
    surveys, missing = _surveys_for(survey_id)
    if missing:
        return _unknown("ghosttrace_targets", args, missing)
    limit = max(1, min(int(_num(limit, 10) or 10), 30))
    with_gt = [sid for sid in surveys if load_ghosttrace(sid)]
    superseded = _supersession(survey_ids())
    tier_wanted = _norm(tier) or None

    rows: list[dict] = []
    for sid in with_gt:
        gt = load_ghosttrace(sid) or {}
        export = load_export(sid) or {}
        synthetic = is_synthetic(export, gt)
        order = (gt.get("recovery_plan") or {}).get("order") or []
        for target in gt.get("targets") or []:
            priority = target.get("priority") or {}
            if tier_wanted and _norm(priority.get("tier")) != tier_wanted:
                continue
            detection_id = target.get("detection_id")
            later = superseded.get((sid, detection_id))
            activity = target.get("activity") or {}
            habitat = target.get("habitat") or {}
            nearest = (habitat.get("nearest") or [None])[0] if habitat.get("available") is not False else None
            people = target.get("people") or {}
            change = target.get("change") or {}
            terms = priority.get("terms") or {}
            data = {
                "survey_id": sid,
                "survey_title": (export.get("metadata") or {}).get("title"),
                "synthetic": synthetic,
                "detection_id": detection_id,
                "object_class": target.get("object_class"),
                "priority_score": priority.get("score"),
                "priority_tier": priority.get("tier"),
                "rank_in_survey": priority.get("rank"),
                "confidence_pct": target.get("confidence_pct"),
                "latitude": target.get("latitude"),
                "longitude": target.get("longitude"),
                "seabed_depth_m": target.get("seabed_depth_m"),
                "activity_level": activity.get("level"),
                "nearest_habitat": {"name": nearest.get("name"), "kind": nearest.get("kind"),
                                    "distance_m": nearest.get("distance_m")} if isinstance(nearest, dict) else None,
                "propeller_hazard": (people.get("propeller_hazard") or {}).get("level")
                if people.get("available") is not False else None,
                "change_status": change.get("status"),
                "moved_m": change.get("moved_m"),
                "previous_survey_id": change.get("previous_survey_id"),
                "previous_detection_id": change.get("previous_detection_id"),
                "priority_terms_contribution": {name: (term or {}).get("contribution")
                                                for name, term in terms.items()
                                                if (term or {}).get("contribution") is not None},
                "confidence_multiplier": (terms.get("confidence") or {}).get("value"),
                "recovery_plan_position": (order.index(detection_id) + 1) if detection_id in order else None,
                "authorities_ghosttrace_lists": [
                    {"name": a.get("name"), "situation": a.get("situation")}
                    for a in (target.get("alert") or {}).get("authorities") or []],
                "latest_observation": later is None,
                "superseded_by": later,
            }
            rows.append(data)

    def key(row: dict) -> tuple:
        return (0 if row["latest_observation"] else 1,
                -(_num(row["priority_score"], 0) or 0),
                _TIER_ORDER.get(str(row["priority_tier"]), 9), row["survey_id"], str(row["detection_id"]))

    rows.sort(key=key)
    current = [r for r in rows if r["latest_observation"]]
    for position, row in enumerate(current, start=1):
        row["cross_survey_rank"] = position
    for row in rows:
        row.setdefault("cross_survey_rank", None)
    shown = rows[:limit]
    query = {
        "surveys_searched": surveys,
        "surveys_with_ghosttrace": with_gt,
        "surveys_without_ghosttrace": [s for s in surveys if s not in with_gt],
        "tier_filter": tier,
        "targets_total": len(rows),
        "latest_observations": len(current),
        "superseded_by_a_later_survey": len(rows) - len(current),
        "returned": len(shown),
        "ranking": ("latest observation of each object first (a target a later survey matched or "
                    "no longer saw is superseded), then GhostTrace priority_score descending"),
        "top_target": ({"survey_id": current[0]["survey_id"], "detection_id": current[0]["detection_id"],
                        "priority_score": current[0]["priority_score"],
                        "priority_tier": current[0]["priority_tier"]} if current else None),
        "heuristic": "GhostTrace priority weights and tiers are configurable heuristics, not official procedure",
    }
    synthetic_any = any(r["synthetic"] for r in shown) or None
    records = [_query_result("ghosttrace_targets", "GhostTrace targets across " + (
        _plural(len(with_gt), "survey")), query, synthetic_any)]
    for row in shown:
        records.append(_record("ghosttrace_target", row["survey_id"],
                               f"ghosttrace.json of {row['survey_id']}", row["detection_id"],
                               f"ghosttrace.json of {row['survey_id']}, target {row['detection_id']}",
                               row, row["synthetic"], survey_link(row["survey_id"], "ghosttrace")))
    return ToolOutput("ghosttrace_targets", args,
                      f"Looked up GhostTrace targets across {_plural(len(with_gt), 'survey')} "
                      f"({_plural(len(rows), 'target')})", records)


def change_report(survey_id: str | None = None) -> ToolOutput:
    args = {"survey_id": survey_id} if survey_id else {}
    surveys, missing = _surveys_for(survey_id)
    if missing:
        return _unknown("change_report", args, missing)
    reports: list[str] = []
    notes: list[str] = []
    for sid in surveys:
        gt = load_ghosttrace(sid)
        if not gt:
            if survey_id:
                notes.append(f"{sid} has no ghosttrace.json; change detection runs in GhostTrace "
                             "and has not run for this survey")
            continue
        compared = (gt.get("change_summary") or {}).get("compared_with") or []
        if compared:
            reports.append(sid)
        elif survey_id:
            later = [other for other in survey_ids()
                     if sid in (((load_ghosttrace(other) or {}).get("change_summary") or {})
                                .get("compared_with") or [])]
            if later:
                notes.append(f"{sid} is the earlier survey; {', '.join(later)} compared itself with it")
                reports.extend(s for s in later if s not in reports)
            else:
                notes.append(f"{sid} was not compared with any earlier survey (change status first_survey)")
    records: list[dict] = []
    totals = {"new": 0, "moved": 0, "persistent": 0, "removed": 0}
    items: list[dict] = []
    synthetic_any = False
    for sid in reports:
        gt = load_ghosttrace(sid) or {}
        export = load_export(sid) or {}
        synthetic = is_synthetic(export, gt)
        synthetic_any = synthetic_any or synthetic
        cs = gt.get("change_summary") or {}
        for k in totals:
            totals[k] += int(_num(cs.get(k), 0) or 0)
        items.append(_record("change_summary", sid, f"ghosttrace.json of {sid}", None,
                             f"ghosttrace.json of {sid}, change_summary",
                             {"survey_id": sid, "compared_with": cs.get("compared_with"),
                              "new": cs.get("new"), "moved": cs.get("moved"),
                              "persistent": cs.get("persistent"), "removed": cs.get("removed"),
                              "synthetic": synthetic},
                             synthetic, survey_link(sid, "ghosttrace")))
        for target in gt.get("targets") or []:
            change = target.get("change") or {}
            if change.get("status") not in ("new", "moved", "persistent"):
                continue
            items.append(_record("change", sid, f"ghosttrace.json of {sid}", target.get("detection_id"),
                                 f"ghosttrace.json of {sid}, change of {target.get('detection_id')}", {
                                     "survey_id": sid, "detection_id": target.get("detection_id"),
                                     "object_class": target.get("object_class"),
                                     "status": change.get("status"), "moved_m": change.get("moved_m"),
                                     "previous_survey_id": change.get("previous_survey_id"),
                                     "previous_detection_id": change.get("previous_detection_id"),
                                     "latitude": target.get("latitude"), "longitude": target.get("longitude"),
                                     "previous_latitude": change.get("previous_latitude"),
                                     "previous_longitude": change.get("previous_longitude"),
                                     "basis": change.get("basis"), "synthetic": synthetic},
                                 synthetic, survey_link(sid, "ghosttrace")))
        for removed in gt.get("removed_since_previous") or []:
            items.append(_record("removed", sid, f"ghosttrace.json of {sid}",
                                 removed.get("previous_detection_id"),
                                 f"ghosttrace.json of {sid}, removed_since_previous "
                                 f"{removed.get('previous_detection_id')}", {
                                     "survey_id": sid, "status": "removed",
                                     "previous_survey_id": removed.get("previous_survey_id"),
                                     "previous_detection_id": removed.get("previous_detection_id"),
                                     "object_class": removed.get("object_class"),
                                     "latitude": removed.get("latitude"), "longitude": removed.get("longitude"),
                                     "basis": removed.get("basis"), "synthetic": synthetic},
                                 synthetic, survey_link(sid, "ghosttrace")))
    query = {"surveys_asked": surveys if survey_id else "all",
             "surveys_with_a_comparison": reports, **({"totals": totals} if len(reports) > 1 else {}),
             "notes": notes,
             "status_meanings": {"new": "not matched, inside an earlier survey's coverage",
                                 "moved": "matched, displaced more than 20 m and within the 75 m gate",
                                 "persistent": "matched within positional uncertainty",
                                 "removed": "an earlier target inside this coverage not seen again; "
                                            "recovered, buried, moved beyond the gate or missed"}}
    records.append(_query_result("change_report", ", ".join(reports) or "no comparison", query,
                                 synthetic_any or None))
    records.extend(items)
    if reports:
        summary = f"Compared surveys: {', '.join(reports)}"
    else:
        summary = "No survey comparison found"
    return ToolOutput("change_report", args, summary, records)


def filtered_detections(survey_id: str | None = None) -> ToolOutput:
    args = {"survey_id": survey_id} if survey_id else {}
    surveys, missing = _surveys_for(survey_id)
    if missing:
        return _unknown("filtered_detections", args, missing)
    items: list[dict] = []
    per_survey: dict[str, dict] = {}
    synthetic_any = False
    for sid in surveys:
        export = load_export(sid) or {}
        synthetic = is_synthetic(export)
        s = export.get("survey_summary") or {}
        synthetic_any = synthetic_any or synthetic
        count = 0
        for det in export.get("detections") or []:
            if not det.get("suppressed"):
                continue
            count += 1
            verification = det.get("verification") or {}
            data = {
                "survey_id": sid,
                "detection_id": det.get("id"),
                "object_class": det.get("object_class"),
                "detector_confidence": det.get("confidence"),
                "verified_confidence_pct": verification.get("confidence_pct", det.get("confidence_pct")),
                "hard_reasons": verification.get("hard_reasons"),
                "reasons": [str(r)[:260] for r in (verification.get("reasons") or [])][:4],
                "suppress_rule": verification.get("suppress_rule"),
                **_position(det, _strip_of(export, det)),
                "kept_in_export": True,
                "synthetic": synthetic,
            }
            items.append(_record("filtered_detection", sid, f"export.json of {sid}", det.get("id"),
                                 f"export.json of {sid}, filtered detection {det.get('id')}",
                                 data, synthetic, survey_link(sid)))
        per_survey[sid] = {"filtered_as_false_positive": count,
                           "duplicates_merged_not_false_positives": s.get("duplicates_removed")}
    query = {"surveys_searched": surveys, "filtered_total": len(items), "by_survey": per_survey,
             "note": ("a filtered detection is flagged suppressed by image verification and kept in "
                      "export.json, never deleted; merged duplicates are a different step")}
    records = [_query_result("filtered_detections", "filtered detections in " + (
        ", ".join(surveys) if len(surveys) <= 2 else f"{len(surveys)} surveys"), query,
        synthetic_any or None)]
    records.extend(items)
    where = "all surveys" if not survey_id else ", ".join(surveys)
    return ToolOutput("filtered_detections", args,
                      f"Checked filtered false positives in {where}: {len(items)} found", records)


# ---------------------------------------------------------------------------
# Registry: what the planner may call, and with which arguments
# ---------------------------------------------------------------------------

_SURVEY_ARG = {"type": "string", "description": "survey id exactly as listed; empty for every survey"}

TOOLS: dict[str, dict] = {
    "list_surveys": {
        "fn": list_surveys,
        "description": "List every processed survey with its title, synthetic flag, detection, "
                       "hotspot and GhostTrace counts.",
        "parameters": {"type": "object", "properties": {}},
    },
    "survey_summary": {
        "fn": survey_summary,
        "description": "Summary of one survey: detections by class and tier, hotspots, filtered "
                       "false positives, GhostTrace counts and change summary.",
        "parameters": {"type": "object", "properties": {"survey_id": {
            "type": "string", "description": "survey id exactly as listed"}},
            "required": ["survey_id"]},
    },
    "find_detections": {
        "fn": find_detections,
        "description": "Find detections by class (e.g. mine, net, shipwreck, cylinder), minimum "
                       "detector confidence, severity tier or position, in one survey or all.",
        "parameters": {"type": "object", "properties": {
            "survey_id": _SURVEY_ARG,
            "object_class": {"type": "string", "description": "class or family, e.g. 'mine', 'net'"},
            "min_confidence": {"type": "number", "description": "detector confidence 0 to 1"},
            "include_filtered": {"type": "boolean", "description": "include filtered false positives"},
            "tier": {"type": "string", "description": "critical, medium or low"},
            "near": {"type": "object", "description": "search around a position", "properties": {
                "latitude": {"type": "number"}, "longitude": {"type": "number"},
                "radius_m": {"type": "number"}}},
            "limit": {"type": "integer", "description": "maximum records, default 20"}}},
    },
    "top_hotspots": {
        "fn": top_hotspots,
        "description": "Highest-risk hotspots (grid cells of detections), in one survey or all.",
        "parameters": {"type": "object", "properties": {
            "survey_id": _SURVEY_ARG,
            "n": {"type": "integer", "description": "how many, default 5"}}},
    },
    "ghosttrace_targets": {
        "fn": ghosttrace_targets,
        "description": "GhostTrace ghost-net targets with priority, tier, change status and the "
                       "authorities GhostTrace lists. Without survey_id, ranks the latest observation "
                       "of every net across all surveys by priority: use it for 'which net first'.",
        "parameters": {"type": "object", "properties": {
            "survey_id": _SURVEY_ARG,
            "tier": {"type": "string", "description": "urgent, high or routine"},
            "limit": {"type": "integer", "description": "maximum records, default 10"}}},
    },
    "change_report": {
        "fn": change_report,
        "description": "What changed between a survey and the earlier survey it was compared with: "
                       "new, moved, persistent and removed targets.",
        "parameters": {"type": "object", "properties": {"survey_id": _SURVEY_ARG}},
    },
    "filtered_detections": {
        "fn": filtered_detections,
        "description": "Detections filtered as likely false positives by image verification, with "
                       "the stated reasons.",
        "parameters": {"type": "object", "properties": {"survey_id": _SURVEY_ARG}},
    },
}


def tool_specs() -> list[dict]:
    """Provider-neutral function declarations (name, description, JSON schema)."""
    return [{"name": name, "description": spec["description"], "parameters": spec["parameters"]}
            for name, spec in TOOLS.items()]


def _coerce(name: str, args: Any) -> dict:
    """Only the declared arguments, with the declared types. Anything else is dropped."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    if not isinstance(args, dict):
        return {}
    props = TOOLS[name]["parameters"].get("properties", {})
    out: dict[str, Any] = {}
    for key, value in args.items():
        if key not in props or value is None or value == "":
            continue
        kind = props[key].get("type")
        if kind == "string":
            out[key] = str(value).strip()
        elif kind == "number":
            number = _num(value)
            if number is not None:
                out[key] = number
        elif kind == "integer":
            number = _num(value)
            if number is not None:
                out[key] = int(number)
        elif kind == "boolean":
            out[key] = value if isinstance(value, bool) else str(value).lower() in {"1", "true", "yes"}
        elif kind == "object":
            if isinstance(value, dict):
                out[key] = value
    return out


def run_tool(name: str, args: Any = None, planned_by: str = "keywords") -> ToolOutput:
    """Run one tool by name. Never raises: a failure is a record saying so."""
    if name not in TOOLS:
        return ToolOutput(name, {}, f"Unknown tool {name}", [
            _query_result(name, "unknown tool", {"error": "no such tool", "tools": list(TOOLS)})],
            error="unknown tool", planned_by=planned_by)
    clean = _coerce(name, args or {})
    fn: Callable[..., ToolOutput] = TOOLS[name]["fn"]
    try:
        output = fn(**clean)
    except TypeError as exc:
        output = ToolOutput(name, clean, f"{name} could not run", [
            _query_result(name, "bad arguments", {"error": f"bad arguments: {exc}"})],
            error="bad arguments")
    except Exception as exc:  # a malformed file must not take the answer down
        output = ToolOutput(name, clean, f"{name} failed", [
            _query_result(name, "failed", {"error": f"{type(exc).__name__}: {exc}"[:300]})],
            error=type(exc).__name__)
    output.planned_by = planned_by
    return output


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def compact(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)


def render_data_block(records: list[dict]) -> str:
    """The DATA block of the answer prompt: one tagged record per line pair."""
    if not records:
        return "DATA\n====\n(no survey records)"
    lines = ["DATA\n===="]
    for record in records:
        synthetic = record.get("synthetic")
        flag = "YES" if synthetic else ("no" if synthetic is False else "n/a")
        lines.append(f"[D{record['n']}] {record['label']} | kind: {record['kind']} | synthetic: {flag}")
        lines.append(compact(record["data"]))
    return "\n".join(lines)


def data_citations(records: list[dict]) -> list[dict]:
    """The records as the citation panel receives them."""
    return [{"n": r["n"], "kind": r["kind"], "survey_id": r.get("survey_id"),
             "label": r["label"], "source_file": r["source_file"], "record_id": r.get("record_id"),
             "synthetic": r.get("synthetic"), "link": r.get("link"), "summary": r["data"]}
            for r in records]
