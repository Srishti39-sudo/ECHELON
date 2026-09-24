"""GhostTrace: from a processed survey to rescue decisions, in one call.

    from ghosttrace.engine import run_ghosttrace
    result = run_ghosttrace("data/surveys/<survey>")

reads export.json (+ manifest.json and any navigation / water-column sidecars)
from a processed survey directory and writes ghosttrace.json and
ghosttrace.geojson beside it. The output contract is documented in
ghosttrace/schema.md; the frontend codes against it.

PIPELINE, PER TARGET
    select    detections whose class is a GhostTrace class (nets, gear, rope,
              debris, tyres, drums, containers); suppressed detections are
              excluded by default and counted
    activity  watercolumn.activity_evidence       (this package, core)
    habitat   habitat.habitat_context              (geo agent)
    drift     drift.drift_forecast                 (geo agent)
    people    safety.people_safety                 (geo agent)
    change    changes.compare_with_previous        (core, all targets at once)
    priority  priority.score_target / rank_targets (core)
    alert     alerts.draft_alert                   (core)
    recovery  recovery.plan_route                  (core, all targets at once)

GRACEFUL DEGRADATION
    The geo stages are written by another agent and may not exist yet, may
    fail to import, or may raise on one target. Each is reached through an
    adapter: an ImportError, a missing function or an exception becomes
    {"available": false, "reason": ...} for that stage and that target, the
    failure is recorded in run.failures, and the run continues. A crash in
    one target never loses the others. Tests inject fakes through `stages`.

DRIFT MODE
    `seabed` by default: side-scan images the seabed and a snagged net is the
    common case. `floating` only when the class carries a floating/midwater
    hint, verification says midwater, or dimensions.height_m >=
    MIDWATER_HEIGHT_M (weak proxy, labelled in drift.mode_basis).

DRIFT START TIME
    The ping time of the detection's row from the navigation sidecar when
    recorded, else the earliest sidecar time, else export processed_at - which
    is when the survey was processed, not recorded, and the caveats say so.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ghosttrace import config_core as cfg
from ghosttrace import alerts as alerts_mod
from ghosttrace import changes as changes_mod
from ghosttrace import priority as priority_mod
from ghosttrace import recovery as recovery_mod
from ghosttrace import watercolumn as wc_mod

EventCallback = Callable[[dict[str, Any]], None]


# --- adapters ---------------------------------------------------------------------

def _import_attr(module: str, names: tuple[str, ...]) -> tuple[Any, str | None]:
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # ImportError, or the module failing at import time
        return None, f"{module} not importable ({type(exc).__name__}: {str(exc)[:160]})"
    for name in names:
        if hasattr(mod, name):
            return getattr(mod, name), None
    return None, f"{module} has none of {', '.join(names)}"


def _trace() -> str:
    """The tail of the current traceback with filesystem paths reduced to file names.

    ghosttrace.json travels (attached to reports, served to a browser), so it
    must not carry anyone's home directory.
    """
    import re
    text = traceback.format_exc(limit=3)
    text = re.sub(r'File "[^"]*[/\\]([^/\\"]+)"', r'File "\1"', text)
    return text[-600:]


def _unavailable(reason: str, **extra: Any) -> dict[str, Any]:
    return {"available": False, "reason": reason, **extra}


class Stages:
    """Resolved stage callables and shared inputs, with a reason for each gap."""

    def __init__(self, overrides: dict[str, Any] | None = None):
        o = dict(overrides or {})
        self.status: dict[str, dict[str, Any]] = {}

        def fn(key: str, module: str, names: tuple[str, ...]):
            if key in o:
                value = o[key]
                self.status[key] = {"available": value is not None, "source": "injected",
                                    "reason": None if value is not None else "disabled by caller"}
                return value
            value, reason = _import_attr(module, names)
            self.status[key] = {"available": value is not None, "source": module, "reason": reason}
            return value

        def loaded(key: str, module: str, names: tuple[str, ...]):
            if key in o:
                value = o[key]
                value = value() if callable(value) else value
                self.status[key] = {"available": value is not None, "source": "injected",
                                    "reason": None if value is not None else "disabled by caller"}
                return value
            loader, reason = _import_attr(module, names)
            if loader is None:
                self.status[key] = {"available": False, "source": module, "reason": reason}
                return None
            try:
                value = loader() if callable(loader) else loader
            except Exception as exc:
                self.status[key] = {"available": False, "source": module,
                                    "reason": f"loader failed ({type(exc).__name__}: {str(exc)[:160]})"}
                return None
            self.status[key] = {"available": value is not None, "source": module,
                                "reason": None if value is not None else "loader returned nothing"}
            return value

        self.habitat = fn("habitat", "ghosttrace.habitat", ("habitat_context",))
        self.drift = fn("drift", "ghosttrace.drift", ("drift_forecast",))
        self.people = fn("people", "ghosttrace.safety", ("people_safety",))
        self.field = loaded("field", "ghosttrace.currents",
                            ("load_default_field", "default_field", "load_field",
                             "load_bundled_field", "get_default_field"))
        self.layers = loaded("layers", "ghosttrace.layers",
                             ("load_default_layers", "load_layers", "default_layers", "get_layers"))
        self.bathymetry = loaded("bathymetry", "ghosttrace.layers",
                                 ("load_bathymetry", "default_bathymetry", "get_bathymetry"))
        if "harbours" in o:
            self.harbours = o["harbours"]
            self.status["harbours"] = {"available": bool(self.harbours), "source": "injected",
                                       "reason": None if self.harbours else "none supplied"}
        else:
            self.harbours = _harbours_from(self.layers)
            self.status["harbours"] = {
                "available": bool(self.harbours), "source": "layers",
                "reason": None if self.harbours else "layers expose no harbour positions"}
        self.data_sources = o.get("data_sources")
        if self.data_sources is None:
            for module in ("ghosttrace.config_geo", "ghosttrace.layers"):
                value, _ = _import_attr(module, ("DATA_SOURCES",))
                if value:
                    self.data_sources = list(value.values()) if isinstance(value, dict) else list(value)
                    break


def _harbours_from(layers: Any) -> list[dict[str, Any]]:
    """Harbour points from whatever shape the geo layers take. Never invented."""
    candidates: Any = None
    if isinstance(layers, dict):
        for key in ("harbours", "harbors", "ports"):
            if layers.get(key):
                candidates = layers[key]
                break
    else:
        for key in ("harbours", "harbors", "ports"):
            attr = getattr(layers, key, None)
            if attr is not None:
                try:
                    candidates = attr() if callable(attr) else attr
                except Exception:
                    candidates = None
                if candidates:
                    break
    if isinstance(candidates, dict) and candidates.get("type") == "FeatureCollection":
        candidates = candidates.get("features")
    out = []
    for c in candidates or []:
        if not isinstance(c, dict):
            continue
        if c.get("type") == "Feature":
            geom = c.get("geometry") or {}
            props = c.get("properties") or {}
            if geom.get("type") == "Point" and len(geom.get("coordinates") or []) >= 2:
                lon, lat = geom["coordinates"][:2]
                out.append({"name": props.get("name"), "latitude": lat, "longitude": lon,
                            "source": props.get("source")})
            continue
        lat = c.get("latitude", c.get("lat"))
        lon = c.get("longitude", c.get("lon"))
        if lat is not None and lon is not None:
            out.append({"name": c.get("name"), "latitude": float(lat), "longitude": float(lon),
                        "source": c.get("source")})
    return out


# --- helpers ----------------------------------------------------------------------------

def _emit(on_event: EventCallback | None, event: dict[str, Any]) -> None:
    if on_event is None:
        return
    try:
        on_event({"type": "ghosttrace", **event})
    except Exception:
        pass  # a broken listener never breaks the run


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _json_safe(obj: Any) -> Any:
    """NaN/inf -> None, tuples -> lists, numpy scalars -> python, datetimes -> ISO."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
        return _json_safe(obj.tolist())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def select_targets(detections: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """(GhostTrace-class detections to analyse, suppressed ones excluded)."""
    chosen, suppressed = [], 0
    for d in detections:
        name = d.get("class_normalized") or d.get("object_class")
        if not changes_mod.is_target_class(name):
            continue
        if d.get("suppressed") and cfg.EXCLUDE_SUPPRESSED:
            suppressed += 1
            continue
        chosen.append(d)
    return chosen, suppressed


def drift_mode_for(det: dict[str, Any]) -> tuple[str, str]:
    name = changes_mod.normalize_class(det.get("class_normalized") or det.get("object_class"))
    hint = next((h for h in cfg.FLOATING_CLASS_HINTS if h in name), None)
    if hint:
        return "floating", f"class name contains floating/midwater hint '{hint}'"
    verification = det.get("verification") or {}
    if isinstance(verification, dict) and verification.get("midwater") is True:
        return "floating", "verification block states midwater: true"
    height = (det.get("dimensions") or {}).get("height_m") if isinstance(det.get("dimensions"), dict) else None
    if isinstance(height, (int, float)) and height >= cfg.MIDWATER_HEIGHT_M:
        return "floating", (f"dimensions.height_m {height:g} >= {cfg.MIDWATER_HEIGHT_M:g} m, taken as "
                            "a net held up by floats (weak proxy; heuristic)")
    return "seabed", ("default: side-scan images the seabed and a snagged net is the common case; "
                      "no floating/midwater indication")


def _nav_row(sidecar: dict[str, Any] | None, det: dict[str, Any]) -> dict[str, Any] | None:
    if not sidecar or not sidecar.get("rows"):
        return None
    rows = sidecar["rows"]
    y = det.get("global_y")
    if y is None:
        return None
    idx = int(min(max(round(float(y) - 0.5), 0), len(rows) - 1))
    return rows[idx]


def _confidence(det: dict[str, Any]) -> tuple[float | None, str]:
    pct = det.get("confidence_pct")
    if isinstance(pct, (int, float)) and math.isfinite(pct):
        return round(float(pct), 2), "verification confidence_pct (fused detector + evidence score)"
    conf = det.get("confidence")
    if isinstance(conf, (int, float)) and math.isfinite(conf):
        return round(float(conf) * 100.0, 2), ("detector confidence x 100 (no verification score; "
                                               "uncalibrated)")
    return None, "no confidence recorded"


def _survey_flags(export: dict[str, Any], sidecars: dict[str, dict[str, Any]]) -> tuple[bool, bool]:
    meta = export.get("metadata") or {}
    demo = bool(meta.get("demo")) or bool((export.get("provenance") or {}).get("demo"))
    synthetic = demo or "SYNTHETIC" in str(meta.get("data_source") or "").upper() or \
        any(bool(s.get("synthetic")) for s in sidecars.values())
    return demo, synthetic


def _cone_geometry(cone: Any, lat: float, lon: float) -> dict[str, Any] | None:
    """A GeoJSON Polygon from a drift cone of unknown shape, or None."""
    if cone is None:
        return None
    if isinstance(cone, dict):
        if cone.get("type") in ("Polygon", "MultiPolygon") and cone.get("coordinates"):
            return {"type": cone["type"], "coordinates": _json_safe(cone["coordinates"])}
        if cone.get("geometry"):
            return _cone_geometry(cone["geometry"], lat, lon)
        for key in ("polygon", "coordinates", "points", "ring"):
            if cone.get(key):
                return _cone_geometry(cone[key], lat, lon)
        return None
    if not isinstance(cone, (list, tuple)) or len(cone) < 3:
        return None
    try:
        pairs = [(float(p[0]), float(p[1])) for p in cone]
    except (TypeError, ValueError, IndexError):
        return None
    a0, b0 = pairs[0]
    # Order is not in the contract: take whichever reading lies nearer the target.
    as_lonlat = changes_mod.haversine_m(lat, lon, b0, a0) if -90 <= b0 <= 90 else math.inf
    as_latlon = changes_mod.haversine_m(lat, lon, a0, b0) if -90 <= a0 <= 90 else math.inf
    ring = [[a, b] for a, b in pairs] if as_lonlat <= as_latlon else [[b, a] for a, b in pairs]
    if ring[0] != ring[-1]:
        ring.append(list(ring[0]))
    return {"type": "Polygon", "coordinates": [ring]}


# --- the run ------------------------------------------------------------------------------

def run_ghosttrace(survey_dir: Any, *, surveys_root: Any = None, horizon_hours: float = 240,
                   n_particles: int = 500, seed: int = 0, on_event: EventCallback | None = None,
                   stages: dict[str, Any] | None = None, write: bool = True,
                   kb_dir: Any = None) -> dict[str, Any]:
    """Run GhostTrace over one processed survey. Returns the ghosttrace.json dict.

    stages: optional overrides {"habitat","drift","people","field","layers",
    "bathymetry","harbours","data_sources"} (None disables one). Used by tests
    and by callers that have already loaded the geo inputs.
    """
    survey_dir = Path(survey_dir)
    export = _load_json(survey_dir / "export.json")
    if not isinstance(export, dict):
        raise FileNotFoundError(f"export.json missing or unreadable in {survey_dir}")
    manifest = _load_json(survey_dir / "manifest.json") or {}
    meta = export.get("metadata") or {}
    survey_id = str(meta.get("survey_id") or survey_dir.name)
    detections = export.get("detections") or []

    _emit(on_event, {"stage": "load", "status": "done", "survey_id": survey_id,
                     "detections": len(detections)})
    sidecars = changes_mod.nav_sidecars(survey_dir, manifest)
    demo, synthetic = _survey_flags(export, sidecars)
    resolved = Stages(stages)
    _emit(on_event, {"stage": "stages", "status": "done", "stages": resolved.status})

    chosen, suppressed = select_targets(detections)
    _emit(on_event, {"stage": "select", "status": "done", "targets": len(chosen),
                     "suppressed_excluded": suppressed})

    time_text, time_basis = changes_mod.survey_time(survey_dir, export, manifest)
    failures: list[dict[str, Any]] = []
    wc_cache: dict[str, Any] = {}
    targets: list[dict[str, Any]] = []
    start_time_notes = set()

    for index, det in enumerate(chosen):
        det_id = str(det.get("id") or f"target-{index}")
        strip = (det.get("provenance") or {}).get("strip")
        errors: list[str] = []

        def record(stage: str, exc: BaseException) -> dict[str, Any]:
            msg = f"{type(exc).__name__}: {str(exc)[:200]}"
            errors.append(f"{stage}: {msg}")
            failures.append({"detection_id": det_id, "stage": stage, "error": msg,
                             "trace": _trace()})
            return _unavailable(f"{stage} stage failed on this target ({msg})")

        pct, pct_basis = _confidence(det)
        lat, lon = det.get("latitude"), det.get("longitude")
        has_pos = lat is not None and lon is not None
        row = _nav_row(sidecars.get(strip), det)
        depth, depth_basis = None, "no navigation row with depth_m and altitude_m for this detection"
        if row and isinstance(row.get("depth_m"), (int, float)) and isinstance(row.get("altitude_m"), (int, float)):
            depth = round(float(row["depth_m"]) + float(row["altitude_m"]), 2)
            depth_basis = ("towfish depth_m + altitude_m at the detection's ping row (seabed depth "
                           "under the towfish, not measured at the object's across-track position)")
        start_time, start_basis = time_text, time_basis
        row_time = changes_mod._row_time(row) if row else None
        if row_time:
            start_time, start_basis = str(row_time), "ping time of the detection's row (navigation sidecar)"
        start_time_notes.add(start_basis)

        target: dict[str, Any] = {
            "detection_id": det_id,
            "object_class": det.get("object_class"),
            "latitude": lat, "longitude": lon,
            "confidence_pct": pct, "confidence_basis": pct_basis,
            "dimensions": det.get("dimensions") if isinstance(det.get("dimensions"), dict) else None,
            "suppressed": bool(det.get("suppressed")),
            "strip": strip,
            "seabed_depth_m": depth, "seabed_depth_basis": depth_basis,
        }

        # activity
        try:
            target["activity"] = wc_mod.activity_evidence(det, survey_dir, other_detections=detections,
                                                          manifest=manifest, cache=wc_cache)
        except Exception as exc:
            target["activity"] = {**record("activity", exc), "score": None, "level": "unknown",
                                  "evidence": None, "basis": wc_mod.BASIS,
                                  "limitations": cfg.WC_LIMITATIONS}

        no_pos = "position unavailable (survey not georeferenced), so no geographic stage can run"
        # habitat
        if not has_pos:
            target["habitat"] = _unavailable(no_pos)
        elif resolved.habitat is None:
            target["habitat"] = _unavailable(f"habitat stage not available: {resolved.status['habitat']['reason']}")
        else:
            try:
                h = resolved.habitat(float(lat), float(lon))
                target["habitat"] = h if isinstance(h, dict) else _unavailable("habitat stage returned no data")
            except Exception as exc:
                target["habitat"] = record("habitat", exc)

        # drift
        mode, mode_basis = drift_mode_for(det)
        if not has_pos:
            target["drift"] = _unavailable(no_pos, requested_mode=mode, mode_basis=mode_basis)
        elif resolved.drift is None:
            target["drift"] = _unavailable(f"drift stage not available: {resolved.status['drift']['reason']}",
                                           requested_mode=mode, mode_basis=mode_basis)
        elif start_time is None:
            target["drift"] = _unavailable("no survey time recorded, so a forecast cannot be anchored",
                                           requested_mode=mode, mode_basis=mode_basis)
        else:
            kwargs = dict(mode=mode, horizon_hours=horizon_hours, n_particles=n_particles,
                          dt_minutes=cfg.DRIFT_DT_MINUTES, field=resolved.field,
                          layers=resolved.layers, seed=seed)
            parsed = changes_mod._parse_time(start_time)
            try:
                try:
                    d = resolved.drift(float(lat), float(lon), parsed or start_time, **kwargs)
                except (TypeError, AttributeError):
                    if parsed is None:
                        raise
                    d = resolved.drift(float(lat), float(lon), parsed.isoformat(), **kwargs)
                if isinstance(d, dict):
                    d = dict(d)
                    d.setdefault("requested_mode", mode)
                    d["mode_basis"] = mode_basis
                    d["start_time"] = parsed.isoformat() if parsed else start_time
                    d["start_time_basis"] = start_basis
                    target["drift"] = d
                else:
                    target["drift"] = _unavailable("drift stage returned no data",
                                                   requested_mode=mode, mode_basis=mode_basis)
            except Exception as exc:
                target["drift"] = {**record("drift", exc), "requested_mode": mode, "mode_basis": mode_basis}

            # A seabed net that the currents cannot move today is not a net that
            # cannot move: storms and trawls refloat gear. The floating run is
            # kept beside the forecast as a labelled SCENARIO, so an operator
            # can see what it would reach if lifted. It never feeds priority,
            # which rests on the forecast for the mode the net was found in.
            if mode == "seabed" and cfg.DRIFT_REFLOAT_SCENARIO:
                try:
                    s_kwargs = dict(kwargs, mode="floating")
                    try:
                        s = resolved.drift(float(lat), float(lon), parsed or start_time, **s_kwargs)
                    except (TypeError, AttributeError):
                        if parsed is None:
                            raise
                        s = resolved.drift(float(lat), float(lon), parsed.isoformat(), **s_kwargs)
                    if isinstance(s, dict):
                        s = dict(s)
                        s.update({
                            "scenario": "if_refloated",
                            "scenario_note": ("what this net would reach if lifted off the seabed "
                                              "(e.g. by a storm or a trawl) at survey time; not a "
                                              "forecast, and not used in the priority score"),
                            "start_time": parsed.isoformat() if parsed else start_time,
                            "start_time_basis": start_basis,
                        })
                        target["drift_scenarios"] = {"if_refloated": s}
                except Exception as exc:
                    target["drift_scenarios"] = {"if_refloated": record("drift", exc)}

        # people
        if not has_pos:
            target["people"] = _unavailable(no_pos)
        elif resolved.people is None:
            target["people"] = _unavailable(f"safety stage not available: {resolved.status['people']['reason']}")
        else:
            try:
                p = resolved.people(target, target["habitat"], target["drift"], resolved.field,
                                    resolved.bathymetry)
                target["people"] = p if isinstance(p, dict) else _unavailable("safety stage returned no data")
            except Exception as exc:
                target["people"] = record("people", exc)

        target["errors"] = errors
        targets.append(target)
        _emit(on_event, {"stage": "target", "status": "done", "index": index + 1, "total": len(chosen),
                         "detection_id": det_id, "activity_level": target["activity"].get("level"),
                         "errors": errors})

    # change
    notes: list[str] = []
    try:
        comparison = changes_mod.compare_with_previous(survey_dir, targets, surveys_root=surveys_root,
                                                       export=export, manifest=manifest)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {str(exc)[:200]}"
        failures.append({"detection_id": None, "stage": "change", "error": msg,
                         "trace": _trace()})
        comparison = {"changes": {}, "removed_since_previous": [],
                      "change_summary": {"compared_with": [], "new": 0, "moved": 0,
                                         "persistent": 0, "removed": 0, "confirmed_recovered": 0},
                      "notes": [f"change detection failed ({msg})"]}
    notes += comparison.get("notes") or []
    for t in targets:
        t["change"] = comparison["changes"].get(t["detection_id"]) or {
            "status": "unmatched_no_prior", "previous_survey_id": None, "previous_detection_id": None,
            "previous_latitude": None, "previous_longitude": None,
            "moved_m": None, "basis": "change detection produced no result for this target"}
    _emit(on_event, {"stage": "change", "status": "done", **comparison["change_summary"]})

    # priority
    for t in targets:
        try:
            t["priority"] = priority_mod.score_target(t)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {str(exc)[:200]}"
            t["errors"].append(f"priority: {msg}")
            failures.append({"detection_id": t["detection_id"], "stage": "priority", "error": msg,
                             "trace": _trace()})
            t["priority"] = {"score": 0.0, "rank": None, "tier": "routine", "terms": {},
                             "formula": "not computed", "basis": f"priority failed ({msg})"}
    priority_mod.rank_targets(targets)
    targets.sort(key=lambda t: t["priority"]["rank"])
    _emit(on_event, {"stage": "priority", "status": "done"})

    # alerts
    for t in targets:
        try:
            t["alert"] = alerts_mod.draft_alert(t, survey_id=survey_id, demo=demo, synthetic=synthetic,
                                                kb_dir=kb_dir)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {str(exc)[:200]}"
            t["errors"].append(f"alert: {msg}")
            failures.append({"detection_id": t["detection_id"], "stage": "alert", "error": msg,
                             "trace": _trace()})
            t["alert"] = {"authorities": [], "subject": None, "draft_text": None,
                          "generated_by": "template", "citations": [],
                          "basis": f"alert drafting failed ({msg})"}
    _emit(on_event, {"stage": "alerts", "status": "done"})

    # recovery
    try:
        plan = recovery_mod.plan_route(targets, resolved.harbours)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {str(exc)[:200]}"
        failures.append({"detection_id": None, "stage": "recovery", "error": msg,
                         "trace": _trace()})
        plan = {"start": None, "order": [], "legs": [], "total_km": 0.0, "method": "not computed",
                "notes": [f"recovery planning failed ({msg})"]}
    _emit(on_event, {"stage": "recovery", "status": "done", "total_km": plan.get("total_km")})

    summary = _summary(targets, suppressed, failures)
    result = {
        "format": cfg.FORMAT,
        "survey_id": survey_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "demo": demo,
        "synthetic_inputs": synthetic,
        "data_sources": _data_sources(targets, resolved, kb_dir),
        "caveats": _caveats(demo, synthetic, targets, resolved, start_time_notes, notes, suppressed,
                            failures),
        "targets": targets,
        "removed_since_previous": comparison["removed_since_previous"],
        "change_summary": comparison["change_summary"],
        "recovery_plan": plan,
        "summary": summary,
        "run": {
            "core_version": cfg.CORE_VERSION,
            "horizon_hours": horizon_hours, "n_particles": n_particles, "seed": seed,
            "drift_dt_minutes": cfg.DRIFT_DT_MINUTES,
            "survey_time": time_text, "survey_time_basis": time_basis,
            "stages": resolved.status,
            "failures": failures,
            "priority_weights": dict(cfg.PRIORITY_WEIGHTS),
            "priority_neutral": dict(cfg.PRIORITY_NEUTRAL),
            "priority_tiers": {n: f for n, f in cfg.PRIORITY_TIERS},
            "heuristic": cfg.HEURISTIC_LABEL,
        },
    }
    result = _json_safe(result)
    if write:
        _atomic_write(survey_dir / cfg.OUTPUT_JSON, result)
        _atomic_write(survey_dir / cfg.OUTPUT_GEOJSON, to_geojson(result))
    _emit(on_event, {"stage": "write" if write else "complete", "status": "done",
                     "summary": summary, "path": cfg.OUTPUT_JSON if write else None})
    return result


def _near_habitat(h: dict[str, Any]) -> bool:
    if not isinstance(h, dict) or h.get("available") is False:
        return False
    if h.get("inside"):
        return True
    if isinstance(h.get("score"), (int, float)) and h["score"] >= cfg.NEAR_HABITAT_SCORE:
        return True
    return any(isinstance(n.get("distance_m"), (int, float)) and n["distance_m"] <= cfg.NEAR_HABITAT_M
               for n in h.get("nearest") or [] if isinstance(n, dict))


def _summary(targets: list[dict[str, Any]], suppressed: int, failures: list) -> dict[str, Any]:
    def level(t):
        return str(((t.get("people") or {}).get("propeller_hazard") or {}).get("level") or "").lower()
    return {
        "targets": len(targets),
        "urgent": sum(1 for t in targets if t["priority"]["tier"] == "urgent"),
        "high": sum(1 for t in targets if t["priority"]["tier"] == "high"),
        "actively_fishing": sum(1 for t in targets if (t.get("activity") or {}).get("level") == "high"),
        "near_sensitive_habitat": sum(1 for t in targets if _near_habitat(t.get("habitat"))),
        "propeller_hazards": sum(1 for t in targets if level(t) in cfg.PROPELLER_HAZARD_LEVELS),
        "suppressed_excluded": suppressed,
        "stage_failures": len(failures),
    }


def _data_sources(targets, resolved: Stages, kb_dir) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(entry: dict[str, Any]) -> None:
        key = str(entry.get("name"))
        if key and key not in seen:
            seen.add(key)
            out.append({"name": entry.get("name"), "url": entry.get("url"),
                        "licence": entry.get("licence", entry.get("license")),
                        "snapshot": entry.get("snapshot"), "used_for": entry.get("used_for")})

    for entry in resolved.data_sources or []:
        if isinstance(entry, dict):
            add(entry)
    for t in targets:
        cs = (t.get("drift") or {}).get("current_source")
        if isinstance(cs, dict):
            add({**cs, "used_for": cs.get("used_for") or "drift forecast currents"})
        elif isinstance(cs, str) and cs:
            add({"name": cs, "url": None, "licence": None, "snapshot": None,
                 "used_for": "drift forecast currents (details not stated by the drift stage)"})
        for n in (t.get("habitat") or {}).get("nearest") or []:
            src = n.get("source") if isinstance(n, dict) else None
            if isinstance(src, dict):
                add({**src, "used_for": src.get("used_for") or "habitat proximity"})
            elif isinstance(src, str) and src:
                add({"name": src, "url": None, "licence": None, "snapshot": None,
                     "used_for": "habitat proximity (details not stated by the habitat stage)"})
    kbd = alerts_mod.kb_dir_path(kb_dir)
    cited = []
    for t in targets:
        for cit in (t.get("alert") or {}).get("citations") or []:
            doc = str(cit.get("doc") or "")
            if doc.startswith("kb/") and doc[3:] not in cited:
                cited.append(doc[3:])
    for doc in cited:
        text = alerts_mod._kb_text(kbd, doc)
        if not text:
            continue
        fm = alerts_mod._front_matter(text)
        add({"name": f"DeepEcho corpus: {fm.get('title') or doc} (kb/{doc})",
             "url": fm.get("source_url"), "licence": None, "snapshot": fm.get("retrieved"),
             "used_for": "alert authorities and report fields (licence not recorded in the corpus)"})
    return out


def _caveats(demo, synthetic, targets, resolved: Stages, start_notes, notes, suppressed, failures
             ) -> list[str]:
    c: list[str] = []
    if demo or synthetic:
        c.append("SYNTHETIC OR DEMO INPUTS: nothing in this file is evidence of a real object.")
    c.append("GhostTrace priorities, tiers, activity scores and weights are configurable heuristics "
             "chosen for this project, not fitted to real ghost-gear recoveries and not an official "
             "procedure. Every detection is automated - verify before action.")
    if any((t.get("activity") or {}).get("available") for t in targets):
        c.append("Water-column activity: " + cfg.WC_LIMITATIONS)
    elif targets:
        c.append("Water-column activity was not measured for any target (no water-column record); "
                 "a neutral value was used in the priority score.")
    for key, label in (("habitat", "Habitat context"), ("drift", "Drift forecast"),
                       ("people", "Propeller and diver safety")):
        st = resolved.status.get(key) or {}
        if not st.get("available"):
            c.append(f"{label} unavailable for this run: {st.get('reason')}.")
    if any(t.get("latitude") is None for t in targets):
        c.append("Some targets have no geographic position (survey not georeferenced); habitat, drift, "
                 "safety, change and routing are unavailable for them.")
    drift_ran = any((t.get("drift") or {}).get("available", True) is not False for t in targets)
    for note in sorted(start_notes):
        if drift_ran and "processed_at" in note:
            c.append("Drift forecasts are anchored to the time the survey was processed, not recorded "
                     "(no ping times in the navigation).")
    c.append("Recovery legs are straight lines; routes around land and shoals are not computed.")
    c.append("Alerts name an authority only where the knowledge base names it for that situation; "
             "'authority not in corpus' marks a gap to fill, and no contact details are invented.")
    if suppressed:
        c.append(f"{suppressed} detection(s) flagged suppressed by verification were excluded from targets.")
    if failures:
        c.append(f"{len(failures)} stage failure(s) were isolated and recorded in run.failures.")
    c.extend(notes)
    return c


def to_geojson(result: dict[str, Any]) -> dict[str, Any]:
    """Targets as Points, final-snapshot cone90 as Polygons, recovery route as a LineString."""
    features = []
    for t in result.get("targets") or []:
        lat, lon = t.get("latitude"), t.get("longitude")
        if lat is None or lon is None:
            continue
        pr = t.get("priority") or {}
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
                         "properties": {
                             "kind": "target", "detection_id": t["detection_id"],
                             "object_class": t.get("object_class"), "rank": pr.get("rank"),
                             "tier": pr.get("tier"), "score": pr.get("score"),
                             "confidence_pct": t.get("confidence_pct"),
                             "activity_level": (t.get("activity") or {}).get("level"),
                             "change_status": (t.get("change") or {}).get("status"),
                             "propeller_hazard": ((t.get("people") or {}).get("propeller_hazard") or {}).get("level"),
                             "synthetic": bool(result.get("synthetic_inputs"))}})
        snaps = (t.get("drift") or {}).get("snapshots") or []
        if snaps:
            final = max(snaps, key=lambda s: s.get("t_hours") or 0)
            geom = _cone_geometry(final.get("cone90"), float(lat), float(lon))
            if geom:
                features.append({"type": "Feature", "geometry": geom, "properties": {
                    "kind": "drift_cone90", "detection_id": t["detection_id"],
                    "t_hours": final.get("t_hours"), "mode": (t.get("drift") or {}).get("mode")
                    or (t.get("drift") or {}).get("requested_mode")}})
    plan = result.get("recovery_plan") or {}
    by_id = {t["detection_id"]: t for t in result.get("targets") or []}
    coords = []
    start = plan.get("start")
    if start and not str(start.get("name", "")).startswith("target "):
        coords.append([start["longitude"], start["latitude"]])
    for ident in plan.get("order") or []:
        t = by_id.get(ident)
        if t and t.get("latitude") is not None:
            coords.append([t["longitude"], t["latitude"]])
    if len(coords) >= 2:
        features.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                         "properties": {"kind": "recovery_route", "total_km": plan.get("total_km"),
                                        "start": (start or {}).get("name"),
                                        "note": "straight-line legs; not a navigable sea route"}})
    return {"type": "FeatureCollection", "format": cfg.FORMAT + "+geojson",
            "survey_id": result.get("survey_id"), "synthetic": bool(result.get("synthetic_inputs")),
            "features": features}
