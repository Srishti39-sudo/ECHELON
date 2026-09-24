"""Who could this net hurt: propeller hazard and a diver brief for recovery planning.

    people_safety(target, habitat, drift, field, bathymetry) -> {
        "available": True,
        "propeller_hazard": {"level": "high" | "moderate" | "low" | "unknown",
                             "points": float, "reasons": [...], "terms": {...},
                             "heuristic": str},
        "diver_brief": {"seabed_depth_m", "depth_source", "net_size_m",
                        "current_mps_at_depth", "current_mps_surface", "current_window",
                        "within_recreational_limit", "within_open_water_limit",
                        "current_ok_for_divers", "entanglement_risk",
                        "recommended_method", "method_reasons", "summary", "notes",
                        "thresholds": {"recreational_depth_m", "advanced_depth_m",
                                       "max_current_mps", "max_current_knots",
                                       "depth_check_uses", "depth_basis", "depth_label",
                                       "current_basis", "current_label", "basis"}},
    }

The arguments are what the GhostTrace engine already holds for a target: the
target record (latitude, longitude, dimensions, seabed_depth_m), the outputs
of habitat_context and drift_forecast (either may be {"available": false}),
the loaded current field and the bathymetry grid (either may be None). Nothing
here fetches data of its own, and every missing input becomes null plus a
reason rather than a default.

PROPELLER HAZARD (a heuristic)
    A lost net is a hazard to vessels: it fouls propellers and rudders, and a
    fouled small fishing boat can be disabled at sea. Points from
    config_geo.PROP_POINTS add up:

        shallow        seabed depth < PROP_SHALLOW_M
        floating       the object is floating / midwater (drift mode)
        harbour_near   a harbour or landing point within PROP_HARBOUR_NEAR_M
        harbour_drift  the drift forecast reaches a harbour buffer with
                       probability >= PROP_HARBOUR_DRIFT_P
        large_net      net size >= PROP_LARGE_NET_M

    The level is the first tier in PROP_TIERS whose floor the total reaches.
    It is "unknown" when none of the inputs could be assessed (no depth, no
    mode, no harbour information, no size), because "low" would then be a
    claim nothing supports. Every term reports whether it was assessed.

DIVER BRIEF
    Depth: the survey's own seabed depth when the engine has one (towfish depth
    plus altitude); otherwise NOAA ETOPO 2022 at the position, labelled as such.
    Current: the modelled NEAR-BOTTOM current at the position, maximum over the
    next DIVER_CURRENT_WINDOW_H hours from the forecast start (the same time
    mapping the drift forecast used). A ~9 km model cell's near-bottom velocity
    is a planning figure, not a measurement at the net.

    Limits (config_geo.py): recreational depth limits are PADI's 18 m (Open
    Water) and 30 m (Advanced Open Water); the diver current limit is 1 knot,
    from US OSHA 29 CFR 1910.424(b)(3) (no SCUBA against currents exceeding
    one knot unless line-tended), used as a planning analogue - it is US
    workplace law, not Indian law. A recovery that fails either points to a
    remotely operated vehicle or a grapnel from the surface. The numbers
    actually used, and where each comes from, are written into
    diver_brief.thresholds so a reader never has to guess them.

    This is planning context for trained, equipped recovery teams, not a dive
    plan and not an operational procedure. If there is any suspicion the
    object is not a net (ordnance, a drum of unknown contents), the corpus's
    unidentified-object protocol applies instead of a recovery.
"""

from __future__ import annotations

import math
from typing import Any

from ghosttrace import config_geo as cfg
from ghosttrace.currents import FieldCollection, to_epoch_seconds


def _num(x: Any) -> float | None:
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else None


def _available(block: Any) -> bool:
    return isinstance(block, dict) and block.get("available", True) is not False


def _net_size(target: dict[str, Any]) -> tuple[float | None, str]:
    dims = target.get("dimensions") if isinstance(target.get("dimensions"), dict) else None
    if not dims:
        return None, "no dimensions recorded for the detection"
    vals = [_num(dims.get(k)) for k in ("length_m", "width_m")]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, "dimensions carry no length_m / width_m"
    return round(max(vals), 2), "largest of dimensions.length_m and width_m (from the sonar image)"


def _depth(target: dict[str, Any], lat: float | None, lon: float | None, bathymetry: Any) -> dict[str, Any]:
    d = _num(target.get("seabed_depth_m"))
    if d is not None:
        return {"depth_m": round(abs(d), 1), "source": "survey navigation",
                "basis": target.get("seabed_depth_basis") or "target seabed_depth_m"}
    if bathymetry is not None and lat is not None and lon is not None:
        rec = bathymetry.depth_at(lat, lon)
        if rec.get("depth_m") is not None:
            return {"depth_m": rec["depth_m"], "source": rec.get("source"), "basis": rec.get("method"),
                    "note": rec.get("note")}
        if rec.get("on_land"):
            return {"depth_m": None, "source": rec.get("source"),
                    "basis": "bathymetry grid reports elevation >= 0 at this position (on land or intertidal)"}
        return {"depth_m": None, "source": None, "basis": rec.get("note") or "position outside the bathymetry grids"}
    return {"depth_m": None, "source": None, "basis": "no survey depth and no bathymetry grid available"}


def _forecast_start(drift: Any) -> tuple[Any, str]:
    if _available(drift):
        tm = drift.get("time_mapping") or {}
        if tm.get("model_start"):
            note = "drift forecast model start"
            if tm.get("shifted"):
                note += " (shifted into the bundled current window, see drift.time_mapping)"
            return tm["model_start"], note
    if isinstance(drift, dict) and drift.get("start_time"):
        return drift["start_time"], "target start time"
    return None, "no forecast start time available"


def _current(field: Any, lat: float, lon: float, start: Any, level: str) -> dict[str, Any]:
    if field is None:
        return {"max_mps": None, "reason": "no current field loaded"}
    f = field.field_for(lat, lon) if isinstance(field, FieldCollection) else field
    if f is None or not bool(f.contains(lat, lon)):
        return {"max_mps": None, "reason": "no bundled current data for this location"}
    if start is None:
        return {"max_mps": None, "reason": "no start time to sample the current at"}
    s = to_epoch_seconds(start)
    if not f.in_time(s):
        # Same rule as the drift stage: use the nearest available window, and say so.
        s_new = min(max(s, float(f.times_s[0])), float(f.times_s[-1]) - cfg.DIVER_CURRENT_WINDOW_H * 3600.0)
        out = f.max_speed(lat, lon, s_new, cfg.DIVER_CURRENT_WINDOW_H, level)
        out["note"] = "start time outside the bundled current window; nearest window used"
    else:
        out = f.max_speed(lat, lon, s, cfg.DIVER_CURRENT_WINDOW_H, level)
    out["level"] = level
    out["source"] = f.source + (" [SYNTHETIC TEST FIELD]" if getattr(f, "synthetic", False) else "")
    if out.get("max_mps") is None:
        out["reason"] = "model has no valid current at this position (model land mask)"
    return out


def _thresholds() -> dict[str, Any]:
    """The limits the diver brief actually applied, each with where it comes from."""
    current_cited = abs(cfg.DIVER_CURRENT_LIMIT_MPS - cfg.DIVER_CURRENT_LIMIT_DEFAULT_MPS) < 1e-9
    depth_basis = cfg.DIVE_DEPTH_LIMIT_CITATION
    if (cfg.DIVE_LIMIT_OPEN_WATER_M, cfg.DIVE_LIMIT_ADVANCED_M) != (18.0, 30.0):
        depth_basis = ("overridden through GHOSTTRACE_DIVE_LIMIT_* (heuristic); the PADI defaults are "
                       "18 m and 30 m")
    current_basis = cfg.DIVER_CURRENT_LIMIT_CITATION if current_cited else (
        "overridden through GHOSTTRACE_DIVER_CURRENT_LIMIT_MPS (heuristic); the cited default is 1 knot, "
        "US OSHA 29 CFR 1910.424(b)(3)")
    return {
        "recreational_depth_m": cfg.DIVE_LIMIT_OPEN_WATER_M,
        "advanced_depth_m": cfg.DIVE_LIMIT_ADVANCED_M,
        "max_current_mps": cfg.DIVER_CURRENT_LIMIT_MPS,
        "max_current_knots": round(cfg.DIVER_CURRENT_LIMIT_MPS / 0.514444, 2),
        "depth_check_uses": "advanced_depth_m",
        "depth_label": "cited" if "overridden" not in depth_basis else "heuristic",
        "depth_basis": depth_basis,
        "current_label": "cited" if current_cited else "heuristic",
        "current_basis": current_basis,
        "current_window_hours": cfg.DIVER_CURRENT_WINDOW_H,
        "basis": ("within_recreational_limit is seabed depth <= advanced_depth_m and within_open_water_limit "
                  "is <= recreational_depth_m; current_ok_for_divers is the modelled near-bottom current "
                  f"(maximum over {cfg.DIVER_CURRENT_WINDOW_H:g} h) <= max_current_mps. Training-agency "
                  "depth limits and a US workplace current rule, used as planning analogues: not Indian "
                  "law, not a dive plan, not an operational procedure."),
    }


def people_safety(target: dict[str, Any], habitat: Any = None, drift: Any = None, field: Any = None,
                  bathymetry: Any = None) -> dict[str, Any]:
    """Propeller hazard and diver brief for one target. See the module docstring."""
    target = target or {}
    lat, lon = _num(target.get("latitude")), _num(target.get("longitude"))
    if lat is None or lon is None:
        return {"available": False, "reason": "position unavailable"}
    if bathymetry is None:
        try:
            from ghosttrace.layers import load_bathymetry
            bathymetry = load_bathymetry()
        except Exception:
            bathymetry = None

    size, size_basis = _net_size(target)
    depth = _depth(target, lat, lon, bathymetry)
    depth_m = depth["depth_m"]
    mode = None
    if isinstance(drift, dict):
        mode = drift.get("mode") or drift.get("requested_mode")

    # -- propeller hazard -----------------------------------------------------------------
    terms: dict[str, Any] = {}
    reasons: list[str] = []
    assessed_any = False

    def term(name: str, assessed: bool, applied: bool, basis: str) -> None:
        nonlocal assessed_any
        pts = cfg.PROP_POINTS[name] if applied else 0.0
        terms[name] = {"assessed": assessed, "applied": applied, "points": pts, "basis": basis}
        assessed_any = assessed_any or assessed
        if applied:
            reasons.append(basis)

    if depth_m is None:
        term("shallow", False, False, f"seabed depth unknown ({depth['basis']})")
    else:
        term("shallow", True, depth_m < cfg.PROP_SHALLOW_M,
             f"seabed depth {depth_m:.1f} m {'<' if depth_m < cfg.PROP_SHALLOW_M else '>='} "
             f"{cfg.PROP_SHALLOW_M:g} m ({depth['source']})")

    if mode is None:
        term("floating", False, False, "drift mode unknown")
    else:
        term("floating", True, mode == "floating",
             "object treated as floating / midwater: in the propeller depth band" if mode == "floating"
             else "object treated as on the seabed")

    harbour = None
    if _available(habitat) and habitat.get("covered"):
        near = [n for n in habitat.get("nearest") or [] if n.get("kind") == "harbour"
                and isinstance(n.get("distance_m"), (int, float))]
        harbour = min(near, key=lambda n: n["distance_m"]) if near else None
        if harbour and harbour["distance_m"] <= cfg.PROP_HARBOUR_NEAR_M:
            term("harbour_near", True, True,
                 f"harbour / landing point {harbour.get('name') or '(unnamed)'} {harbour['distance_m'] / 1000:.1f} km away "
                 f"(<= {cfg.PROP_HARBOUR_NEAR_M / 1000:g} km; {harbour.get('source')})")
        else:
            term("harbour_near", True, False,
                 f"no mapped harbour within {cfg.PROP_HARBOUR_NEAR_M / 1000:g} km (OSM harbour coverage is incomplete)")
    else:
        term("harbour_near", False, False, "habitat context not available or position not covered by bundled layers")

    if _available(drift) and isinstance(drift.get("impacts"), list):
        hp = max([float(i.get("probability") or 0) for i in drift["impacts"] if i.get("kind") == "harbour"] or [0.0])
        term("harbour_drift", True, hp >= cfg.PROP_HARBOUR_DRIFT_P,
             f"drift reaches a harbour buffer with probability {hp:.2f} "
             f"({'>=' if hp >= cfg.PROP_HARBOUR_DRIFT_P else '<'} {cfg.PROP_HARBOUR_DRIFT_P:g})")
    else:
        term("harbour_drift", False, False, "no drift forecast available")

    if size is None:
        term("large_net", False, False, f"net size unknown ({size_basis})")
    else:
        term("large_net", True, size >= cfg.PROP_LARGE_NET_M,
             f"net size {size:.1f} m {'>=' if size >= cfg.PROP_LARGE_NET_M else '<'} {cfg.PROP_LARGE_NET_M:g} m")

    points = sum(t["points"] for t in terms.values())
    if not assessed_any:
        level = "unknown"
    else:
        level = next(name for name, floor in cfg.PROP_TIERS if points >= floor)
        if level == "low" and not terms["shallow"]["assessed"] and not terms["floating"]["applied"]:
            level = "unknown"
            reasons.append("depth unknown, so a low rating cannot be supported")

    propeller = {"level": level, "points": points, "reasons": reasons, "terms": terms,
                 "tiers": [list(t) for t in cfg.PROP_TIERS], "heuristic": cfg.HEURISTIC_LABEL}

    # -- diver brief -----------------------------------------------------------------------
    start, start_note = _forecast_start(drift)
    bottom = _current(field, lat, lon, start, "bottom")
    surface = _current(field, lat, lon, start, "surface")
    cur = _num(bottom.get("max_mps"))

    within_rec = None if depth_m is None else depth_m <= cfg.DIVE_LIMIT_ADVANCED_M
    within_ow = None if depth_m is None else depth_m <= cfg.DIVE_LIMIT_OPEN_WATER_M
    current_ok = None if cur is None else cur <= cfg.DIVER_CURRENT_LIMIT_MPS

    method_reasons: list[str] = []
    if depth_m is None or cur is None:
        method = "ROV / grapnel from the surface until depth and current are known"
        if depth_m is None:
            method_reasons.append("seabed depth unknown")
        if cur is None:
            method_reasons.append(f"near-bottom current unknown ({bottom.get('reason')})")
    elif depth_m > cfg.DIVE_LIMIT_ADVANCED_M or not current_ok:
        method = "ROV / grapnel from the surface"
        if depth_m > cfg.DIVE_LIMIT_ADVANCED_M:
            method_reasons.append(f"depth {depth_m:.1f} m beyond the {cfg.DIVE_LIMIT_ADVANCED_M:g} m "
                                  "recreational limit (professional diving only)")
        if not current_ok:
            method_reasons.append(f"near-bottom current up to {cur:.2f} m/s exceeds the "
                                  f"{cfg.DIVER_CURRENT_LIMIT_MPS:g} m/s (1 knot) SCUBA current limit")
    else:
        if size is not None and size >= cfg.PROP_LARGE_NET_M:
            method = "diver recovery with lift bags / surface winch (trained team)"
            method_reasons.append(f"net size {size:.1f} m: too large to hand-lift")
        else:
            method = "diver recovery (trained team, cutting tools, surface support)"
        method_reasons.append(f"depth {depth_m:.1f} m within {cfg.DIVE_LIMIT_ADVANCED_M:g} m and near-bottom "
                              f"current up to {cur:.2f} m/s within {cfg.DIVER_CURRENT_LIMIT_MPS:g} m/s")

    if size is not None and size >= cfg.PROP_LARGE_NET_M:
        ent = "high: large net; entanglement of divers and ROV tethers is likely without a cutting plan"
    elif cur is not None and cur > 0.5 * cfg.DIVER_CURRENT_LIMIT_MPS:
        ent = "high: current will push loose netting onto a diver or tether"
    elif size is None:
        ent = "present, size unknown: any net is an entanglement hazard to divers and ROV tethers"
    else:
        ent = "moderate: any net is an entanglement hazard; carry cutting tools and use a buddy / tender"

    thresholds = _thresholds()
    notes = [
        "planning context for trained recovery teams; not a dive plan or an operational procedure",
        "depth limits: PADI Open Water 18 m, Advanced Open Water 30 m (training-agency limits, not law)",
        (f"current limit {cfg.DIVER_CURRENT_LIMIT_MPS:g} m/s (1 knot): US OSHA 29 CFR 1910.424(b)(3), "
         "no SCUBA against currents over one knot unless line-tended; a US regulation used as a "
         "planning analogue, not Indian law"
         if thresholds["current_label"] == "cited" else
         f"current limit {cfg.DIVER_CURRENT_LIMIT_MPS:g} m/s is an overridden value (heuristic), "
         "not the cited 1-knot rule"),
        "near-bottom current is a ~9 km model-cell value at the deepest model level, not a measurement at the net",
        "if the object may not be a net (ordnance, drum of unknown contents), do not recover: follow the "
        "unidentified-object protocol",
    ]
    if bottom.get("note"):
        notes.append(bottom["note"])
    brief = {
        "seabed_depth_m": depth_m, "depth_source": depth["source"], "depth_basis": depth["basis"],
        "net_size_m": size, "net_size_basis": size_basis,
        "current_mps_at_depth": cur, "current_mps_surface": _num(surface.get("max_mps")),
        "current_window": {"start": start, "start_basis": start_note, "hours": cfg.DIVER_CURRENT_WINDOW_H,
                           "bottom": bottom, "surface": surface},
        "within_recreational_limit": within_rec, "within_open_water_limit": within_ow,
        "recreational_limit_m": {"open_water": cfg.DIVE_LIMIT_OPEN_WATER_M, "advanced": cfg.DIVE_LIMIT_ADVANCED_M},
        "current_ok_for_divers": current_ok, "diver_current_limit_mps": cfg.DIVER_CURRENT_LIMIT_MPS,
        "entanglement_risk": ent,
        "recommended_method": method, "method_reasons": method_reasons,
        "notes": notes,
        "thresholds": thresholds,
    }
    parts = [f"depth {depth_m:.1f} m ({depth['source']})" if depth_m is not None else "depth unknown",
             f"near-bottom current up to {cur:.2f} m/s over {cfg.DIVER_CURRENT_WINDOW_H:g} h" if cur is not None
             else "near-bottom current unknown",
             f"method: {method}"]
    brief["summary"] = "; ".join(parts)
    return {"available": True, "propeller_hazard": propeller, "diver_brief": brief}
