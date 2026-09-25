"""What a ghost net is lying on or near: habitat context for one position.

    habitat_context(lat, lon) -> {
        "available": True, "covered": bool, "region": str | None, "reason": str | None,
        "inside":  [{"layer", "kind", "name", "source", "geometry_quality", ...}],
        "nearest": [{"layer", "kind", "name", "distance_m", "bearing_deg",
                     "source", "geometry_quality", ...}],        one per kind
        "kinds_assessed": [...], "kinds_not_assessed": [...],
        "score": 0..1 | None, "terms": {kind: {...}}, "formula": str, "heuristic": str,
        "search_radius_m": float,
    }

COVERAGE FIRST
    Before any search, layers.region_for() is asked whether bundled data covers
    the point. Outside coverage the answer is covered: false, score: null,
    empty lists and a reason that says "no bundled data for this location".
    An empty `nearest` list INSIDE coverage means "searched, nothing within the
    radius"; outside coverage it means nothing at all, and the `covered` flag is
    what tells a reader which. A kind with no layer in the covered region (for
    example turtle nesting in the Gulf of Mannar box) is listed under
    kinds_not_assessed so its absence is not read as a clean result.

DISTANCES AND BEARINGS
    Candidates are prefiltered by bounding box, then measured in an azimuthal
    equidistant projection centred on the target (layers.to_local), which finds
    the nearest point of each feature. The reported distance and bearing are
    then recomputed on the WGS84 ellipsoid with pyproj.Geod.inv between the
    target and that nearest point, so the numbers are geodesic. bearing_deg is
    the true azimuth FROM the target TO the feature, clockwise from north. A
    target inside a polygon has distance_m 0 and bearing_deg null.

SCORE (a heuristic, and labelled as one)
    term(kind) = sensitivity(kind) * exp(-distance_m / HABITAT_DECAY_M)
    score      = 1 - product over kinds (1 - term(kind))

    Only the nearest feature of each kind counts, so a dense reef layer cannot
    add up to certainty on its own. Harbours have sensitivity 0: they are a
    people exposure, handled in safety.py, and are reported under `nearest` for
    context only. The sensitivities and decay length live in config_geo.py and
    are copied into every result.

    What the score is not: an ecological risk assessment. It says how close
    the net is to mapped sensitive features, weighted by a configurable
    consequence. It knows nothing about the net's condition, the season (the
    olive ridley arribada is roughly November to April) or feature quality
    beyond the geometry_quality label, which is passed through for the reader.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import shapely
from shapely.geometry import Point
from shapely.ops import nearest_points

from ghosttrace import config_geo as cfg
from ghosttrace.layers import GEOD, LayerSet, degree_pad, load_default_layers, to_local, to_lonlat

HABITAT_KINDS = ("reef", "seagrass", "protected_area", "turtle_nesting", "dugong", "harbour")

_PASS_THROUGH = ("citation", "construction", "osm_id", "osm_url", "designation", "species",
                 "source_id", "note", "fishing")


def _record(layer, kind: str, props: dict[str, Any]) -> dict[str, Any]:
    rec = {"layer": layer.name, "kind": kind, "name": props.get("name"),
           "source": props.get("source"), "geometry_quality": props.get("geometry_quality")}
    for key in _PASS_THROUGH:
        if props.get(key) is not None:
            rec[key] = props[key]
    return rec


def formula_text() -> str:
    return (f"score = 1 - prod_kinds(1 - sensitivity(kind) * exp(-distance_m / {cfg.HABITAT_DECAY_M:g})); "
            "nearest feature per kind; distance 0 inside a polygon")


def habitat_context(lat: float, lon: float, *, layers: LayerSet | None = None,
                    radius_m: float | None = None) -> dict[str, Any]:
    """Habitat context for one position. See the module docstring for the contract."""
    layers = layers if layers is not None else load_default_layers()
    radius = float(radius_m if radius_m is not None else cfg.HABITAT_SEARCH_RADIUS_M)
    base = {"available": True, "search_radius_m": radius, "formula": formula_text(),
            "heuristic": cfg.HEURISTIC_LABEL,
            "sensitivity": {k: cfg.KIND_SENSITIVITY.get(k, 0.0) for k in HABITAT_KINDS},
            "decay_m": cfg.HABITAT_DECAY_M}
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return {**base, "covered": False, "region": None, "reason": "position is not a finite number",
                "inside": [], "nearest": [], "kinds_assessed": [], "kinds_not_assessed": list(HABITAT_KINDS),
                "score": None, "terms": {}}
    coverage = layers.region_for(lat, lon)
    if not coverage["covered"]:
        return {**base, "covered": False, "region": coverage["region"], "reason": coverage["reason"],
                "inside": [], "nearest": [], "kinds_assessed": [], "kinds_not_assessed": list(HABITAT_KINDS),
                "score": None, "terms": {}}

    region = coverage["region"]
    target = Point(lon, lat)
    origin = Point(0.0, 0.0)
    dlat, dlon = degree_pad(lat, radius)
    inside: list[dict[str, Any]] = []
    nearest: list[dict[str, Any]] = []
    assessed = [k for k in HABITAT_KINDS if k in coverage["kinds_present"]]
    not_assessed = [k for k in HABITAT_KINDS if k not in coverage["kinds_present"]]
    terms: dict[str, Any] = {}

    for kind in assessed:
        best = None
        for layer in layers.by_kind(kind):
            if layer.regions() and region not in layer.regions():
                continue
            idx = layer.query_bbox(lat - dlat, lon - dlon, lat + dlat, lon + dlon)
            for i in np.asarray(idx, dtype=int):
                geom = layer.geometries[i]
                props = layer.properties[i]
                if geom.geom_type in ("Polygon", "MultiPolygon") and geom.covers(target):
                    inside.append(_record(layer, kind, props))
                    cand = (0.0, None, layer, props)
                else:
                    local = to_local(geom, lat, lon)
                    d_proj = local.distance(origin)
                    if d_proj > radius * 1.01:
                        continue
                    p_near = nearest_points(local, origin)[0]
                    near_ll = to_lonlat(p_near, lat, lon)
                    az, _, dist = GEOD.inv(lon, lat, near_ll.x, near_ll.y)
                    if dist > radius:
                        continue
                    cand = (float(dist), float(az % 360.0), layer, props)
                if best is None or cand[0] < best[0]:
                    best = cand
        sens = float(cfg.KIND_SENSITIVITY.get(kind, 0.0))
        if best is None:
            terms[kind] = {"sensitivity": sens, "distance_m": None, "value": 0.0,
                           "basis": f"no {kind} feature within {radius:.0f} m"}
            continue
        dist, bearing, layer, props = best
        rec = _record(layer, kind, props)
        rec["distance_m"] = round(dist, 1)
        rec["bearing_deg"] = None if bearing is None else round(bearing, 1)
        rec["inside"] = dist == 0.0 and bearing is None
        nearest.append(rec)
        value = sens * math.exp(-dist / cfg.HABITAT_DECAY_M)
        terms[kind] = {"sensitivity": sens, "distance_m": round(dist, 1), "value": round(value, 4),
                       "feature": rec["name"], "geometry_quality": rec["geometry_quality"]}

    prod = 1.0
    for t in terms.values():
        prod *= (1.0 - float(t["value"]))
    score = round(1.0 - prod, 4)
    nearest.sort(key=lambda r: r["distance_m"])
    notes = []
    if any((r.get("geometry_quality") or "").startswith("approximate") for r in nearest + inside):
        notes.append("one or more reported features have approximate geometry; see geometry_quality and construction")
    if "reef" in assessed:
        reef_sources = {l.meta.get("source_id") for l in layers.by_kind("reef") if region in l.regions()}
        detailed = {"wcmc_reefs", "allen_coral_atlas"} & reef_sources
        if not detailed:
            notes.append("neither the UNEP-WCMC nor the Allen Coral Atlas reef layer is present for this "
                         "region (run tools/fetch_ghosttrace_data.py, or import an Atlas download with "
                         "tools/import_allen_coral_atlas.py); reef results use the sparse OpenStreetMap "
                         "layer only")
        elif "allen_coral_atlas" not in reef_sources:
            notes.append("reef outlines come from UNEP-WCMC and OpenStreetMap; the Allen Coral Atlas "
                         "satellite map is not imported for this region "
                         "(tools/import_allen_coral_atlas.py)")
    return {**base, "covered": True, "region": region, "reason": None, "inside": inside, "nearest": nearest,
            "kinds_assessed": assessed, "kinds_not_assessed": not_assessed, "score": score, "terms": terms,
            "notes": notes}
