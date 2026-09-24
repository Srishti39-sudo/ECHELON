"""In what order should the recovery vessel visit the targets?

A priority rank alone makes a bad voyage plan: the most urgent net and the
second most urgent can be 40 km apart with a routine net sitting between them.
But a pure shortest-tour ignores urgency entirely. This module does both, in
that order of importance:

    1. Tiers are visited strictly in order: every urgent target, then every
       high, then every routine. Urgency is never traded for distance.
    2. Within a tier, a nearest-neighbour tour from wherever the vessel is
       (the start, or the last target of the previous tier), improved by 2-opt
       on the open path with its first point fixed.

START
    The nearest harbour to the targets' centroid when the geo agent's layers
    expose harbours (name, latitude, longitude, source). Otherwise the route
    starts at the first target in the visiting order, and `start.source` says
    so. A harbour is never invented.

DISTANCES
    Geodesic (haversine) straight lines. Sea routes around land, shoals,
    exclusion zones and traffic separation schemes are NOT computed, so a leg
    can cross a headland. `notes` says this on every plan. The route does not
    return to the start; add the return leg by hand if the vessel must.

Targets with no position cannot be routed and are listed in `notes`.
"""

from __future__ import annotations

from typing import Any

from ghosttrace import config_core as cfg
from ghosttrace.changes import haversine_m

TIER_ORDER = [name for name, _ in cfg.PRIORITY_TIERS]


def _km(a: tuple[float, float], b: tuple[float, float]) -> float:
    return haversine_m(a[0], a[1], b[0], b[1]) / 1000.0


def _path_len(start: tuple[float, float] | None, pts: list[tuple[float, float]]) -> float:
    seq = ([start] if start else []) + pts
    return sum(_km(seq[i], seq[i + 1]) for i in range(len(seq) - 1))


def _nearest_neighbour(start: tuple[float, float] | None,
                       items: list[tuple[str, tuple[float, float]]]) -> list[tuple[str, tuple[float, float]]]:
    remaining = sorted(items, key=lambda it: it[0])
    if not remaining:
        return []
    out = []
    here = start
    if here is None:
        first = remaining.pop(0)
        out.append(first)
        here = first[1]
    while remaining:
        j = min(range(len(remaining)), key=lambda k: (_km(here, remaining[k][1]), remaining[k][0]))
        nxt = remaining.pop(j)
        out.append(nxt)
        here = nxt[1]
    return out


def _two_opt(start: tuple[float, float] | None,
             tour: list[tuple[str, tuple[float, float]]]) -> list[tuple[str, tuple[float, float]]]:
    """2-opt on an open path. With no start, the first point stays first."""
    # Work on indices into a distance matrix; node 0 is the start when given.
    # Reversing positions i..k of an open path changes only the edge into i
    # and the edge out of k (none out of k when k is last), so each candidate
    # is an O(1) delta rather than a full path length.
    nodes = ([start] if start is not None else []) + [p for _, p in tour]
    n = len(nodes)
    if n < 3:
        return list(tour)
    dist = [[_km(a, b) for b in nodes] for a in nodes]
    path = list(range(n))
    improved, passes = True, 0
    while improved and passes < 100:
        improved, passes = False, passes + 1
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                a, b = path[i - 1], path[i]
                c = path[k]
                d = path[k + 1] if k + 1 < n else None
                before = dist[a][b] + (dist[c][d] if d is not None else 0.0)
                after = dist[a][c] + (dist[b][d] if d is not None else 0.0)
                if after + 1e-9 < before:
                    path[i:k + 1] = reversed(path[i:k + 1])
                    improved = True
    offset = 1 if start is not None else 0
    return [tour[j - offset] for j in path[offset:]]


def plan_route(targets: list[dict[str, Any]], harbours: list[dict[str, Any]] | None = None
               ) -> dict[str, Any]:
    """recovery_plan block. targets need detection_id, latitude, longitude, priority.tier."""
    notes = ["Legs are straight geodesic lines; sea routes around land, shoals and "
             "exclusion zones are not computed.",
             "The route ends at the last target and does not return to the start."]
    routable = [t for t in targets if t.get("latitude") is not None and t.get("longitude") is not None]
    missing = [t["detection_id"] for t in targets if t not in routable]
    if missing:
        notes.append("Not routed (no position): " + ", ".join(missing))

    method = ("tiers in order (urgent, high, routine); within a tier nearest-neighbour from the "
              "current position then 2-opt on the open path; geodesic straight-line legs")
    if not routable:
        return {"start": None, "order": [], "legs": [], "total_km": 0.0, "method": method,
                "notes": notes + ["No target has a position, so no route was planned."]}

    start = None
    usable_harbours = [h for h in (harbours or [])
                       if h.get("latitude") is not None and h.get("longitude") is not None]
    if usable_harbours:
        clat = sum(float(t["latitude"]) for t in routable) / len(routable)
        clon = sum(float(t["longitude"]) for t in routable) / len(routable)
        h = min(usable_harbours, key=lambda h: (_km((clat, clon), (float(h["latitude"]),
                                                                    float(h["longitude"]))),
                                                str(h.get("name"))))
        # Many mapped harbours carry no name. The position is real; the label
        # says the name is missing rather than leaving a blank or inventing one.
        name = h.get("name") or (f"unnamed harbour ({float(h['latitude']):.4f}, "
                                 f"{float(h['longitude']):.4f})")
        start = {"name": name, "latitude": float(h["latitude"]),
                 "longitude": float(h["longitude"]),
                 "source": h.get("source") or "harbour layer (source not stated by the layer)"}
    else:
        notes.append("No harbour layer was available; the route starts at the first target.")

    here = (start["latitude"], start["longitude"]) if start else None
    order: list[tuple[str, tuple[float, float]]] = []
    for tier in TIER_ORDER:
        items = [(t["detection_id"], (float(t["latitude"]), float(t["longitude"])))
                 for t in routable if (t.get("priority") or {}).get("tier") == tier]
        if not items:
            continue
        tour = _two_opt(here, _nearest_neighbour(here, items))
        order.extend(tour)
        here = tour[-1][1]
    # Anything with an unrecognised tier goes last, in id order.
    seen = {i for i, _ in order}
    order.extend(sorted(((t["detection_id"], (float(t["latitude"]), float(t["longitude"])))
                         for t in routable if t["detection_id"] not in seen), key=lambda x: x[0]))

    if start is None:
        first_id, first_pos = order[0]
        start = {"name": f"target {first_id}", "latitude": first_pos[0], "longitude": first_pos[1],
                 "source": "first target in visiting order (no harbour layer available)"}
        prev_name, prev_pos = first_id, first_pos
        rest = order[1:]
    else:
        prev_name, prev_pos = start["name"], (start["latitude"], start["longitude"])
        rest = order

    legs = []
    for ident, pos in rest:
        legs.append({"from": prev_name, "to": ident, "distance_km": round(_km(prev_pos, pos), 3)})
        prev_name, prev_pos = ident, pos
    return {"start": start, "order": [i for i, _ in order], "legs": legs,
            "total_km": round(sum(l["distance_km"] for l in legs), 3), "method": method,
            "notes": notes}
