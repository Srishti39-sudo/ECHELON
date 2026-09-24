"""Simulate the GhostTrace field kit against a running backend. Every payload says "simulated": true.

    .venv/bin/python tools/sim_field_devices.py --api http://127.0.0.1:8000 \\
        --survey demo-ghosttrace-mannar --speedup 120

Three simulated devices post to POST /telemetry/ingest in time order:

    SIM-TAG-01   drifter_tag thrown in at the drifter target. First fix carries
                 event "deployed" and the target, which makes the server link it
                 and start a floating drift forecast. It then MOVES WITH THE
                 BUNDLED HYCOM SURFACE CURRENTS (data/ghosttrace/currents) plus
                 noise: a random walk (--tag-k m^2/s) and a slowly varying
                 velocity error (Ornstein-Uhlenbeck, --tag-noise m/s, 6 h
                 memory) standing in for wind and unresolved flow. Land in the
                 bundled layer stops it.
    SIM-FINDER-01 net_finder on the recovery boat. Leaves the survey's recovery
                 start harbour, runs a straight line to the finder target at
                 --boat-speed, sends "arrived" within 50 m, holds while the crew
                 works, then sends "recovered".
    SIM-NAV-01   nav_logger on the same boat: position, heading and a synthetic
                 roll / pitch / heave record (sinusoids plus noise).

WHAT THIS PROVES AND WHAT IT DOES NOT
    It exercises the whole pipeline (ingest, link, live forecast-vs-actual,
    recovery overlay, rescue queue, change tracking) with hardware-shaped data.
    Because the simulated tag moves with the SAME model currents the forecast
    uses, its agreement with the cone demonstrates the plumbing, not the model's
    skill; the "Model trust" validation against real NOAA drifters does that.
    Nothing here is an observation.

TIME
    Simulated time runs --speedup times faster than the wall clock and, by
    default, ends at the moment the simulator was started (so no fix is in the
    future). --start puts the run anywhere; it must lie inside the bundled
    current window, and the simulator refuses otherwise rather than invent
    currents.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ghosttrace.currents import iso, load_default_field  # noqa: E402
from ghosttrace.layers import load_default_layers  # noqa: E402

SURVEYS = Path(os.environ.get("DEEPECHO_SURVEYS_DIR", str(REPO / "data" / "surveys")))
R_EARTH = 6371008.8


def log(msg: str) -> None:
    print(f"[sim {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH * math.asin(min(1.0, math.sqrt(h)))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def move(lat, lon, east_m, north_m):
    return (lat + math.degrees(north_m / R_EARTH),
            lon + math.degrees(east_m / (R_EARTH * math.cos(math.radians(lat)))))


def radio(lat, lon, gw_lat, gw_lon, rng) -> tuple[float, float]:
    """Plausible LoRa RSSI / SNR from distance to the gateway (log-distance path loss). Simulated."""
    d_km = max(haversine_m(lat, lon, gw_lat, gw_lon) / 1000.0, 0.05)
    rssi = -60.0 - 25.0 * math.log10(d_km * 10.0) + rng.normal(0, 2.0)
    snr = 10.0 - 0.45 * (-(rssi) - 80.0) + rng.normal(0, 1.0)
    return round(max(-139.0, min(-30.0, rssi)), 1), round(max(-20.0, min(15.0, snr)), 1)


# --- scenario ----------------------------------------------------------------------------------


def load_targets(survey: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = SURVEYS / survey / "ghosttrace.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    targets = [t for t in doc.get("targets") or [] if t.get("latitude") is not None and t.get("longitude") is not None]
    targets.sort(key=lambda t: (t.get("priority") or {}).get("rank") or 1e9)
    if not targets:
        raise SystemExit(f"survey {survey} has no located GhostTrace targets")
    return doc, targets


def pick(targets, wanted: str | None, default_index: int) -> dict[str, Any]:
    if wanted:
        for t in targets:
            if t["detection_id"] == wanted:
                return t
        raise SystemExit(f"no target {wanted!r}; located targets: {', '.join(t['detection_id'] for t in targets)}")
    return targets[min(default_index, len(targets) - 1)]


def drifter_track(target, start_s, hours, interval_s, *, field, layers, k_m2s, noise_mps, seed) -> list[dict[str, Any]]:
    """Positions every interval_s from the target, advected by bundled surface currents plus noise."""
    f = field.field_for(target["latitude"], target["longitude"])
    if f is None:
        raise SystemExit("no bundled current field covers the drifter target; not simulating (no synthetic currents)")
    end_s = start_s + hours * 3600.0
    if start_s < f.times_s[0] or end_s > f.times_s[-1]:
        raise SystemExit(f"simulated window {iso(start_s)}..{iso(end_s)} is outside the bundled currents "
                         f"{iso(f.times_s[0])}..{iso(f.times_s[-1])}; choose --start/--hours inside it")
    region = layers.region_for(target["latitude"], target["longitude"])
    land = layers.land(region["region"]) if region.get("covered") else None
    rng = np.random.default_rng(seed)
    lat, lon = move(target["latitude"], target["longitude"], rng.normal(0, 8), rng.normal(0, 8))
    dt = 300.0
    tau = 6 * 3600.0
    eu = ev = 0.0
    out = [{"t_s": start_s, "lat": lat, "lon": lon}]
    t = start_s
    next_fix = start_s + interval_s
    stranded = False
    import shapely

    while t < end_s - 1e-6:
        if not stranded:
            u, v = f.sample(lat, lon, t, "surface")
            if not (math.isfinite(u) and math.isfinite(v)):
                u = v = 0.0
            a = math.exp(-dt / tau)
            eu = a * eu + math.sqrt(1 - a * a) * noise_mps * rng.normal()
            ev = a * ev + math.sqrt(1 - a * a) * noise_mps * rng.normal()
            sigma = math.sqrt(2 * k_m2s * dt)
            nlat, nlon = move(lat, lon, (u + eu) * dt + rng.normal(0, sigma), (v + ev) * dt + rng.normal(0, sigma))
            if land is not None and bool(shapely.contains_xy(land, nlon, nlat)):
                stranded = True
            else:
                lat, lon = nlat, nlon
        t += dt
        if t >= next_fix - 1e-6:
            out.append({"t_s": next_fix, "lat": lat, "lon": lon})
            next_fix += interval_s
    return out


def boat_track(start, target, depart_s, speed_mps, interval_s, hold_s, rng) -> tuple[list[dict[str, Any]], float, float]:
    """(fixes, arrived_s, recovered_s) for a straight run harbour -> target, a hold, then done."""
    total = haversine_m(start["latitude"], start["longitude"], target["latitude"], target["longitude"])
    brg = bearing_deg(start["latitude"], start["longitude"], target["latitude"], target["longitude"])
    travel_s = total / speed_mps
    fixes = []
    t = depart_s
    arrived_s = depart_s + travel_s
    while t < arrived_s:
        frac = (t - depart_s) / travel_s
        east = math.sin(math.radians(brg)) * total * frac
        north = math.cos(math.radians(brg)) * total * frac
        lat, lon = move(start["latitude"], start["longitude"], east + rng.normal(0, 3), north + rng.normal(0, 3))
        fixes.append({"t_s": t, "lat": lat, "lon": lon, "heading": brg, "moving": True})
        t += interval_s
    lat, lon = move(target["latitude"], target["longitude"], rng.normal(0, 6), rng.normal(0, 6))
    fixes.append({"t_s": arrived_s, "lat": lat, "lon": lon, "heading": brg, "moving": False, "event": "arrived"})
    t = arrived_s + interval_s
    recovered_s = arrived_s + hold_s
    while t < recovered_s:
        lat, lon = move(target["latitude"], target["longitude"], rng.normal(0, 12), rng.normal(0, 12))
        fixes.append({"t_s": t, "lat": lat, "lon": lon, "heading": (brg + rng.normal(0, 40)) % 360, "moving": False})
        t += interval_s
    lat, lon = move(target["latitude"], target["longitude"], rng.normal(0, 6), rng.normal(0, 6))
    fixes.append({"t_s": recovered_s, "lat": lat, "lon": lon, "heading": brg, "moving": False, "event": "recovered"})
    return fixes, arrived_s, recovered_s


def build_payloads(args) -> list[tuple[float, dict[str, Any]]]:
    doc, targets = load_targets(args.survey)
    tag_target = pick(targets, args.drifter_target, 0)
    finder_target = pick(targets, args.finder_target, 1)
    plan_start = (doc.get("recovery_plan") or {}).get("start")
    if not plan_start or plan_start.get("latitude") is None:
        raise SystemExit("the survey's recovery plan has no start harbour to leave from")
    now = time.time()
    duration_s = max(args.hours * 3600.0, 0.0)
    start_s = (datetime.fromisoformat(args.start.replace("Z", "+00:00")).timestamp() if args.start
               else now - duration_s)
    if start_s + duration_s > now + 60:
        raise SystemExit("the simulated run would end in the future; move --start earlier")
    rng = np.random.default_rng(args.seed)
    field = load_default_field()
    if field is None:
        raise SystemExit("no bundled currents (run tools/fetch_ghosttrace_data.py --only currents); not simulating")
    layers = load_default_layers()
    tag = drifter_track(tag_target, start_s, args.hours, args.tag_interval_min * 60.0, field=field, layers=layers,
                        k_m2s=args.tag_k, noise_mps=args.tag_noise, seed=args.seed)
    gw = (plan_start["latitude"], plan_start["longitude"])
    out: list[tuple[float, dict[str, Any]]] = []
    target_ref = lambda t: {"survey_id": args.survey, "detection_id": t["detection_id"]}  # noqa: E731
    for i, fix in enumerate(tag):
        rssi, snr = radio(fix["lat"], fix["lon"], *gw, rng)
        p = {"device_id": "SIM-TAG-01", "device_type": "drifter_tag", "t": iso(fix["t_s"]),
             "lat": round(fix["lat"], 6), "lon": round(fix["lon"], 6),
             "battery_v": round(4.15 - 0.2 * i / max(1, len(tag) - 1), 3), "rssi": rssi, "snr": snr,
             "seq": i, "fw": "sim-0.1", "simulated": True}
        if i == 0:
            p["event"] = "deployed"
            p["target"] = target_ref(tag_target)
        out.append((fix["t_s"], p))

    depart_s = start_s + args.finder_delay_h * 3600.0
    boat, arrived_s, recovered_s = boat_track(plan_start, finder_target, depart_s, args.boat_speed,
                                              args.finder_interval_min * 60.0, args.hold_min * 60.0, rng)
    if recovered_s > start_s + duration_s:
        log(f"note: the recovery at {iso(recovered_s)} falls after the drifter's last fix; the run is extended")
    for i, fix in enumerate(boat):
        rssi, snr = radio(fix["lat"], fix["lon"], *gw, rng)
        p = {"device_id": "SIM-FINDER-01", "device_type": "net_finder", "t": iso(fix["t_s"]),
             "lat": round(fix["lat"], 6), "lon": round(fix["lon"], 6), "heading": round(fix["heading"] % 360, 1),
             "battery_v": round(3.95 - 0.1 * i / max(1, len(boat) - 1), 3), "rssi": rssi, "snr": snr, "seq": i,
             "fw": "sim-0.1", "simulated": True}
        if fix.get("event"):
            p["event"] = fix["event"]
            p["target"] = target_ref(finder_target)
        out.append((fix["t_s"], p))
        if args.with_nav:
            phase = fix["t_s"] / 7.0
            sea = 1.0 if fix["moving"] else 0.6
            n = {"device_id": "SIM-NAV-01", "device_type": "nav_logger", "t": iso(fix["t_s"]),
                 "lat": p["lat"], "lon": p["lon"], "heading": p["heading"],
                 "roll": round(sea * 6.0 * math.sin(phase) + rng.normal(0, 0.8), 2),
                 "pitch": round(sea * 2.5 * math.sin(phase * 0.7 + 1.0) + rng.normal(0, 0.4), 2),
                 "heave": round(sea * 0.4 * math.sin(phase * 1.3) + rng.normal(0, 0.05), 3),
                 "battery_v": 12.6, "seq": i, "fw": "sim-0.1", "simulated": True}
            out.append((fix["t_s"] + 0.001, n))
    out.sort(key=lambda x: x[0])
    latest_allowed = time.time() + 60
    future = [p for s, p in out if s > latest_allowed]
    if future:
        raise SystemExit(f"{len(future)} payload(s) would be in the future (last {future[-1]['t']}); move --start earlier "
                         "or shorten --hours / --hold-min")
    log(f"scenario: survey {args.survey}; tag on {tag_target['detection_id']} ({len(tag)} fixes); finder to "
        f"{finder_target['detection_id']} from {plan_start.get('name')} ({len(boat)} fixes, arrives {iso(arrived_s)}, "
        f"recovers {iso(recovered_s)}); simulated time {iso(out[0][0])} .. {iso(out[-1][0])}")
    return out


def post(api: str, payload: dict[str, Any], token: str | None) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-DeepEcho-Token"] = token
    for attempt in range(6):
        r = requests.post(f"{api.rstrip('/')}/telemetry/ingest", data=json.dumps(payload), headers=headers, timeout=30)
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", "2"))
            log(f"rate limited; waiting {wait:g} s")
            time.sleep(wait)
            continue
        if r.status_code >= 400:
            raise SystemExit(f"ingest rejected ({r.status_code}): {r.text[:300]}")
        return r.json()
    raise SystemExit("ingest kept rate limiting; giving up")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--api", default=os.environ.get("VITE_API_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--survey", default="demo-ghosttrace-mannar")
    ap.add_argument("--drifter-target", default=None, help="detection id (default: rank 1)")
    ap.add_argument("--finder-target", default=None, help="detection id (default: rank 2, else rank 1)")
    ap.add_argument("--start", default=None, help="simulated start, ISO-8601 UTC (default: now minus --hours)")
    ap.add_argument("--hours", type=float, default=24.0, help="simulated drifter duration")
    ap.add_argument("--tag-interval-min", type=float, default=15.0)
    ap.add_argument("--finder-interval-min", type=float, default=5.0)
    ap.add_argument("--finder-delay-h", type=float, default=2.0, help="boat departs this long after the tag deploys")
    ap.add_argument("--boat-speed", type=float, default=4.0, help="m/s (about 8 knots)")
    ap.add_argument("--hold-min", type=float, default=40.0, help="minutes on scene before 'recovered'")
    ap.add_argument("--tag-k", type=float, default=5.0, help="random-walk diffusivity for the simulated tag, m^2/s")
    ap.add_argument("--tag-noise", type=float, default=0.04, help="velocity error std for the simulated tag, m/s")
    ap.add_argument("--no-nav", dest="with_nav", action="store_false")
    ap.add_argument("--speedup", type=float, default=60.0, help="simulated seconds per wall second; 0 = no waiting")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--token", default=os.environ.get("DEEPECHO_TELEMETRY_TOKEN"))
    ap.add_argument("--dry-run", action="store_true", help="print payloads instead of posting")
    args = ap.parse_args(argv)

    payloads = build_payloads(args)
    if args.dry_run:
        for _, p in payloads:
            print(json.dumps(p))
        return 0
    t_sim0 = payloads[0][0]
    wall0 = time.monotonic()
    for n, (t_sim, p) in enumerate(payloads):
        if args.speedup > 0:
            due = wall0 + (t_sim - t_sim0) / args.speedup
            delay = due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        res = post(args.api, p, args.token)
        if p.get("event"):
            log(f"{p['device_id']} {p['event']} {p['target']['detection_id']} at {p['t']} -> "
                f"{json.dumps({k: res.get(k) for k in ('link', 'recovery') if k in res})[:240]}")
        elif n % 25 == 0:
            log(f"posted {n + 1}/{len(payloads)} (sim {p['t']})")
    log(f"done: {len(payloads)} simulated payloads posted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
