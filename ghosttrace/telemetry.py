"""Field kit telemetry: surface devices that close the loop on a GhostTrace target.

Three device types report over LoRa to a gateway, which posts JSON to
POST /telemetry/ingest (backend/app/routes/telemetry.py). This module owns
everything that is not HTTP: payload validation, the SQLite store, linking a
drifter tag to a target, the live forecast-versus-actual comparison, and the
recovery overlay that the rescue queue and change tracking read.

    drifter_tag   a GPS + LoRa float thrown in at a net (surface only). Position
                  fixes. A fix with event "deployed" and a target
                  {survey_id, detection_id} links the tag to that target and
                  starts a floating-mode drift forecast from the deployment
                  position and time, so every later fix can be compared with it.
    net_finder    a handheld GPS unit on the recovery boat. Position fixes and
                  the events "arrived" (on scene at the target) and "recovered"
                  (the crew has the net on board), each naming its target.
    nav_logger    a boat-mounted GPS + IMU. Position, heading, roll, pitch,
                  heave: the motion record a side-scan survey needs.

HONESTY RULES
    * A payload carrying "simulated": true is stored and served with that flag,
      and every view of it says SIMULATED DEVICE. Nothing turns a simulated fix
      into an observation.
    * A recovery is a report from a device, not a verified fact. The overlay
      keeps which device reported it, when, where it was, how far that was from
      the target, and whether the device was simulated.
    * GPS and LoRa do not work under water. Every device here is a surface device.

STORAGE
    data/telemetry/ (DEEPECHO_TELEMETRY_DIR), git-ignored by a .gitignore written
    into it on first use:
        telemetry.db      SQLite: fixes, links (drifter tag -> target + forecast)
        recoveries.json   the recovery overlay, rewritten atomically on each event
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

DEVICE_TYPES = ("drifter_tag", "net_finder", "nav_logger")
EVENTS: dict[str, tuple[str, ...]] = {
    "drifter_tag": ("deployed", "retrieved"),
    "net_finder": ("arrived", "recovered"),
    "nav_logger": (),
}
TARGET_EVENTS = ("deployed", "arrived", "recovered")

# (min, max) for optional numeric fields. Out of range is rejected, not clipped.
NUMERIC_FIELDS: dict[str, tuple[float, float]] = {
    "heading": (0.0, 360.0),
    "roll": (-180.0, 180.0),
    "pitch": (-90.0, 90.0),
    "heave": (-30.0, 30.0),
    "battery_v": (0.0, 30.0),
    "rssi": (-200.0, 0.0),
    "snr": (-50.0, 50.0),
    "hdop": (0.0, 100.0),
    "sats": (0.0, 64.0),
    "seq": (0.0, 4294967295.0),
}
KNOWN_KEYS = {"device_id", "device_type", "t", "lat", "lon", "event", "target", "simulated", "fw",
              *NUMERIC_FIELDS}

_DEVICE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}")
_SURVEY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_DETECTION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}")

FUTURE_TOLERANCE_S = float(os.environ.get("DEEPECHO_TELEMETRY_FUTURE_TOLERANCE_S", "300"))
FORECAST_HORIZON_H = float(os.environ.get("DEEPECHO_TELEMETRY_FORECAST_H", "72"))
FORECAST_PARTICLES = int(os.environ.get("DEEPECHO_TELEMETRY_FORECAST_PARTICLES", "300"))
FORECAST_SNAPSHOT_H = float(os.environ.get("DEEPECHO_TELEMETRY_FORECAST_SNAPSHOT_H", "2"))
# A recovery reported further than this from the target is kept, and flagged.
RECOVERY_FAR_M = float(os.environ.get("DEEPECHO_TELEMETRY_RECOVERY_FAR_M", "500"))


class TelemetryError(ValueError):
    """A payload that does not meet the contract. The message says which field and why."""


def telemetry_dir() -> Path:
    return Path(os.environ.get("DEEPECHO_TELEMETRY_DIR", str(REPO_ROOT / "data" / "telemetry")))


def iso(seconds: float) -> str:
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc).isoformat().replace("+00:00", "Z")


# --- validation ----------------------------------------------------------------------------------


def _number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise TelemetryError(f"{key} must be a finite number")
    return float(value)


def parse_time(value: Any, *, now_s: float | None = None) -> float:
    """ISO-8601 with a timezone -> epoch seconds. Naive times and far-future times are rejected."""
    if not isinstance(value, str) or not value.strip():
        raise TelemetryError("t must be an ISO-8601 time string, e.g. 2026-09-15T04:30:00Z")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        raise TelemetryError(f"t {value!r} is not ISO-8601")
    if parsed.tzinfo is None:
        raise TelemetryError("t must carry a timezone (use Z for UTC); a GPS time is UTC")
    seconds = parsed.timestamp()
    now_s = time.time() if now_s is None else now_s
    if seconds > now_s + FUTURE_TOLERANCE_S:
        raise TelemetryError(f"t {value} is more than {FUTURE_TOLERANCE_S:g} s in the future (device clock wrong?)")
    if parsed.year < 2000:
        raise TelemetryError(f"t {value} is before 2000 (GPS time not acquired?)")
    return seconds


def validate_payload(payload: Any, *, now_s: float | None = None) -> dict[str, Any]:
    """A clean record from one ingest payload, or TelemetryError."""
    if not isinstance(payload, dict):
        raise TelemetryError("payload must be a JSON object")
    unknown = sorted(set(payload) - KNOWN_KEYS)
    if unknown:
        raise TelemetryError(f"unknown field(s): {', '.join(unknown)}")
    device_id = payload.get("device_id")
    if not isinstance(device_id, str) or not _DEVICE_ID.fullmatch(device_id):
        raise TelemetryError("device_id must be 1-64 characters of letters, digits, . _ : - (starting alphanumeric)")
    device_type = payload.get("device_type")
    if device_type not in DEVICE_TYPES:
        raise TelemetryError(f"device_type must be one of {', '.join(DEVICE_TYPES)}")
    t_s = parse_time(payload.get("t"), now_s=now_s)
    lat, lon = _number(payload, "lat"), _number(payload, "lon")
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        raise TelemetryError("lat must be in [-90, 90] and lon in [-180, 180]")
    if lat == 0.0 and lon == 0.0:
        raise TelemetryError("lat/lon 0,0 is the no-fix value of most GPS modules; send a fix only when the GPS has one")
    record: dict[str, Any] = {"device_id": device_id, "device_type": device_type, "t": iso(t_s), "t_s": t_s,
                              "lat": lat, "lon": lon}
    for key, (lo, hi) in NUMERIC_FIELDS.items():
        if key in payload and payload[key] is not None:
            value = _number(payload, key)
            if not (lo <= value <= hi) or (key == "heading" and value == 360.0):
                raise TelemetryError(f"{key} {value:g} outside [{lo:g}, {hi:g}{')' if key == 'heading' else ']'}")
            record[key] = value
        else:
            record[key] = None
    simulated = payload.get("simulated", False)
    if not isinstance(simulated, bool):
        raise TelemetryError("simulated must be true or false")
    record["simulated"] = simulated
    fw = payload.get("fw")
    if fw is not None and (not isinstance(fw, str) or len(fw) > 32):
        raise TelemetryError("fw must be a string of at most 32 characters")
    record["fw"] = fw

    event = payload.get("event")
    target = payload.get("target")
    record["event"] = None
    record["survey_id"] = None
    record["detection_id"] = None
    if event is not None:
        if event not in EVENTS[device_type]:
            allowed = ", ".join(EVENTS[device_type]) or "none"
            raise TelemetryError(f"event {event!r} is not valid for {device_type} (allowed: {allowed})")
        record["event"] = event
    if target is not None:
        if not isinstance(target, dict) or set(target) - {"survey_id", "detection_id"}:
            raise TelemetryError("target must be an object {survey_id, detection_id}")
        sid, did = target.get("survey_id"), target.get("detection_id")
        if not isinstance(sid, str) or not _SURVEY_ID.fullmatch(sid) or ".." in sid:
            raise TelemetryError("target.survey_id is not a valid survey id")
        if not isinstance(did, str) or not _DETECTION_ID.fullmatch(did) or ".." in did:
            raise TelemetryError("target.detection_id is not a valid detection id")
        if event is None:
            raise TelemetryError("target is only accepted with an event (deployed, arrived or recovered)")
        record["survey_id"], record["detection_id"] = sid, did
    if record["event"] in TARGET_EVENTS and record["survey_id"] is None:
        raise TelemetryError(f"event {record['event']!r} needs target {{survey_id, detection_id}}")
    return record


# --- rate limiting ----------------------------------------------------------------------------------


class RateLimiter:
    """Token bucket per key: `per_minute` messages sustained, `burst` at once."""

    def __init__(self, per_minute: float, burst: float | None = None):
        self.rate = float(per_minute) / 60.0
        self.burst = float(burst if burst is not None else max(1.0, per_minute / 6.0))
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> tuple[bool, float]:
        """(allowed, seconds to wait before the next message would be allowed)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            tokens, last = self._buckets.get(key, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                return True, 0.0
            self._buckets[key] = (tokens, now)
            return False, (1.0 - tokens) / self.rate if self.rate > 0 else 60.0


# --- storage ---------------------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS fixes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL, device_type TEXT NOT NULL,
    t TEXT NOT NULL, t_s REAL NOT NULL, lat REAL NOT NULL, lon REAL NOT NULL,
    heading REAL, roll REAL, pitch REAL, heave REAL, battery_v REAL, rssi REAL, snr REAL,
    hdop REAL, sats REAL, seq REAL, fw TEXT,
    event TEXT, survey_id TEXT, detection_id TEXT,
    simulated INTEGER NOT NULL DEFAULT 0, received_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS fixes_device_t ON fixes (device_id, t_s);
CREATE TABLE IF NOT EXISTS links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL, survey_id TEXT NOT NULL, detection_id TEXT NOT NULL,
    deployed_t_s REAL NOT NULL, lat REAL NOT NULL, lon REAL NOT NULL, simulated INTEGER NOT NULL,
    forecast_status TEXT NOT NULL, forecast_json TEXT, forecast_reason TEXT,
    ended_t_s REAL, created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS links_survey ON links (survey_id);
"""

_FIX_COLUMNS = ("device_id", "device_type", "t", "t_s", "lat", "lon", "heading", "roll", "pitch", "heave",
                "battery_v", "rssi", "snr", "hdop", "sats", "seq", "fw", "event", "survey_id", "detection_id",
                "simulated")


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, allow_nan=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class TelemetryStore:
    """SQLite fixes and links plus the JSON recovery overlay, under one directory."""

    def __init__(self, directory: Path | None = None):
        self.dir = Path(directory) if directory is not None else telemetry_dir()
        self.dir.mkdir(parents=True, exist_ok=True)
        ignore = self.dir / ".gitignore"
        if not ignore.exists():
            ignore.write_text("# Field-kit telemetry is runtime data, not source.\n*\n", encoding="utf-8")
        self.db_path = self.dir / "telemetry.db"
        self.recoveries_path = self.dir / "recoveries.json"
        self._lock = threading.RLock()
        with self._connect() as con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=10)
        con.row_factory = sqlite3.Row
        return con

    # -- fixes ------------------------------------------------------------------------------------

    def insert_fix(self, record: dict[str, Any]) -> int:
        values = [record.get(c) for c in _FIX_COLUMNS]
        values[_FIX_COLUMNS.index("simulated")] = 1 if record.get("simulated") else 0
        with self._lock, self._connect() as con:
            cur = con.execute(
                f"INSERT INTO fixes ({', '.join(_FIX_COLUMNS)}, received_at) VALUES "
                f"({', '.join('?' for _ in _FIX_COLUMNS)}, ?)", [*values, time.time()])
            return int(cur.lastrowid)

    @staticmethod
    def _fix_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = {k: row[k] for k in row.keys()}
        d["simulated"] = bool(d.get("simulated"))
        return d

    def fixes_after(self, last_id: int, limit: int = 500) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute("SELECT * FROM fixes WHERE id > ? ORDER BY id LIMIT ?", (int(last_id), int(limit))).fetchall()
        return [self._fix_dict(r) for r in rows]

    def max_fix_id(self) -> int:
        with self._connect() as con:
            row = con.execute("SELECT COALESCE(MAX(id), 0) AS m FROM fixes").fetchone()
        return int(row["m"])

    def tracks(self, since_s: float | None = None, device_id: str | None = None,
               limit_per_device: int = 2000) -> list[dict[str, Any]]:
        query = "SELECT * FROM fixes WHERE 1=1"
        args: list[Any] = []
        if since_s is not None:
            query += " AND t_s >= ?"
            args.append(since_s)
        if device_id is not None:
            query += " AND device_id = ?"
            args.append(device_id)
        query += " ORDER BY device_id, t_s, id"
        with self._connect() as con:
            rows = con.execute(query, args).fetchall()
        by_dev: dict[str, dict[str, Any]] = {}
        for r in rows:
            d = self._fix_dict(r)
            entry = by_dev.setdefault(d["device_id"], {"device_id": d["device_id"], "device_type": d["device_type"],
                                                       "simulated": False, "points": []})
            entry["simulated"] = entry["simulated"] or d["simulated"]
            entry["points"].append({k: d[k] for k in ("id", "t", "lat", "lon", "heading", "roll", "pitch", "heave",
                                                      "battery_v", "rssi", "snr", "event", "simulated")})
        out = []
        for entry in by_dev.values():
            if len(entry["points"]) > limit_per_device:
                entry["truncated_from"] = len(entry["points"])
                entry["points"] = entry["points"][-limit_per_device:]
            out.append(entry)
        return out

    def devices(self) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT f.* , c.n AS n_fixes, c.first_t_s, c.any_sim FROM fixes f JOIN ("
                "  SELECT device_id, MAX(t_s) AS last_t_s, COUNT(*) AS n, MIN(t_s) AS first_t_s, MAX(simulated) AS any_sim"
                "  FROM fixes GROUP BY device_id) c ON f.device_id = c.device_id AND f.t_s = c.last_t_s "
                "ORDER BY f.device_id, f.id DESC").fetchall()
            links = {r["device_id"]: r for r in con.execute(
                "SELECT * FROM links WHERE id IN (SELECT MAX(id) FROM links GROUP BY device_id)").fetchall()}
        seen = set()
        out = []
        for r in rows:
            if r["device_id"] in seen:
                continue
            seen.add(r["device_id"])
            link = links.get(r["device_id"])
            out.append({
                "device_id": r["device_id"], "device_type": r["device_type"],
                "simulated": bool(r["any_sim"]), "fixes": int(r["n_fixes"]),
                "first_fix": iso(r["first_t_s"]),
                "last_fix": {k: r[k] for k in ("t", "lat", "lon", "heading", "battery_v", "rssi", "snr", "event")},
                "link": None if link is None else {"survey_id": link["survey_id"], "detection_id": link["detection_id"],
                                                   "deployed": iso(link["deployed_t_s"]),
                                                   "forecast_status": link["forecast_status"]},
            })
        return out

    # -- links ------------------------------------------------------------------------------------

    def create_link(self, record: dict[str, Any]) -> int:
        with self._lock, self._connect() as con:
            con.execute("UPDATE links SET ended_t_s = ? WHERE device_id = ? AND ended_t_s IS NULL",
                        (record["t_s"], record["device_id"]))
            cur = con.execute(
                "INSERT INTO links (device_id, survey_id, detection_id, deployed_t_s, lat, lon, simulated, "
                "forecast_status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
                (record["device_id"], record["survey_id"], record["detection_id"], record["t_s"], record["lat"],
                 record["lon"], 1 if record["simulated"] else 0, time.time()))
            return int(cur.lastrowid)

    def end_link(self, device_id: str, t_s: float) -> None:
        with self._lock, self._connect() as con:
            con.execute("UPDATE links SET ended_t_s = ? WHERE device_id = ? AND ended_t_s IS NULL", (t_s, device_id))

    def set_link_forecast(self, link_id: int, status: str, forecast: dict[str, Any] | None,
                          reason: str | None = None) -> None:
        with self._lock, self._connect() as con:
            con.execute("UPDATE links SET forecast_status = ?, forecast_json = ?, forecast_reason = ? WHERE id = ?",
                        (status, None if forecast is None else json.dumps(forecast, allow_nan=False), reason, link_id))

    def links_for_survey(self, survey_id: str) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute("SELECT * FROM links WHERE survey_id = ? ORDER BY id", (survey_id,)).fetchall()
        out = []
        for r in rows:
            d = {k: r[k] for k in r.keys()}
            d["simulated"] = bool(d["simulated"])
            d["forecast"] = json.loads(d.pop("forecast_json")) if d.get("forecast_json") else None
            out.append(d)
        return out

    def fixes_for_link(self, link: dict[str, Any]) -> list[dict[str, Any]]:
        query = "SELECT * FROM fixes WHERE device_id = ? AND t_s >= ?"
        args: list[Any] = [link["device_id"], link["deployed_t_s"]]
        if link.get("ended_t_s") is not None:
            query += " AND t_s <= ?"
            args.append(link["ended_t_s"])
        with self._connect() as con:
            rows = con.execute(query + " ORDER BY t_s, id", args).fetchall()
        return [self._fix_dict(r) for r in rows]

    # -- recoveries -------------------------------------------------------------------------------

    def record_target_event(self, record: dict[str, Any], target_position: tuple[float, float] | None) -> dict[str, Any]:
        """Update the overlay for a net_finder arrived/recovered event. Returns the overlay record."""
        with self._lock:
            overlay = load_recoveries(self.dir)
            key = f"{record['survey_id']}|{record['detection_id']}"
            existing = overlay.get(key) or {"survey_id": record["survey_id"], "detection_id": record["detection_id"],
                                            "status": None, "history": []}
            distance = None
            if target_position is not None:
                distance = round(_haversine_m(record["lat"], record["lon"], *target_position), 1)
            event = {"event": record["event"], "t": record["t"], "device_id": record["device_id"],
                     "device_type": record["device_type"], "lat": record["lat"], "lon": record["lon"],
                     "simulated": record["simulated"], "distance_to_target_m": distance,
                     "received_at": iso(time.time())}
            existing["history"] = (existing.get("history") or []) + [event]
            if record["event"] == "recovered":
                existing.update({"status": "recovered", "recovered_at": record["t"], "device_id": record["device_id"],
                                 "simulated": record["simulated"], "lat": record["lat"], "lon": record["lon"],
                                 "distance_to_target_m": distance})
            elif existing.get("status") != "recovered":
                existing.update({"status": "arrived", "arrived_at": record["t"], "device_id": record["device_id"],
                                 "simulated": record["simulated"], "distance_to_target_m": distance})
            flags = []
            if distance is None:
                flags.append("target position unknown to the server (no ghosttrace.json target); distance not checked")
            elif distance > RECOVERY_FAR_M:
                flags.append(f"reported {distance:.0f} m from the target position (> {RECOVERY_FAR_M:g} m): verify it "
                             "is the same object")
            if record["simulated"]:
                flags.append("SIMULATED DEVICE: this report did not come from real hardware")
            existing["flags"] = flags
            existing["basis"] = ("reported by field device over telemetry; a device report, not an independent "
                                 "verification of recovery")
            overlay[key] = existing
            _atomic_write_json(self.recoveries_path, {"format": "deepecho-ghosttrace-recoveries/1",
                                                      "updated": iso(time.time()),
                                                      "records": sorted(overlay.values(),
                                                                        key=lambda r: (r["survey_id"], r["detection_id"]))})
            return existing


def load_recoveries(directory: Path | None = None) -> dict[str, dict[str, Any]]:
    """The recovery overlay as {"survey_id|detection_id": record}; empty when none has been written."""
    path = Path(directory or telemetry_dir()) / "recoveries.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for rec in (data.get("records") if isinstance(data, dict) else None) or []:
        if isinstance(rec, dict) and rec.get("survey_id") and rec.get("detection_id"):
            out[f"{rec['survey_id']}|{rec['detection_id']}"] = rec
    return out


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


# --- forecast for a deployed drifter tag ------------------------------------------------------------


def _simplify(geom: dict[str, Any] | None, tol_deg: float = 0.0005) -> dict[str, Any] | None:
    if not geom:
        return None
    from shapely.geometry import mapping, shape

    g = shape(geom).simplify(tol_deg, preserve_topology=True)
    if g.is_empty:
        return None

    def rnd(c):
        if isinstance(c, (list, tuple)) and c and isinstance(c[0], (int, float)):
            return [round(float(c[0]), 5), round(float(c[1]), 5)]
        return [rnd(x) for x in c]

    m = mapping(g)
    return {"type": m["type"], "coordinates": rnd(m["coordinates"])}


def deployment_forecast(lat: float, lon: float, t_s: float, *, field: Any = None, layers: Any = None,
                        horizon_hours: float | None = None, n_particles: int | None = None,
                        snapshot_every_hours: float | None = None, seed: int = 0) -> dict[str, Any]:
    """A compact floating-mode forecast from a tag's deployment position and time.

    Keeps per snapshot only the ensemble mean and the simplified 50% / 90% cones,
    so a link row stays small.
    """
    from ghosttrace.drift import drift_forecast

    horizon = FORECAST_HORIZON_H if horizon_hours is None else float(horizon_hours)
    particles = FORECAST_PARTICLES if n_particles is None else int(n_particles)
    every = FORECAST_SNAPSHOT_H if snapshot_every_hours is None else float(snapshot_every_hours)
    fc = drift_forecast(lat, lon, t_s, mode="floating", horizon_hours=horizon, n_particles=particles,
                        dt_minutes=30, field=field, layers=layers, seed=seed, snapshot_every_hours=every,
                        max_snapshot_points=particles)
    if not fc.get("available"):
        return {"available": False, "reason": fc.get("reason")}
    snaps = []
    for s in fc["snapshots"]:
        pts = s.get("points") or []
        mean = [round(sum(p[0] for p in pts) / len(pts), 6), round(sum(p[1] for p in pts) / len(pts), 6)] if pts else None
        snaps.append({"t_hours": s["t_hours"], "mean": mean, "cone50": _simplify(s.get("cone50")),
                      "cone90": _simplify(s.get("cone90")), "counts": s.get("counts")})
    cs = fc.get("current_source") or {}
    return {
        "available": True, "mode": "floating", "start": iso(t_s), "horizon_hours": fc.get("effective_horizon_hours"),
        "n_particles": particles, "snapshot_every_hours": every, "time_mapping": fc.get("time_mapping"),
        "current_source": {"name": cs.get("name"), "time_start": cs.get("time_start"), "time_end": cs.get("time_end"),
                           "synthetic": cs.get("synthetic")},
        "k_surface_m2s": _k_used(fc), "snapshots": snaps,
        "assumptions": fc.get("assumptions"), "limitations": fc.get("limitations"),
    }


def _k_used(fc: dict[str, Any]) -> float | None:
    for text in fc.get("assumptions") or []:
        m = re.search(r"K = ([0-9.eE+-]+) m\^2/s", str(text))
        if m:
            return float(m.group(1))
    return None


# --- live comparison ---------------------------------------------------------------------------------


def compare_fix(forecast: dict[str, Any], elapsed_h: float, lat: float, lon: float) -> dict[str, Any]:
    """Forecast versus one actual position at `elapsed_h` hours after deployment.

    The mean is interpolated linearly between the bracketing snapshots; the
    cones are those of the nearest snapshot (their time is stated).
    """
    from ghosttrace.validation import point_in_geometry

    snaps = [s for s in (forecast or {}).get("snapshots") or [] if s.get("mean")]
    if not snaps:
        return {"available": False, "reason": "forecast has no snapshots"}
    last = snaps[-1]["t_hours"]
    if elapsed_h < 0:
        return {"available": False, "reason": "fix is earlier than the deployment"}
    if elapsed_h > last + 1e-9:
        return {"available": False, "reason": f"elapsed {elapsed_h:.1f} h is beyond the {last:g} h forecast horizon"}
    lo = max((s for s in snaps if s["t_hours"] <= elapsed_h), key=lambda s: s["t_hours"])
    hi = min((s for s in snaps if s["t_hours"] >= elapsed_h), key=lambda s: s["t_hours"])
    if hi["t_hours"] == lo["t_hours"]:
        mean = lo["mean"]
    else:
        a = (elapsed_h - lo["t_hours"]) / (hi["t_hours"] - lo["t_hours"])
        mean = [lo["mean"][0] + a * (hi["mean"][0] - lo["mean"][0]), lo["mean"][1] + a * (hi["mean"][1] - lo["mean"][1])]
    nearest = min(snaps, key=lambda s: (abs(s["t_hours"] - elapsed_h), -s["t_hours"]))
    cone_t = nearest["t_hours"]
    has_cone = nearest.get("cone90") is not None
    return {
        "available": True, "elapsed_hours": round(elapsed_h, 3),
        "forecast_mean": [round(mean[0], 6), round(mean[1], 6)],
        "separation_km": round(_haversine_m(lat, lon, mean[0], mean[1]) / 1000.0, 3),
        "cone_t_hours": cone_t,
        "inside_cone50": point_in_geometry(nearest.get("cone50"), lat, lon) if has_cone else None,
        "inside_cone90": point_in_geometry(nearest.get("cone90"), lat, lon) if has_cone else None,
        "cone_note": None if has_cone else "no cone at this time (t = 0 or a single-position cloud)",
    }


def link_view(store: TelemetryStore, link: dict[str, Any], *, max_series: int = 120) -> dict[str, Any]:
    """One linked drifter tag with its track, the forecast to draw, and live forecast-vs-actual stats."""
    fixes = store.fixes_for_link(link)
    forecast = link.get("forecast")
    series = []
    latest = None
    if forecast and forecast.get("available"):
        step = max(1, len(fixes) // max_series)
        picked = fixes[::step]
        if fixes and picked[-1] is not fixes[-1]:
            picked.append(fixes[-1])
        for f in picked:
            cmp = compare_fix(forecast, (f["t_s"] - link["deployed_t_s"]) / 3600.0, f["lat"], f["lon"])
            if cmp.get("available"):
                series.append({"t": f["t"], "elapsed_hours": cmp["elapsed_hours"], "separation_km": cmp["separation_km"],
                               "inside_cone50": cmp["inside_cone50"], "inside_cone90": cmp["inside_cone90"]})
        if fixes:
            f = fixes[-1]
            latest = compare_fix(forecast, (f["t_s"] - link["deployed_t_s"]) / 3600.0, f["lat"], f["lon"])
    cone_snapshot = None
    if forecast and forecast.get("available") and latest and latest.get("available"):
        cone_snapshot = next((s for s in forecast["snapshots"] if s["t_hours"] == latest["cone_t_hours"]), None)
    simulated = bool(link["simulated"]) or any(f["simulated"] for f in fixes)
    return {
        "device_id": link["device_id"], "device_type": "drifter_tag",
        "survey_id": link["survey_id"], "detection_id": link["detection_id"],
        "deployed": iso(link["deployed_t_s"]), "deployed_position": [link["lat"], link["lon"]],
        "ended": None if link.get("ended_t_s") is None else iso(link["ended_t_s"]),
        "simulated": simulated,
        "track": [{"t": f["t"], "lat": f["lat"], "lon": f["lon"], "battery_v": f["battery_v"], "rssi": f["rssi"],
                   "snr": f["snr"]} for f in fixes],
        "forecast_status": link["forecast_status"], "forecast_reason": link.get("forecast_reason"),
        "forecast_meta": None if not forecast else {k: forecast.get(k) for k in (
            "available", "mode", "start", "horizon_hours", "n_particles", "snapshot_every_hours", "time_mapping",
            "current_source", "k_surface_m2s", "limitations", "reason")},
        "forecast_mean_track": [] if not (forecast and forecast.get("available")) else
        [{"t_hours": s["t_hours"], "mean": s["mean"]} for s in forecast["snapshots"] if s.get("mean")],
        "cone_now": cone_snapshot and {"t_hours": cone_snapshot["t_hours"], "cone50": cone_snapshot["cone50"],
                                       "cone90": cone_snapshot["cone90"]},
        "latest": latest, "series": series,
        "notes": [
            "Forecast: GhostTrace floating drift from the tag's deployment position and time over the bundled "
            "HYCOM surface currents (no wind). Separation is from the ensemble mean at the same elapsed time.",
            "Cone coverage of a single tag is one sample; the 'Model trust' validation reports it over many real drifters.",
        ] + (["SIMULATED DEVICE: positions come from tools/sim_field_devices.py, not from hardware."] if simulated else []),
    }
