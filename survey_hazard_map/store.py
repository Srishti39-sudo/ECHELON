"""Where scans and detections are kept, and the two places they can be kept.

The routes used to call Supabase directly, so every one of them returned 503
without credentials: no history, no stats, no hazard map, and an upload that ran
the detector and then threw the result away. That is a heavy price for a service
this application does not need in order to be useful.

So storage is an interface with two implementations behind it. SQLite is the
default and needs nothing at all: a file under data/ and a directory for the
uploaded tiles. Supabase is used instead the moment SUPABASE_URL and SUPABASE_KEY
are set, with the same method names and the same row shapes, so nothing upstream
changes when you switch.

The row vocabulary is Supabase's, because those tables already existed and the
existing frontend contract was written against them. SQLite matches it column
for column rather than inventing a second spelling of the same record.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend import config
from survey_hazard_map.supabase_client import supabase_ready, supabase

# Uploaded tiles and the SQLite file live beside the surveys, under data/.
# DEEPECHO_DB_PATH and DEEPECHO_UPLOAD_DIR move both elsewhere, so a test run
# or a throwaway server never writes into the operator's own history.
DATA_DIR = config.ROOT / "data"
UPLOAD_DIR = Path(os.environ.get("DEEPECHO_UPLOAD_DIR") or DATA_DIR / "uploads")
DB_PATH = Path(os.environ.get("DEEPECHO_DB_PATH") or DATA_DIR / "deepecho.db")

# The bucket Supabase uploads go to. Unused by the SQLite backend, which writes
# the bytes to UPLOAD_DIR instead.
BUCKET = "sonar-images"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- SQLite ----------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id          TEXT PRIMARY KEY,
    image_url   TEXT,
    filename    TEXT,
    content_type TEXT,
    bytes       INTEGER,
    latitude    REAL,
    longitude   REAL,
    status      TEXT,
    stub        INTEGER DEFAULT 0,
    models      TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS detections (
    id           TEXT PRIMARY KEY,
    scan_id      TEXT NOT NULL,
    object_class TEXT,
    confidence   REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    anomaly      INTEGER DEFAULT 0,
    severity     TEXT,
    record       TEXT,
    created_at   TEXT NOT NULL,
    FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS detections_scan_id ON detections(scan_id);
CREATE INDEX IF NOT EXISTS scans_created_at ON scans(created_at DESC);
"""


class SqliteStore:
    """The no-credentials default. One file, and the tiles beside it.

    FastAPI runs sync route handlers in a threadpool, so the connection is
    shared with check_same_thread off and every write goes through one lock.
    The volumes here are an operator uploading tiles by hand, so a single
    writer is not a constraint worth engineering around.
    """

    kind = "sqlite"

    def __init__(self, db_path: Path = DB_PATH, upload_dir: Path = UPLOAD_DIR):
        self.db_path = db_path
        self.upload_dir = upload_dir
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def status(self) -> str:
        try:
            where = self.db_path.relative_to(config.ROOT)
        except ValueError:
            where = self.db_path
        return f"local sqlite at {where}"

    # -- images --

    def put_image(self, path: str, data: bytes, content_type: str) -> str:
        """Write the tile and return the path a client can fetch it back by."""
        target = self.upload_dir / Path(path).name
        target.write_bytes(data)
        return target.name

    def open_image(self, path: str) -> Path | None:
        """The file on disk for a stored tile, or None if it is not there."""
        target = self.upload_dir / Path(path).name
        return target if target.is_file() else None

    # -- scans --

    def create_scan(self, scan: dict[str, Any]) -> dict[str, Any]:
        row = {
            "id": scan["id"],
            "image_url": scan.get("image_url"),
            "filename": scan.get("filename"),
            "content_type": scan.get("content_type"),
            "bytes": scan.get("bytes"),
            "latitude": scan.get("latitude"),
            "longitude": scan.get("longitude"),
            "status": scan.get("status", "analysed"),
            "stub": 1 if scan.get("stub") else 0,
            "models": json.dumps(scan.get("models") or []),
            "created_at": scan.get("created_at") or _now(),
        }
        with self._lock:
            self._conn.execute(
                "INSERT INTO scans (id, image_url, filename, content_type, bytes, "
                "latitude, longitude, status, stub, models, created_at) VALUES "
                "(:id, :image_url, :filename, :content_type, :bytes, :latitude, "
                ":longitude, :status, :stub, :models, :created_at)", row)
            self._conn.commit()
        return self._scan_out(row)

    def list_scans(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM scans ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._scan_out(dict(r)) for r in rows]

    def get_scan(self, scan_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
        return self._scan_out(dict(row)) if row else None

    def delete_scan(self, scan_id: str) -> bool:
        scan = self.get_scan(scan_id)
        if scan is None:
            return False
        if scan.get("image_url"):
            target = self.open_image(scan["image_url"])
            if target:
                target.unlink(missing_ok=True)
        with self._lock:
            self._conn.execute("DELETE FROM detections WHERE scan_id = ?", (scan_id,))
            self._conn.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
            self._conn.commit()
        return True

    # -- detections --

    def insert_detections(self, scan_id: str, rows: list[dict[str, Any]]) -> list[dict]:
        if not rows:
            return []
        payload = []
        for d in rows:
            bbox = d.get("bbox") or [0, 0, 0, 0]
            payload.append({
                "id": d["id"],
                "scan_id": scan_id,
                "object_class": d.get("class") or d.get("object_class"),
                "confidence": d.get("confidence"),
                "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3],
                "anomaly": 1 if d.get("anomaly") else 0,
                "severity": d.get("severity"),
                # The full detector record, provenance and any withheld class
                # included. The flat columns are what queries use; this is what
                # the assistant is handed when the operator asks about one.
                "record": json.dumps(d.get("record") or {}),
                "created_at": _now(),
            })
        with self._lock:
            self._conn.executemany(
                "INSERT INTO detections (id, scan_id, object_class, confidence, "
                "x1, y1, x2, y2, anomaly, severity, record, created_at) VALUES "
                "(:id, :scan_id, :object_class, :confidence, :x1, :y1, :x2, :y2, "
                ":anomaly, :severity, :record, :created_at)", payload)
            self._conn.commit()
        return [self._detection_out(p) for p in payload]

    def list_detections(self, scan_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM detections"
        args: tuple = ()
        if scan_id:
            sql += " WHERE scan_id = ?"
            args = (scan_id,)
        sql += " ORDER BY created_at DESC"
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [self._detection_out(dict(r)) for r in rows]

    # -- shaping --

    @staticmethod
    def _scan_out(row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        out["stub"] = bool(out.get("stub"))
        models = out.get("models")
        if isinstance(models, str):
            try:
                out["models"] = json.loads(models)
            except json.JSONDecodeError:
                out["models"] = []
        return out

    @staticmethod
    def _detection_out(row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        out["anomaly"] = bool(out.get("anomaly"))
        record = out.get("record")
        if isinstance(record, str):
            try:
                out["record"] = json.loads(record)
            except json.JSONDecodeError:
                out["record"] = {}
        return out


# --- Supabase --------------------------------------------------------------

class SupabaseStore:
    """The same interface over the tables the project already had.

    Column names and the bucket are unchanged from the original routes, so an
    existing Supabase project keeps working and no migration is needed for the
    rows that are already there.
    """

    kind = "supabase"

    def __init__(self, client):
        self.client = client

    def status(self) -> str:
        return "connected"

    # -- images --

    def put_image(self, path: str, data: bytes, content_type: str) -> str:
        self.client.storage.from_(BUCKET).upload(
            path=path, file=data,
            file_options={"content-type": content_type, "upsert": "false"})
        return path

    def open_image(self, path: str) -> Path | None:
        """Cached to a local file so one route can serve both backends.

        Supabase storage hands back bytes, not a path, and FileResponse wants a
        path. Writing it under data/uploads means a tile fetched once is served
        from disk afterwards.
        """
        target = UPLOAD_DIR / Path(path).name
        if target.is_file():
            return target
        try:
            data = self.client.storage.from_(BUCKET).download(path)
        except Exception:
            return None
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target

    # -- scans --

    def create_scan(self, scan: dict[str, Any]) -> dict[str, Any]:
        row = {
            "id": scan["id"],
            "image_url": scan.get("image_url"),
            "latitude": scan.get("latitude"),
            "longitude": scan.get("longitude"),
            "status": scan.get("status", "analysed"),
        }
        self.client.table("scans").insert(row).execute()
        # Echo back the fields the caller supplied. The extra columns are not
        # required to exist in an older Supabase project, so they are not sent
        # and not read back.
        return {**row, "filename": scan.get("filename"), "bytes": scan.get("bytes"),
                "stub": bool(scan.get("stub")), "models": scan.get("models") or [],
                "created_at": scan.get("created_at") or _now()}

    def list_scans(self, limit: int = 200) -> list[dict[str, Any]]:
        res = (self.client.table("scans").select("*")
               .order("created_at", desc=True).limit(limit).execute())
        return res.data or []

    def get_scan(self, scan_id: str) -> dict[str, Any] | None:
        res = self.client.table("scans").select("*").eq("id", scan_id).execute()
        rows = res.data or []
        return rows[0] if rows else None

    def delete_scan(self, scan_id: str) -> bool:
        if self.get_scan(scan_id) is None:
            return False
        self.client.table("detections").delete().eq("scan_id", scan_id).execute()
        self.client.table("scans").delete().eq("id", scan_id).execute()
        return True

    # -- detections --

    def insert_detections(self, scan_id: str, rows: list[dict[str, Any]]) -> list[dict]:
        if not rows:
            return []
        payload = []
        for d in rows:
            bbox = d.get("bbox") or [0, 0, 0, 0]
            payload.append({
                "id": d["id"],
                "scan_id": scan_id,
                "object_class": d.get("class") or d.get("object_class"),
                "confidence": d.get("confidence"),
                "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3],
                "anomaly": bool(d.get("anomaly")),
                "severity": d.get("severity"),
            })
        res = self.client.table("detections").insert(payload).execute()
        return res.data or payload

    def list_detections(self, scan_id: str | None = None) -> list[dict[str, Any]]:
        query = self.client.table("detections").select("*")
        if scan_id:
            query = query.eq("scan_id", scan_id)
        return query.execute().data or []


# --- Selection -------------------------------------------------------------

_store: SqliteStore | SupabaseStore | None = None
_store_lock = threading.Lock()


def get_store():
    """Supabase when it is configured, SQLite otherwise. Chosen once."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = SupabaseStore(supabase) if supabase_ready() else SqliteStore()
    return _store


def storage_status() -> str:
    """What /health reports: which backend, and where it is."""
    store = get_store()
    return f"{store.kind}: {store.status()}"
