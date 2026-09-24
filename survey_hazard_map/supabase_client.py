"""The Supabase client, and a way to run without one.

The original raised at import time when SUPABASE_URL or SUPABASE_KEY was
missing. That was right when this was the only backend: no database, no product.
It is wrong now. The grounded assistant and the detector need no database at
all, so an absent key should cost you history and persistence, not the whole
application. A teammate cloning this to try the chat should not be blocked on
credentials for a service they are not using.

So the client is built lazily and its absence is reported, not raised. Routes
that genuinely need storage call require_supabase() and return a clear 503.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import HTTPException

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent

# backend/.env first, then the repository root .env, so one file can hold the
# Supabase credentials and the model provider keys together if you prefer.
for env_file in (BASE_DIR / ".env", ROOT_DIR / ".env"):
    if env_file.exists():
        load_dotenv(env_file, override=False)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

_client = None
_error: str | None = None

if not SUPABASE_URL or not SUPABASE_KEY:
    _error = ("SUPABASE_URL and SUPABASE_KEY are not set, so uploads, history "
              "and stored scans are unavailable. The assistant and the detector "
              "do not need them.")
else:
    try:
        from supabase import create_client

        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as exc:  # a bad URL or an SDK problem, not a missing key
        _error = f"Supabase client could not be created: {exc}"

# Kept as a module-level name because the existing routes import it directly.
# It is None when unconfigured, which is why they go through require_supabase().
supabase = _client


def supabase_ready() -> bool:
    return _client is not None


def supabase_status() -> str:
    return "connected" if _client is not None else (_error or "unconfigured")


def require_supabase():
    """The client, or a 503 that says exactly what is missing."""
    if _client is None:
        raise HTTPException(status_code=503, detail=_error)
    return _client
