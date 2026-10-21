"""Tiny SQLite key-value cache: a product is read once, then reused for every user."""

import hashlib
import json
import sqlite3
import threading
import time

from .config import settings

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _db() -> sqlite3.Connection | None:
    global _conn
    if settings.cache_path is None:
        return None
    if _conn is None:
        _conn = sqlite3.connect(settings.cache_path, check_same_thread=False)
        _conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL, t REAL NOT NULL)")
    return _conn


def key(*parts: str | bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else p.encode())
        h.update(b"\x00")
    return h.hexdigest()


def get(k: str, max_age_s: float | None = None):
    db = _db()
    if db is None:
        return None
    with _lock:
        row = db.execute("SELECT v, t FROM kv WHERE k = ?", (k,)).fetchone()
    if not row or (max_age_s is not None and time.time() - row[1] > max_age_s):
        return None
    return json.loads(row[0])


def put(k: str, value) -> None:
    db = _db()
    if db is None:
        return
    with _lock:
        db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?, ?)", (k, json.dumps(value), time.time()))
        db.commit()
