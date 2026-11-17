import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.data.quality import load_config

DB_PATH = Path(load_config()["api"]["db_path"])


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db() -> None:
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            model_version TEXT,
            inputs TEXT,
            prediction REAL,
            duration_ms REAL,
            status TEXT NOT NULL,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            prediction_id INTEGER NOT NULL REFERENCES predictions(id),
            actual_quantity REAL NOT NULL,
            comment TEXT
        );
        """)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_prediction(model_version, inputs, prediction, duration_ms, status, error=None) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO predictions (ts, model_version, inputs, prediction, duration_ms, status, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (_now(), None if model_version is None else str(model_version),
             json.dumps(inputs) if inputs is not None else None,
             prediction, duration_ms, status, error),
        )
        return cur.lastrowid


def add_feedback(prediction_id: int, actual_quantity: float, comment) -> int | None:
    with _conn() as c:
        exists = c.execute("SELECT 1 FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
        if not exists:
            return None
        cur = c.execute(
            "INSERT INTO feedback (ts, prediction_id, actual_quantity, comment) VALUES (?, ?, ?, ?)",
            (_now(), prediction_id, actual_quantity, comment),
        )
        return cur.lastrowid


def stats() -> dict:
    with _conn() as c:
        by_status = {r["status"]: r["n"] for r in c.execute(
            "SELECT status, COUNT(*) AS n FROM predictions GROUP BY status")}
        n_fb = c.execute("SELECT COUNT(*) AS n FROM feedback").fetchone()["n"]
    return {"predictions_by_status": by_status, "feedbacks": n_fb}


_PIPELINE_DDL = ("CREATE TABLE IF NOT EXISTS pipeline_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                 "ts REAL NOT NULL, name TEXT, status TEXT NOT NULL, duration_s REAL)")


def log_pipeline_run(name, status, duration_s) -> None:
    import time as _time
    with _conn() as c:
        c.execute(_PIPELINE_DDL)
        c.execute("INSERT INTO pipeline_runs (ts, name, status, duration_s) VALUES (?, ?, ?, ?)",
                  (_time.time(), name, status, duration_s))


def pipeline_stats() -> dict:
    with _conn() as c:
        c.execute(_PIPELINE_DDL)
        by = {r["status"]: r["n"] for r in c.execute(
            "SELECT status, COUNT(*) AS n FROM pipeline_runs GROUP BY status")}
        last = c.execute("SELECT * FROM pipeline_runs ORDER BY id DESC LIMIT 1").fetchone()
    return {"by_status": by, "last": dict(last) if last else None}

