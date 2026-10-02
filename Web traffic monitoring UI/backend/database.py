"""SQLite setup and small persistence helpers for the TrafficVision API."""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STORE_DIR = Path(os.getenv("TRAFFIC_STORE_DIR", str(Path(__file__).parent / "storage"))).resolve()
DB_PATH = Path(os.getenv("TRAFFIC_DB_PATH", str(STORE_DIR / "traffic.sqlite3")))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def connect():
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, camera_id INTEGER, filename TEXT NOT NULL, media_type TEXT NOT NULL, media_path TEXT NOT NULL, created_at TEXT NOT NULL, duration REAL, width INTEGER, height INTEGER, provider TEXT, model TEXT, processing_status TEXT NOT NULL DEFAULT 'complete', processing_error TEXT, llm_analysis_json TEXT, llm_updated_at TEXT, llm_status TEXT NOT NULL DEFAULT 'idle');
        CREATE TABLE IF NOT EXISTS frames (id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, frame_index INTEGER NOT NULL, timestamp_sec REAL NOT NULL, image_path TEXT NOT NULL, analysis_json TEXT, analyzed_at TEXT, yolo_json TEXT);
        CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, level TEXT NOT NULL, message TEXT NOT NULL, session_id TEXT);
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_frames_session ON frames(session_id, frame_index);
        CREATE INDEX IF NOT EXISTS idx_logs_created ON logs(created_at DESC);
        """)
        frame_columns = {row[1] for row in db.execute("PRAGMA table_info(frames)")}
        session_columns = {row[1] for row in db.execute("PRAGMA table_info(sessions)")}
        for name, declaration in (("processing_status", "TEXT NOT NULL DEFAULT 'complete'"), ("processing_error", "TEXT"), ("llm_analysis_json", "TEXT"), ("llm_updated_at", "TEXT"), ("llm_status", "TEXT NOT NULL DEFAULT 'idle'")):
            if name not in session_columns:
                db.execute(f"ALTER TABLE sessions ADD COLUMN {name} {declaration}")
        for name, declaration in (("yolo_json", "TEXT"),):
            if name not in frame_columns:
                db.execute(f"ALTER TABLE frames ADD COLUMN {name} {declaration}")
        defaults = {"provider": "ollama", "ollama_model": "qwen2.5vl:7b", "gemini_model": "gemini-2.5-flash", "gpt_model": "gpt-4o-mini", "detector_model": "yolo11s.pt", "sample_every_seconds": 5, "llm_sample_every_seconds": 5, "detector_fps": 5, "max_frames": 30000, "motorcycle_alert_threshold": 150}
        for key, value in defaults.items():
            db.execute("INSERT OR IGNORE INTO settings(key,value) VALUES (?,?)", (key, json.dumps(value)))
        for key, old, new in (("sample_every_seconds", 10, 5), ("max_frames", 8, 100), ("max_frames", 30, 100), ("max_frames", 100, 30000)):
            db.execute("UPDATE settings SET value=? WHERE key=? AND value=?", (json.dumps(new), key, json.dumps(old)))
        # The previous 30 FPS value was the built-in default (there is no UI control for it).
        db.execute("UPDATE settings SET value=? WHERE key='detector_fps' AND value=?", (json.dumps(5), json.dumps(30)))


def log(message: str, level: str = "INFO", session_id: str | None = None):
    with connect() as db:
        db.execute("INSERT INTO logs(created_at,level,message,session_id) VALUES (?,?,?,?)", (datetime.now(timezone.utc).isoformat(), level, message, session_id))


def get_settings() -> dict[str, Any]:
    with connect() as db:
        return {row["key"]: json.loads(row["value"]) for row in db.execute("SELECT key,value FROM settings")}
