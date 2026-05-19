import json
import sqlite3
import time
from typing import Any


class StateStore:
    def __init__(self, path: str = "./v1/v1_state.sqlite3"):
        self.path = path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    user_id INTEGER PRIMARY KEY,
                    data TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS file_cache (
                    url TEXT PRIMARY KEY,
                    file_id TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )

    def get_session(self, user_id: int) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT data FROM sessions WHERE user_id = ?", (user_id,)).fetchone()
        if not row:
            return {}
        try:
            return json.loads(row[0])
        except Exception:
            return {}

    def set_session(self, user_id: int, data: dict[str, Any]) -> None:
        now = int(time.time())
        payload = json.dumps(data, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(user_id, data, updated_at)
                VALUES(?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at
                """,
                (user_id, payload, now),
            )

    def clear_session(self, user_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

    def cleanup_expired_sessions(self, max_age_seconds: int = 1200) -> int:
        """
        Deletes sessions not updated within `max_age_seconds`.
        Returns number of deleted rows.
        """
        cutoff = int(time.time()) - int(max_age_seconds)
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM sessions WHERE updated_at < ?", (cutoff,))
            return int(cur.rowcount or 0)

    def get_file_id(self, url: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT file_id FROM file_cache WHERE url = ?", (url,)).fetchone()
        return row[0] if row else None

    def set_file_id(self, url: str, file_id: str) -> None:
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO file_cache(url, file_id, updated_at)
                VALUES(?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET file_id=excluded.file_id, updated_at=excluded.updated_at
                """,
                (url, file_id, now),
            )
