"""SQLite append-only checkpoint ledger utilities."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "ledger.db"


def init_db() -> None:
    """Create the checkpoint ledger table if it does not already exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoint_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                update_id TEXT UNIQUE,
                query_id TEXT,
                client_id TEXT,
                status TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def log_update(update_id: str, query_id: str, client_id: str, status: str) -> bool:
    """Append an update event unless its idempotency key already exists."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO checkpoint_ledger (
                    update_id,
                    query_id,
                    client_id,
                    status
                )
                VALUES (?, ?, ?, ?)
                """,
                (update_id, query_id, client_id, status),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def is_duplicate(update_id: str) -> bool:
    """Return True if an update_id has already been committed."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "SELECT 1 FROM checkpoint_ledger WHERE update_id = ? LIMIT 1",
            (update_id,),
        )
        return cursor.fetchone() is not None


if __name__ == "__main__":
    init_db()
    print("Success: SQLite append-only ledger initialized at /ledger/ledger.db")
