"""Coordinator database helpers for idempotent update processing."""

import sqlite3
from datetime import datetime
from pathlib import Path

_DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "ledger" / "ledger.db"


def _resolve_db_path(db_path: str) -> str:
    if db_path == "ledger.db":
        return str(_DEFAULT_DB_PATH)
    return db_path


def check_if_duplicate(update_id: str, db_path: str = "ledger.db") -> bool:
    resolved_path = _resolve_db_path(db_path)

    with sqlite3.connect(resolved_path, timeout=30.0) as conn:
        cursor = conn.execute(
            """
            SELECT update_id
            FROM checkpoint_ledger
            WHERE update_id = ?
              AND status = 'update_committed'
            """,
            (update_id,),
        )
        row = cursor.fetchone()

    return row is not None


def log_to_ledger(
    query_id: str,
    client_id: str,
    update_id: str,
    status: str,
    db_path: str = "ledger.db",
) -> None:
    resolved_path = _resolve_db_path(db_path)
    timestamp = datetime.now().isoformat()
    ledger_update_id = update_id

    if status == "duplicate_ignored":
        # Keep the original committed update immutable and append a separate audit event.
        ledger_update_id = f"{update_id}::duplicate_ignored::{query_id}"

    with sqlite3.connect(resolved_path, timeout=30.0) as conn:
        try:
            conn.execute(
                """
                INSERT INTO checkpoint_ledger (
                    query_id,
                    client_id,
                    update_id,
                    status,
                    timestamp
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (query_id, client_id, ledger_update_id, status, timestamp),
            )
        except sqlite3.IntegrityError:
            return
