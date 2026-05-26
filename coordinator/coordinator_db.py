"""Coordinator database helpers for idempotent update processing."""

import sqlite3
from pathlib import Path

_DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "ledger" / "ledger.db"


def check_if_duplicate(update_id: str, db_path: str = "ledger.db") -> bool:
    """Return True if update_id already exists in the checkpoint ledger."""
    resolved_path = db_path
    if db_path == "ledger.db":
        resolved_path = str(_DEFAULT_DB_PATH)

    with sqlite3.connect(resolved_path, timeout=30.0) as conn:
        cursor = conn.execute(
            """
            SELECT update_id
            FROM checkpoint_ledger
            WHERE update_id = ?
            """,
            (update_id,),
        )
        row = cursor.fetchone()

    return row is not None
