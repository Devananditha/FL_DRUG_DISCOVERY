"""Collect exactly-once recovery metrics from the SQLite checkpoint ledger."""

import json
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "ledger" / "ledger.db"
METRICS_DIR = Path(__file__).resolve().parent / "metrics"
LEDGER_METRICS_PATH = METRICS_DIR / "ledger_metrics.json"


def load_ledger() -> pd.DataFrame:
    """Load all checkpoint ledger rows into a DataFrame."""
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM checkpoint_ledger", conn)


def calculate_metrics(ledger_df: pd.DataFrame) -> dict[str, int]:
    """Calculate final evaluation metrics from checkpoint ledger events."""
    return {
        "total_ledger_entries": len(ledger_df),
        "committed_updates": int((ledger_df["status"] == "update_committed").sum()),
        "ignored_duplicates": int((ledger_df["status"] == "duplicate_ignored").sum()),
    }


def print_summary(metrics: dict[str, int]) -> None:
    """Print a presentation-friendly metrics summary."""
    print("\n=== Exactly-Once Recovery Metrics ===")
    print(f"Total Ledger Entries: {metrics['total_ledger_entries']}")
    print(f"Committed Updates: {metrics['committed_updates']}")
    print(f"Ignored Duplicates: {metrics['ignored_duplicates']}")


def save_metrics(metrics: dict[str, int]) -> None:
    """Save ledger metrics for chart generation."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved ledger metrics to {LEDGER_METRICS_PATH}")


if __name__ == "__main__":
    ledger = load_ledger()
    summary_metrics = calculate_metrics(ledger)
    print_summary(summary_metrics)
    save_metrics(summary_metrics)
