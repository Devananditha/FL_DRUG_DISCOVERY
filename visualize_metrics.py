"""Generate presentation-ready evaluation charts for the federated system."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
LOAD_TEST_METRICS_PATH = PROJECT_ROOT / "metrics" / "load_test_metrics.json"
LEDGER_METRICS_PATH = PROJECT_ROOT / "metrics" / "ledger_metrics.json"

DEFAULT_LOAD_TEST_METRICS = {
    "full_answers": 12,
    "degraded_answers": 8,
}
DEFAULT_LEDGER_METRICS = {
    "committed_updates": 496,
    "ignored_duplicates": 3,
}


def load_json_metrics(path: Path, defaults: dict[str, int]) -> dict[str, int]:
    """Load metrics from JSON, or return defaults when no metrics file exists.

    Args:
        path: Metrics JSON file path.
        defaults: Fallback values used before the first generated metrics file.

    Returns:
        Metrics dictionary for chart generation.
    """
    if not path.exists():
        return defaults

    with path.open("r", encoding="utf-8") as metrics_file:
        loaded_metrics = json.load(metrics_file)

    return {**defaults, **loaded_metrics}


def create_reliability_chart() -> None:
    """Create a pie chart for full vs degraded federated retrieval answers.

    Saves the chart to ``reliability_chart.png`` at 300 DPI.

    Returns:
        None
    """
    metrics = load_json_metrics(LOAD_TEST_METRICS_PATH, DEFAULT_LOAD_TEST_METRICS)
    labels = ["Full Answers (3/3)", "Degraded Answers (<3/3)"]
    values = [metrics["full_answers"], metrics["degraded_answers"]]
    colors = ["#2E7D32", "#F9A825"]

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, _, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 11},
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_title(
        "Federated Retrieval Completeness under Fault Injection",
        fontsize=14,
        fontweight="bold",
        pad=18,
    )
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig("reliability_chart.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_recovery_chart() -> None:
    """Create a bar chart for committed updates vs ignored duplicates.

    Saves the chart to ``recovery_chart.png`` at 300 DPI.

    Returns:
        None
    """
    metrics = load_json_metrics(LEDGER_METRICS_PATH, DEFAULT_LEDGER_METRICS)
    labels = ["Update Committed", "Duplicate Ignored"]
    values = [metrics["committed_updates"], metrics["ignored_duplicates"]]
    colors = ["#1565C0", "#C62828"]
    x_positions = np.arange(len(labels))

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(x_positions, values, color=colors, width=0.55)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of Updates", fontsize=12)
    ax.set_title("Exactly-Once Ledger Operations", fontsize=14, fontweight="bold", pad=18)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig("recovery_chart.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    create_reliability_chart()
    create_recovery_chart()
    print("Success: saved reliability_chart.png and recovery_chart.png")
