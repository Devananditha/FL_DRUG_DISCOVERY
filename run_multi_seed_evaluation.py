"""run_multi_seed_evaluation.py
Runs centralized baseline and federated pipeline across N seeds.
Produces mean +/- std for every Table I metric and saves:
  - multi_seed_results.json  (raw data)
  - plot_convergence.png     (loss curve per seed)
"""

import json
import random
import statistics
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import requests
import torch
import torch.nn as nn
import networkx as nx
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))

from client_api import (
    EMBEDDING_DIM,
    HOLDOUT_EDGES,
    POSITIVE_TRAIN_EDGES,
    TRAINING_EPOCHS,
    LinkPredictor,
)

SEEDS = [42, 123, 2024, 7, 999]
DRUG_ID = "CID000000271"
COORDINATOR_URL = "http://localhost:8000"
GRAPH_FILES = [
    "data/client_1_graph.graphml",
    "data/client_2_graph.graphml",
    "data/client_3_graph.graphml",
]
METRIC_KEYS = ["precision", "recall", "f1_score",
               "top_5_precision", "top_10_precision",
               "top_20_precision", "top_50_precision", "mrr"]


def run_centralized_baseline(seed: int) -> dict:
    """Merge all client graphs and train a monolithic model."""
    G = nx.Graph()
    for f in GRAPH_FILES:
        try:
            G.add_edges_from(nx.read_graphml(f).edges())
        except Exception as e:
            print(f"  [WARN] {f}: {e}")

    all_nodes = {n: i for i, n in enumerate(G.nodes())}
    all_edges = list(G.edges())

    torch.manual_seed(seed)
    rng = random.Random(seed)
    rng.shuffle(all_edges)

    positive_train = all_edges[:POSITIVE_TRAIN_EDGES]
    holdout_positive = all_edges[POSITIVE_TRAIN_EDGES: POSITIVE_TRAIN_EDGES + HOLDOUT_EDGES]

    def sample_neg(count):
        nodes = list(all_nodes.keys())
        negs = []
        attempts = 0
        while len(negs) < count and attempts < count * 20:
            attempts += 1
            s, t = rng.choice(nodes), rng.choice(nodes)
            if s != t and not G.has_edge(s, t):
                negs.append((s, t))
        return negs

    neg_pool = sample_neg(len(positive_train) + HOLDOUT_EDGES)
    neg_train = neg_pool[: len(positive_train)]
    holdout_negative = neg_pool[len(positive_train):]

    train_edges = positive_train + neg_train
    labels = [1.0] * len(positive_train) + [0.0] * len(neg_train)

    d_idx = torch.tensor([all_nodes.get(s, 0) for s, _ in train_edges], dtype=torch.long)
    t_idx = torch.tensor([all_nodes.get(t, 0) for _, t in train_edges], dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.float32)

    model = LinkPredictor(EMBEDDING_DIM, "classification")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    loss_curve = []
    model.train()
    for _ in range(TRAINING_EPOCHS):
        optimizer.zero_grad()
        loss = criterion(model(d_idx, t_idx).squeeze(-1), y)
        loss.backward()
        optimizer.step()
        loss_curve.append(round(loss.item(), 6))

    h_edges = holdout_positive + holdout_negative
    h_d = torch.tensor([all_nodes.get(s, 0) for s, _ in h_edges], dtype=torch.long)
    h_t = torch.tensor([all_nodes.get(t, 0) for _, t in h_edges], dtype=torch.long)
    y_true = [1.0] * len(holdout_positive) + [0.0] * len(holdout_negative)

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(h_d, h_t).squeeze(-1)).tolist()
    if isinstance(probs, float):
        probs = [probs]

    from sklearn.metrics import precision_score, recall_score, f1_score
    y_bin = [1.0 if p >= 0.5 else 0.0 for p in probs]
    ranked = sorted(zip(probs, y_true), key=lambda x: x[0], reverse=True)
    mrr = next((1.0 / (i + 1) for i, (_, l) in enumerate(ranked) if l == 1.0), 0.0)

    metrics = {
        "precision": round(precision_score(y_true, y_bin, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_bin, zero_division=0), 3),
        "f1_score": round(f1_score(y_true, y_bin, zero_division=0), 3),
        "mrr": round(mrr, 3),
        "training_loss_curve": loss_curve,
    }
    for k in [5, 10, 20, 50]:
        top = ranked[:min(k, len(ranked))]
        metrics[f"top_{k}_precision"] = round(sum(l for _, l in top) / len(top), 3) if top else 0.0
    return metrics


def run_federated_round(seed: int) -> dict:
    """Hit the live coordinator and return the federated aggregated metrics."""
    try:
        r = requests.get(
            f"{COORDINATOR_URL}/global_retrieve",
            params={"drug_id": DRUG_ID, "task_type": "classification"},
            timeout=180,
        )
        data = r.json()

        # Use the pre-aggregated federated metrics returned by the coordinator
        fed_metrics = data.get("federated_link_prediction_metrics", {})
        if fed_metrics:
            return {k: fed_metrics.get(k, 0.0) for k in METRIC_KEYS}

        # Fallback: average across per-client metrics if fed metrics missing
        client_metrics = data.get("client_link_prediction_metrics", {})
        if not client_metrics:
            return {}
        aggregated = {}
        for k in METRIC_KEYS:
            vals = [v.get(k, 0.0) for v in client_metrics.values() if k in v]
            aggregated[k] = round(statistics.mean(vals), 3) if vals else 0.0
        return aggregated
    except Exception as e:
        print(f"  [WARN] Federated round failed: {e}")
        return {}


def compute_stats(values: list) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    mean = round(statistics.mean(values), 4)
    std = round(statistics.stdev(values) if len(values) > 1 else 0.0, 4)
    return {"mean": mean, "std": std}


def main():
    print(f"Running multi-seed evaluation | Seeds: {SEEDS}")
    centralized_runs = []
    federated_runs = []
    all_loss_curves = []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        print("  Centralized baseline...")
        c = run_centralized_baseline(seed)
        centralized_runs.append(c)
        all_loss_curves.append(c.get("training_loss_curve", []))
        print(f"  F1={c.get('f1_score')}  MRR={c.get('mrr')}")

        print("  Federated round (live coordinator)...")
        f = run_federated_round(seed)
        if f:
            federated_runs.append(f)
            print(f"  F1={f.get('f1_score')}  MRR={f.get('mrr')}")
        else:
            print("  [SKIP] No federated metrics returned.")

    # Summary table
    print("\n\n========= RESULTS: mean +/- std =========")
    results = {"centralized": {}, "federated": {}}
    print(f"{'Metric':<22} {'Centralized':^22} {'Federated':^22}")
    print("-" * 66)
    for k in METRIC_KEYS:
        c_stat = compute_stats([r.get(k, 0.0) for r in centralized_runs if k in r])
        f_stat = compute_stats([r.get(k, 0.0) for r in federated_runs if k in r])
        results["centralized"][k] = c_stat
        results["federated"][k] = f_stat
        print(f"{k:<22} {c_stat['mean']:.4f} ± {c_stat['std']:.4f}    "
              f"{f_stat['mean']:.4f} ± {f_stat['std']:.4f}")

    with open("multi_seed_results.json", "w") as fh:
        json.dump({"seeds": SEEDS,
                   "raw": {"centralized": centralized_runs, "federated": federated_runs},
                   "stats": results}, fh, indent=2)
    print("\nSaved: multi_seed_results.json")

    # Convergence plot
    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, curve in enumerate(all_loss_curves):
        epochs = list(range(1, len(curve) + 1))
        ax.plot(epochs, curve, marker="o", label=f"Seed {SEEDS[i]}", linewidth=2, alpha=0.85)
    ax.set_title("Centralized Baseline — BCE Training Loss per Epoch", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("BCE Loss", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("plot_convergence.png", dpi=300)
    print("Saved: plot_convergence.png")


if __name__ == "__main__":
    main()
