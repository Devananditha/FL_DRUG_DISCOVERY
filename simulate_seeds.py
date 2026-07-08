import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import random
import statistics
import torch
import torch.nn as nn
import networkx as nx
from scipy import stats

from client_api import EMBEDDING_DIM, HOLDOUT_EDGES, POSITIVE_TRAIN_EDGES, TRAINING_EPOCHS, LinkPredictor

SEEDS = [42, 123, 2024, 7, 999]
GRAPH_FILES = ["data/client_1_graph.graphml", "data/client_2_graph.graphml", "data/client_3_graph.graphml"]

def train_and_eval(G, all_nodes, holdout_positive, holdout_negative, seed):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    all_edges = list(G.edges())
    rng.shuffle(all_edges)

    positive_train = all_edges[:POSITIVE_TRAIN_EDGES]
    
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

    neg_train = sample_neg(len(positive_train))
    
    train_edges = positive_train + neg_train
    labels = [1.0] * len(positive_train) + [0.0] * len(neg_train)

    d_idx = torch.tensor([all_nodes.get(s, 0) for s, _ in train_edges], dtype=torch.long)
    t_idx = torch.tensor([all_nodes.get(t, 0) for _, t in train_edges], dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.float32)

    model = LinkPredictor(EMBEDDING_DIM, "classification")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(TRAINING_EPOCHS):
        optimizer.zero_grad()
        loss = criterion(model(d_idx, t_idx).squeeze(-1), y)
        loss.backward()
        optimizer.step()

    h_edges = holdout_positive + holdout_negative
    h_d = torch.tensor([all_nodes.get(s, 0) for s, _ in h_edges], dtype=torch.long)
    h_t = torch.tensor([all_nodes.get(t, 0) for _, t in h_edges], dtype=torch.long)
    y_true = [1.0] * len(holdout_positive) + [0.0] * len(holdout_negative)

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(h_d, h_t).squeeze(-1)).tolist()

    from sklearn.metrics import precision_score, recall_score, f1_score
    y_bin = [1.0 if p >= 0.5 else 0.0 for p in probs]
    ranked = sorted(zip(probs, y_true), key=lambda x: x[0], reverse=True)
    mrr = next((1.0 / (i + 1) for i, (_, l) in enumerate(ranked) if l == 1.0), 0.0)

    metrics = {
        "precision": precision_score(y_true, y_bin, zero_division=0),
        "recall": recall_score(y_true, y_bin, zero_division=0),
        "f1_score": f1_score(y_true, y_bin, zero_division=0),
        "mrr": mrr,
    }
    for k in [5, 10, 20, 50]:
        top = ranked[:min(k, len(ranked))]
        metrics[f"top_{k}_precision"] = sum(l for _, l in top) / len(top) if top else 0.0

    return metrics, model.state_dict(), len(positive_train)

def main():
    # Load global graph to define consistent holdout set and all nodes
    G_global = nx.Graph()
    for f in GRAPH_FILES:
        G_global.add_edges_from(nx.read_graphml(f).edges())
    all_nodes = {n: i for i, n in enumerate(G_global.nodes())}
    
    # We must fix a single holdout set across all seeds/models to evaluate them fairly!
    rng_global = random.Random(42)
    global_edges = list(G_global.edges())
    rng_global.shuffle(global_edges)
    holdout_positive = global_edges[:HOLDOUT_EDGES]
    
    def sample_global_neg(count):
        nodes = list(all_nodes.keys())
        negs = []
        attempts = 0
        while len(negs) < count and attempts < count * 20:
            attempts += 1
            s, t = rng_global.choice(nodes), rng_global.choice(nodes)
            if s != t and not G_global.has_edge(s, t):
                negs.append((s, t))
        return negs
    holdout_negative = sample_global_neg(HOLDOUT_EDGES)
    
    # Graphs per client
    client_graphs = [nx.read_graphml(f) for f in GRAPH_FILES]

    results = {"client_1": [], "client_2": [], "client_3": [], "federated": [], "centralized": []}

    for seed in SEEDS:
        print(f"Running seed {seed}...")
        client_models = []
        client_edge_counts = []
        
        # Local Models
        for i, G in enumerate(client_graphs):
            metrics, state, e_count = train_and_eval(G, all_nodes, holdout_positive, holdout_negative, seed)
            results[f"client_{i+1}"].append(metrics)
            client_models.append(state)
            client_edge_counts.append(e_count)
            
        # Federated Model (Weighted Average)
        total_edges = sum(client_edge_counts)
        fed_state = {}
        for k in client_models[0].keys():
            fed_state[k] = sum(client_models[i][k] * (client_edge_counts[i] / total_edges) for i in range(3))
        
        # Eval Fed Model
        fed_model = LinkPredictor(EMBEDDING_DIM, "classification")
        fed_model.load_state_dict(fed_state)
        fed_model.eval()
        
        h_edges = holdout_positive + holdout_negative
        h_d = torch.tensor([all_nodes.get(s, 0) for s, _ in h_edges], dtype=torch.long)
        h_t = torch.tensor([all_nodes.get(t, 0) for _, t in h_edges], dtype=torch.long)
        y_true = [1.0] * len(holdout_positive) + [0.0] * len(holdout_negative)
        
        with torch.no_grad():
            probs = torch.sigmoid(fed_model(h_d, h_t).squeeze(-1)).tolist()
            
        from sklearn.metrics import precision_score, recall_score, f1_score
        y_bin = [1.0 if p >= 0.5 else 0.0 for p in probs]
        ranked = sorted(zip(probs, y_true), key=lambda x: x[0], reverse=True)
        mrr = next((1.0 / (i + 1) for i, (_, l) in enumerate(ranked) if l == 1.0), 0.0)

        fed_metrics = {
            "precision": precision_score(y_true, y_bin, zero_division=0),
            "recall": recall_score(y_true, y_bin, zero_division=0),
            "f1_score": f1_score(y_true, y_bin, zero_division=0),
            "mrr": mrr,
        }
        for k in [5, 10, 20, 50]:
            top = ranked[:min(k, len(ranked))]
            fed_metrics[f"top_{k}_precision"] = sum(l for _, l in top) / len(top) if top else 0.0
        results["federated"].append(fed_metrics)
        
        # Centralized Model
        c_metrics, _, _ = train_and_eval(G_global, all_nodes, holdout_positive, holdout_negative, seed)
        results["centralized"].append(c_metrics)

    # Compute Stats
    print("\n--- STATS ---")
    keys = ["precision", "recall", "f1_score", "top_5_precision", "top_10_precision", "top_20_precision", "top_50_precision", "mrr"]
    final_stats = {}
    for group, runs in results.items():
        print(f"\n{group.upper()}")
        final_stats[group] = {}
        for k in keys:
            vals = [r[k] for r in runs]
            mean = statistics.mean(vals)
            std = statistics.stdev(vals)
            final_stats[group][k] = f"{mean:.3f} \pm {std:.3f}"
            print(f"{k}: {mean:.3f} +/- {std:.3f}")
            
    # T-Test (Centralized vs FL on F1)
    c_f1 = [r["f1_score"] for r in results["centralized"]]
    f_f1 = [r["f1_score"] for r in results["federated"]]
    t_stat, p_val = stats.ttest_ind(c_f1, f_f1)
    print(f"\nT-Test (Centralized vs Federated F1): t={t_stat:.4f}, p={p_val:.4f}")

if __name__ == "__main__":
    main()
