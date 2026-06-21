"""baseline_centralized.py - Monolithic baseline for Table 1"""
import random
import sys
from pathlib import Path
import networkx as nx
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client_api import (
    LinkPredictor, _edges_to_index_tensors, _evaluate_model,
    _sample_negative_edges, _affinity_labels, POSITIVE_TRAIN_EDGES, 
    HOLDOUT_EDGES, TRAINING_EPOCHS, EMBEDDING_DIM
)

def run():
    G = nx.Graph()
    for f in ["data/client_1_graph.graphml", "data/client_2_graph.graphml", "data/client_3_graph.graphml"]:
        try:
            G.add_edges_from(nx.read_graphml(f).edges())
        except Exception as e:
            print(f"Failed to load {f}: {e}")

    import client_api
    client_api.G = G
    client_api.NODE_TO_IDX = {node: index for index, node in enumerate(G.nodes())}
    client_api.NUM_NODES = len(client_api.NODE_TO_IDX)
    
    seed = 42
    torch.manual_seed(seed)
    rng = random.Random(seed)
    
    edges = list(G.edges())
    rng.shuffle(edges)
    
    pos_train = edges[:POSITIVE_TRAIN_EDGES]
    pos_test = edges[POSITIVE_TRAIN_EDGES:POSITIVE_TRAIN_EDGES+HOLDOUT_EDGES]
    neg_pool = _sample_negative_edges(len(pos_train) + HOLDOUT_EDGES, rng)
    
    train_edges = pos_train + neg_pool[:len(pos_train)]
    y_train = _affinity_labels(len(pos_train), True, "classification", rng) + \
              _affinity_labels(len(neg_pool[:len(pos_train)]), False, "classification", rng)
              
    d_idx, t_idx = _edges_to_index_tensors(train_edges)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    model = LinkPredictor(EMBEDDING_DIM, "classification")
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for _ in range(TRAINING_EPOCHS):
        opt.zero_grad()
        loss = nn.BCEWithLogitsLoss()(model(d_idx, t_idx).squeeze(-1), y_tensor)
        loss.backward()
        opt.step()
        
    metrics = _evaluate_model(model, pos_test, neg_pool[len(pos_train):], "classification", rng)
    print("Centralized Baseline Metrics:", metrics)

if __name__ == "__main__":
    run()
