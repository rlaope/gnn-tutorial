"""Tiny GAT with inspectable incoming attention weights."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from graph_tutorial.datasets import suspicious_accounts_graph
from graph_tutorial.gnn_layers import SingleHeadGATLayer
from graph_tutorial.training import NodeTrainingResult, train_node_classifier


class GATClassifier(nn.Module):
    """One attention layer plus a classifier head."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.attention = SingleHeadGATLayer(in_channels, hidden_channels)
        self.output = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.attention(x, edge_index)
        return self.output(F.elu(h))

    def attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor):
        _, attention_edges, weights = self.attention(x, edge_index, return_attention=True)
        return attention_edges, weights


def top_incoming_attention(
    edge_index: torch.Tensor,
    weights: torch.Tensor,
    *,
    target_node: int,
    top_k: int = 3,
) -> list[tuple[int, int, float]]:
    """Return top incoming attention edges for a target node."""

    rows: list[tuple[int, int, float]] = []
    for idx, (src, dst) in enumerate(edge_index.t().tolist()):
        if int(dst) == target_node:
            rows.append((int(src), int(dst), float(weights[idx])))
    return sorted(rows, key=lambda row: row[2], reverse=True)[:top_k]


def run_gat_lab() -> tuple[NodeTrainingResult, list[tuple[int, int, float]]]:
    """Train a tiny GAT and collect attention for node 8."""

    torch.manual_seed(31)
    graph = suspicious_accounts_graph()
    model = GATClassifier(graph.x.shape[1], hidden_channels=8, out_channels=2)
    result = train_node_classifier(model, graph, epochs=90, lr=0.04, weight_decay=1e-3)
    model.eval()
    with torch.no_grad():
        edge_index, weights = model.attention_weights(graph.x, graph.edge_index)
    return result, top_incoming_attention(edge_index, weights, target_node=8)


def main() -> None:
    result, attention_rows = run_gat_lab()
    print("# GAT neighbor-attention trace")
    print(f"loss: first={result.losses[0]:.4f} final={result.losses[-1]:.4f}")
    print(f"train accuracy: {result.train_accuracy:.2f}")
    print("top incoming attention for node 8:")
    for src, dst, weight in attention_rows:
        print(f"{src} -> {dst}: {weight:.4f}")


if __name__ == "__main__":
    main()
