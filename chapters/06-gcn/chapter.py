"""Tiny GCN for suspicious-node classification."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from graph_tutorial.datasets import suspicious_accounts_graph
from graph_tutorial.gnn_layers import GCNLayer
from graph_tutorial.training import NodeTrainingResult, train_node_classifier


class GCNClassifier(nn.Module):
    """Two-layer GCN node classifier."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = GCNLayer(in_channels, hidden_channels)
        self.conv2 = GCNLayer(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


def run_gcn_lab() -> tuple[NodeTrainingResult, torch.Tensor]:
    """Train a small GCN and return probabilities."""

    torch.manual_seed(12)
    graph = suspicious_accounts_graph()
    model = GCNClassifier(graph.x.shape[1], hidden_channels=8, out_channels=2)
    result = train_node_classifier(model, graph, epochs=90, lr=0.05, weight_decay=1e-3)
    probabilities = torch.softmax(result.logits, dim=1)
    return result, probabilities


def main() -> None:
    result, probabilities = run_gcn_lab()
    predictions = result.logits.argmax(dim=1).tolist()
    print("# GCN suspicious-node trace")
    print(f"loss: first={result.losses[0]:.4f} final={result.losses[-1]:.4f}")
    print(f"train accuracy: {result.train_accuracy:.2f}")
    if result.test_accuracy is None:
        print("test: n/a")
    else:
        print(f"test accuracy: {result.test_accuracy:.2f}")
    print("predicted classes:", predictions)
    print("suspicious probabilities:", probabilities[:, 1].round(decimals=4).tolist())


if __name__ == "__main__":
    main()
