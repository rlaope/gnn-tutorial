"""GraphSAGE for new-node inference."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from graph_tutorial.datasets import GraphData, suspicious_accounts_graph
from graph_tutorial.gnn_layers import GraphSAGELayer
from graph_tutorial.graph_utils import edge_list_to_edge_index
from graph_tutorial.training import NodeTrainingResult, train_node_classifier


class GraphSAGEClassifier(nn.Module):
    """Two-layer mean GraphSAGE node classifier."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.sage1 = GraphSAGELayer(in_channels, hidden_channels)
        self.sage2 = GraphSAGELayer(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.sage1(x, edge_index))
        return self.sage2(x, edge_index)


def append_new_account(graph: GraphData) -> GraphData:
    """Append one new account connected to two existing device nodes."""

    new_features = torch.tensor([[1.0, 0.0, 0.85]], dtype=torch.float32)
    new_edges = edge_list_to_edge_index([(10, 4), (10, 5)], undirected=True)
    return GraphData(
        x=torch.cat([graph.x, new_features], dim=0),
        edge_index=torch.cat([graph.edge_index, new_edges], dim=1),
        name="suspicious_accounts_plus_new",
    )


def run_graphsage_lab() -> tuple[NodeTrainingResult, float, list[int]]:
    """Train on the old graph, then score a new account in an extended graph."""

    torch.manual_seed(21)
    train_graph = suspicious_accounts_graph()
    model = GraphSAGEClassifier(train_graph.x.shape[1], hidden_channels=8, out_channels=2)
    result = train_node_classifier(model, train_graph, epochs=90, lr=0.04, weight_decay=1e-3)
    extended = append_new_account(train_graph)
    model.eval()
    with torch.no_grad():
        logits = model(extended.x, extended.edge_index)
        probability = torch.softmax(logits, dim=1)[10, 1].item()
    return result, float(probability), [4, 5]


def main() -> None:
    result, new_probability, neighbors = run_graphsage_lab()
    print("# GraphSAGE new-node trace")
    print(f"loss: first={result.losses[0]:.4f} final={result.losses[-1]:.4f}")
    print(f"train accuracy: {result.train_accuracy:.2f}")
    print("new account neighbors:", neighbors)
    print(f"new account suspicious probability: {new_probability:.4f}")


if __name__ == "__main__":
    main()
