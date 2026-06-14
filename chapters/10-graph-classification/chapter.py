"""Whole-graph classification with mean pooling."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from graph_tutorial.datasets import GraphData
from graph_tutorial.gnn_layers import GCNLayer
from graph_tutorial.graph_utils import degree, edge_list_to_edge_index
from graph_tutorial.metrics import accuracy


def graph_from_edges(name: str, edges: list[tuple[int, int]], label: int) -> GraphData:
    """Create a graph with degree and density node features."""

    num_nodes = max(max(src, dst) for src, dst in edges) + 1
    edge_index = edge_list_to_edge_index(edges, undirected=True)
    normalized_degree = degree(edge_index, num_nodes, incoming=True) / max(1, num_nodes - 1)
    density = 2.0 * len(edges) / max(1, num_nodes * (num_nodes - 1))
    density_feature = torch.full((num_nodes,), float(density))
    x = torch.stack([normalized_degree, density_feature], dim=1)
    return GraphData(x=x, edge_index=edge_index, y=torch.tensor([label]), name=name)


def shape_dataset() -> list[GraphData]:
    """Return tiny graph shapes: open structures vs closed structures."""

    return [
        graph_from_edges("chain_3", [(0, 1), (1, 2)], 0),
        graph_from_edges("chain_4", [(0, 1), (1, 2), (2, 3)], 0),
        graph_from_edges("star_4", [(0, 1), (0, 2), (0, 3)], 0),
        graph_from_edges("triangle", [(0, 1), (1, 2), (2, 0)], 1),
        graph_from_edges("cycle_4", [(0, 1), (1, 2), (2, 3), (3, 0)], 1),
        graph_from_edges("square_diag", [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], 1),
    ]


class MeanPoolGraphClassifier(nn.Module):
    """GCN encoder plus mean-pooling graph readout."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = GCNLayer(in_channels, hidden_channels)
        self.output = nn.Linear(hidden_channels, out_channels)

    def forward_graph(self, graph: GraphData) -> torch.Tensor:
        h = F.relu(self.conv(graph.x, graph.edge_index))
        pooled = h.mean(dim=0, keepdim=True)
        return self.output(pooled)


def train_graph_classifier() -> dict[str, object]:
    """Train a tiny graph classifier over all toy shapes."""

    torch.manual_seed(61)
    graphs = shape_dataset()
    model = MeanPoolGraphClassifier(in_channels=2, hidden_channels=8, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08, weight_decay=1e-3)
    losses: list[float] = []
    for _ in range(160):
        optimizer.zero_grad()
        graph_losses = []
        for graph in graphs:
            graph_losses.append(F.cross_entropy(model.forward_graph(graph), graph.y))
        loss = torch.stack(graph_losses).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    with torch.no_grad():
        logits = torch.cat([model.forward_graph(graph) for graph in graphs], dim=0)
    labels = torch.cat([graph.y for graph in graphs])
    predictions = logits.argmax(dim=1)
    return {
        "loss_first": losses[0],
        "loss_final": losses[-1],
        "accuracy": accuracy(predictions, labels),
        "predictions": list(
            zip([graph.name for graph in graphs], predictions.tolist(), strict=True)
        ),
    }


def main() -> None:
    result = train_graph_classifier()
    print("# Graph classification pooling trace")
    print(f"loss: first={result['loss_first']:.4f} final={result['loss_final']:.4f}")
    print(f"graph accuracy: {result['accuracy']:.2f}")
    print("predictions:")
    for name, prediction in result["predictions"]:
        print(f"{name}: class {prediction}")


if __name__ == "__main__":
    main()
