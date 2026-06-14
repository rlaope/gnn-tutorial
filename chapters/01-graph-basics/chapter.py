"""Build the basic graph tensors from tiny product records."""

from __future__ import annotations

from typing import TypedDict

import torch

from graph_tutorial.datasets import GraphData
from graph_tutorial.graph_utils import (
    degree,
    edge_index_to_adjacency,
    edge_list_to_edge_index,
)

CUSTOMERS = [
    {"node_id": 0, "segment": "creator", "activity": 1.0, "support_score": 0.0},
    {"node_id": 1, "segment": "creator", "activity": 0.9, "support_score": 0.1},
    {"node_id": 2, "segment": "creator", "activity": 0.8, "support_score": 0.2},
    {"node_id": 3, "segment": "buyer", "activity": 0.1, "support_score": 0.9},
    {"node_id": 4, "segment": "buyer", "activity": 0.2, "support_score": 0.8},
    {"node_id": 5, "segment": "buyer", "activity": 0.0, "support_score": 1.0},
]

FRIENDSHIPS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 2), (3, 5), (1, 4)]


class GraphSummary(TypedDict):
    x_shape: tuple[int, int]
    edge_index_shape: tuple[int, int]
    adjacency: torch.Tensor
    degree: torch.Tensor


def records_to_graph() -> GraphData:
    """Convert row-like product records into graph tensors."""

    features = [[row["activity"], row["support_score"]] for row in CUSTOMERS]
    labels = [0 if row["segment"] == "creator" else 1 for row in CUSTOMERS]
    return GraphData(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_list_to_edge_index(FRIENDSHIPS, undirected=True),
        y=torch.tensor(labels, dtype=torch.long),
        name="customer_friendships",
    )


def graph_summary(graph: GraphData) -> GraphSummary:
    """Return the inspectable graph objects for the lab."""

    return {
        "x_shape": (int(graph.x.shape[0]), int(graph.x.shape[1])),
        "edge_index_shape": (int(graph.edge_index.shape[0]), int(graph.edge_index.shape[1])),
        "adjacency": edge_index_to_adjacency(graph.edge_index, graph.num_nodes),
        "degree": degree(graph.edge_index, graph.num_nodes),
    }


def main() -> None:
    graph = records_to_graph()
    summary = graph_summary(graph)
    print("# Product records")
    for row in CUSTOMERS:
        print(row)
    print("\n# Tensor trace")
    print(f"x shape: {summary['x_shape']}")
    print(f"edge_index shape: {summary['edge_index_shape']}")
    print(f"labels: {graph.y.tolist() if graph.y is not None else None}")
    print("degree:", summary["degree"].tolist())
    print("adjacency:")
    print(summary["adjacency"].to(torch.int64))


if __name__ == "__main__":
    main()
