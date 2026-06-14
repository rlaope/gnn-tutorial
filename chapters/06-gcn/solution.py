"""Solution for chapter 06 GCN aggregation."""

from __future__ import annotations

import torch

from graph_tutorial.datasets import suspicious_accounts_graph
from graph_tutorial.graph_utils import normalized_adjacency


def apply_gcn_aggregation(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Apply symmetric normalized aggregation."""

    return normalized_adjacency(edge_index, x.shape[0]) @ x


def main() -> None:
    graph = suspicious_accounts_graph()
    aggregated = apply_gcn_aggregation(graph.x, graph.edge_index)
    print("Input shape:", tuple(graph.x.shape))
    print("Aggregated shape:", tuple(aggregated.shape))
    print("Node 0 before:", graph.x[0].tolist())
    print("Node 0 after:", aggregated[0].round(decimals=4).tolist())


if __name__ == "__main__":
    main()
