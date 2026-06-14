"""Exercise scaffold for GCN normalized aggregation."""

from __future__ import annotations

import torch

from graph_tutorial.datasets import suspicious_accounts_graph


def apply_gcn_aggregation(x: torch.Tensor, _edge_index: torch.Tensor) -> torch.Tensor:
    """Return normalized adjacency times features.

    TODO: Use graph_tutorial.graph_utils.normalized_adjacency.
    """

    return x


def main() -> None:
    graph = suspicious_accounts_graph()
    aggregated = apply_gcn_aggregation(graph.x, graph.edge_index)
    print("Input shape:", tuple(graph.x.shape))
    print("Aggregated shape:", tuple(aggregated.shape))
    print("Node 0 before:", graph.x[0].tolist())
    print("Node 0 after:", aggregated[0].round(decimals=4).tolist())


if __name__ == "__main__":
    main()
