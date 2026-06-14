"""Exercise scaffold for preserving a node's own signal."""

from __future__ import annotations

import torch

from graph_tutorial.datasets import toy_social_graph
from graph_tutorial.graph_utils import mean_aggregate


def aggregate_with_self_loops(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Return mean aggregation that includes each node's own features.

    TODO: Add one self-loop per node before calling mean_aggregate.
    """

    return mean_aggregate(x, edge_index, num_nodes)


def main() -> None:
    graph = toy_social_graph()
    updated = aggregate_with_self_loops(graph.x, graph.edge_index, graph.num_nodes)
    print("Node 0 original:", graph.x[0].tolist())
    print("Node 0 updated:", updated[0].round(decimals=4).tolist())
    print("TODO: node 0 should still include some of its own [1.0, 0.0] signal.")


if __name__ == "__main__":
    main()
