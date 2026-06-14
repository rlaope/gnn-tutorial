"""Solution for chapter 02 self-loop aggregation."""

from __future__ import annotations

import torch

from graph_tutorial.datasets import toy_social_graph
from graph_tutorial.graph_utils import add_self_loops, mean_aggregate


def aggregate_with_self_loops(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Add self-loops, then average incoming messages."""

    edge_index = add_self_loops(edge_index, num_nodes)
    return mean_aggregate(x, edge_index, num_nodes)


def main() -> None:
    graph = toy_social_graph()
    updated = aggregate_with_self_loops(graph.x, graph.edge_index, graph.num_nodes)
    print("Node 0 original:", graph.x[0].tolist())
    print("Node 0 updated:", updated[0].round(decimals=4).tolist())


if __name__ == "__main__":
    main()
