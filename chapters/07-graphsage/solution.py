"""Solution for chapter 07 neighbor sampling."""

from __future__ import annotations

import torch

from graph_tutorial.datasets import suspicious_accounts_graph
from graph_tutorial.walks import adjacency_lists


def sample_neighbors(edge_index: torch.Tensor, *, node: int, max_neighbors: int) -> list[int]:
    """Return up to max_neighbors sorted neighbors for node."""

    return adjacency_lists(edge_index)[node][:max_neighbors]


def main() -> None:
    graph = suspicious_accounts_graph()
    print("Neighbors for node 1:", sample_neighbors(graph.edge_index, node=1, max_neighbors=2))


if __name__ == "__main__":
    main()
