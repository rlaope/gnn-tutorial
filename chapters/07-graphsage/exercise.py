"""Exercise scaffold for bounded neighbor sampling."""

from __future__ import annotations

import torch

from graph_tutorial.datasets import suspicious_accounts_graph


def sample_neighbors(_edge_index: torch.Tensor, *, _node: int, _max_neighbors: int) -> list[int]:
    """Return up to max_neighbors sorted neighbors for node.

    TODO: Build adjacency lists and slice the selected node's neighbors.
    """

    return []


def main() -> None:
    graph = suspicious_accounts_graph()
    print("Neighbors for node 1:", sample_neighbors(graph.edge_index, _node=1, _max_neighbors=2))


if __name__ == "__main__":
    main()
