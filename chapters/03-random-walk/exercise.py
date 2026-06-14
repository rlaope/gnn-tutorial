"""Exercise scaffold for collecting one walk from each node."""

from __future__ import annotations

import torch

from graph_tutorial.datasets import toy_social_graph


def collect_training_walks(
    _edge_index: torch.Tensor,
    *,
    num_nodes: int,
    _walk_length: int,
) -> list[list[int]]:
    """Return one random walk starting from every node.

    TODO: Use graph_tutorial.walks.random_walk and keep the seed deterministic.
    """

    return [[node] for node in range(num_nodes)]


def main() -> None:
    graph = toy_social_graph()
    walks = collect_training_walks(graph.edge_index, num_nodes=graph.num_nodes, _walk_length=4)
    print("Collected walks:")
    for walk in walks:
        print(walk)


if __name__ == "__main__":
    main()
