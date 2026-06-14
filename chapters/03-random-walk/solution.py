"""Solution for chapter 03 walk collection."""

from __future__ import annotations

import torch

from graph_tutorial.datasets import toy_social_graph
from graph_tutorial.walks import random_walk


def collect_training_walks(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    walk_length: int,
) -> list[list[int]]:
    """Return one deterministic walk starting from every node."""

    return [
        random_walk(
            edge_index,
            start=node,
            walk_length=walk_length,
            num_nodes=num_nodes,
            seed=10 + node,
        )
        for node in range(num_nodes)
    ]


def main() -> None:
    graph = toy_social_graph()
    walks = collect_training_walks(graph.edge_index, num_nodes=graph.num_nodes, walk_length=4)
    print("Collected walks:")
    for walk in walks:
        print(walk)


if __name__ == "__main__":
    main()
