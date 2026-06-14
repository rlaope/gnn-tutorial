"""Exercise scaffold for building DeepWalk context counts."""

from __future__ import annotations

import torch

from graph_tutorial.datasets import toy_social_graph
from graph_tutorial.walks import generate_walks


def build_context_counts(_walks: list[list[int]], *, num_nodes: int) -> torch.Tensor:
    """Return co-occurrence counts from walk windows.

    TODO: Pass a window_size of at least 1.
    """

    return torch.zeros((num_nodes, num_nodes), dtype=torch.float32)


def main() -> None:
    graph = toy_social_graph()
    walks = generate_walks(graph.edge_index, num_nodes=graph.num_nodes, walk_length=4)
    counts = build_context_counts(walks, num_nodes=graph.num_nodes)
    print("Context count shape:", tuple(counts.shape))
    print("Total context pairs:", int(counts.sum().item()))


if __name__ == "__main__":
    main()
