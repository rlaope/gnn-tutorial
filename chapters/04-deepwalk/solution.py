"""Solution for chapter 04 context counts."""

from __future__ import annotations

import torch

from graph_tutorial.datasets import toy_social_graph
from graph_tutorial.walks import cooccurrence_counts, factorize_ppmi, generate_walks


def build_context_counts(
    walks: list[list[int]],
    *,
    num_nodes: int,
    window_size: int = 2,
) -> torch.Tensor:
    """Return co-occurrence counts from walk windows."""

    return cooccurrence_counts(walks, num_nodes=num_nodes, window_size=window_size)


def main() -> None:
    graph = toy_social_graph()
    walks = generate_walks(graph.edge_index, num_nodes=graph.num_nodes, walk_length=4)
    counts = build_context_counts(walks, num_nodes=graph.num_nodes)
    embeddings = factorize_ppmi(counts, embedding_dim=2)
    print("Context count shape:", tuple(counts.shape))
    print("Embedding shape:", tuple(embeddings.shape))


if __name__ == "__main__":
    main()
