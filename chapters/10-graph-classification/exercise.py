"""Exercise scaffold for graph-level mean pooling."""

from __future__ import annotations

import torch


def mean_pool(node_embeddings: torch.Tensor) -> torch.Tensor:
    """Return one graph embedding from all node embeddings.

    TODO: Average across the node dimension and keep a batch dimension.
    """

    return node_embeddings[:1]


def main() -> None:
    node_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    print("pooled:", mean_pool(node_embeddings).tolist())


if __name__ == "__main__":
    main()
