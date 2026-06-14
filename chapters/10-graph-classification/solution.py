"""Solution for chapter 10 graph-level mean pooling."""

from __future__ import annotations

import torch


def mean_pool(node_embeddings: torch.Tensor) -> torch.Tensor:
    """Average node embeddings into one graph embedding."""

    return node_embeddings.mean(dim=0, keepdim=True)


def main() -> None:
    node_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    print("pooled:", mean_pool(node_embeddings).tolist())


if __name__ == "__main__":
    main()
