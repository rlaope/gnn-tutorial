"""Solution for chapter 09 edge scoring."""

from __future__ import annotations

import torch


def score_edges(embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Return source-target dot-product scores."""

    src, dst = edge_index
    return (embeddings[src] * embeddings[dst]).sum(dim=1)


def main() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    edge_index = torch.tensor([[0, 0], [1, 2]])
    print("scores:", score_edges(embeddings, edge_index).tolist())


if __name__ == "__main__":
    main()
