"""Exercise scaffold for edge dot-product scoring."""

from __future__ import annotations

import torch


def score_edges(embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Return one score per edge.

    TODO: Multiply source and target embeddings elementwise, then sum per edge.
    """

    return torch.zeros(edge_index.shape[1])


def main() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    edge_index = torch.tensor([[0, 0], [1, 2]])
    print("scores:", score_edges(embeddings, edge_index).tolist())


if __name__ == "__main__":
    main()
