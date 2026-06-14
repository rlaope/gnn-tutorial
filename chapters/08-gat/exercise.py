"""Exercise scaffold for reading incoming attention weights."""

from __future__ import annotations

import torch


def top_incoming_attention(
    _edge_index: torch.Tensor,
    _weights: torch.Tensor,
    *,
    _target_node: int,
    _top_k: int = 3,
) -> list[tuple[int, int, float]]:
    """Return top incoming attention edges for a target node.

    TODO: Filter edges where dst == target_node and sort by weight descending.
    """

    return []


def main() -> None:
    edge_index = torch.tensor([[0, 1, 2], [2, 2, 1]])
    weights = torch.tensor([0.2, 0.8, 1.0])
    print(top_incoming_attention(edge_index, weights, _target_node=2))


if __name__ == "__main__":
    main()
