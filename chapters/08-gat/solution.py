"""Solution for chapter 08 incoming attention inspection."""

from __future__ import annotations

import torch


def top_incoming_attention(
    edge_index: torch.Tensor,
    weights: torch.Tensor,
    *,
    target_node: int,
    top_k: int = 3,
) -> list[tuple[int, int, float]]:
    """Return top incoming attention edges for a target node."""

    rows: list[tuple[int, int, float]] = []
    for idx, (src, dst) in enumerate(edge_index.t().tolist()):
        if int(dst) == target_node:
            rows.append((int(src), int(dst), float(weights[idx])))
    return sorted(rows, key=lambda row: row[2], reverse=True)[:top_k]


def main() -> None:
    edge_index = torch.tensor([[0, 1, 2], [2, 2, 1]])
    weights = torch.tensor([0.2, 0.8, 1.0])
    print(top_incoming_attention(edge_index, weights, target_node=2))


if __name__ == "__main__":
    main()
