"""Solution for chapter 01 edge_index construction."""

from __future__ import annotations

import torch

from graph_tutorial.graph_utils import edge_list_to_edge_index

FRIENDSHIPS = [(0, 1), (1, 2), (2, 3)]


def build_missing_edge_index(friendships: list[tuple[int, int]]) -> torch.Tensor:
    """Represent each undirected friendship in both directions."""

    return edge_list_to_edge_index(friendships, undirected=True)


def main() -> None:
    edge_index = build_missing_edge_index(FRIENDSHIPS)
    print("Directed entries:", edge_index.shape[1])
    print(edge_index)


if __name__ == "__main__":
    main()
