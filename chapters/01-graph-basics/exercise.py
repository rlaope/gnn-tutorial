"""Exercise scaffold for creating an undirected edge_index tensor."""

from __future__ import annotations

import torch

from graph_tutorial.graph_utils import edge_list_to_edge_index

FRIENDSHIPS = [(0, 1), (1, 2), (2, 3)]


def build_missing_edge_index(_friendships: list[tuple[int, int]]) -> torch.Tensor:
    """Return a directed edge_index for an undirected friendship graph.

    TODO: Use edge_list_to_edge_index with the right option so every edge appears
    in both directions.
    """

    return edge_list_to_edge_index([])


def main() -> None:
    edge_index = build_missing_edge_index(FRIENDSHIPS)
    print("Expected directed entries:", len(FRIENDSHIPS) * 2)
    print("Current directed entries:", edge_index.shape[1])
    print(edge_index)


if __name__ == "__main__":
    main()
