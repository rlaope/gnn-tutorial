"""Tiny deterministic datasets for the tutorial chapters."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import torch

from graph_tutorial.graph_utils import edge_list_to_edge_index, train_val_test_masks


@dataclass
class GraphData:
    """Minimal graph container used before introducing full graph libraries."""

    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor | None = None
    train_mask: torch.Tensor | None = None
    val_mask: torch.Tensor | None = None
    test_mask: torch.Tensor | None = None
    name: str = "graph"

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])


def toy_social_graph() -> GraphData:
    """Small homophily graph for basics and message-passing labs."""

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (0, 2),
        (3, 5),
        (1, 4),
    ]
    x = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.2, 0.8],
            [0.0, 1.0],
        ]
    )
    y = torch.tensor([0, 0, 0, 1, 1, 1])
    masks = train_val_test_masks(num_nodes=6, train_ratio=0.5, val_ratio=0.17, seed=7)
    return GraphData(
        x=x,
        edge_index=edge_list_to_edge_index(edges, undirected=True),
        y=y,
        name="toy_social",
        train_mask=masks[0],
        val_mask=masks[1],
        test_mask=masks[2],
    )


def karate_club_graph() -> GraphData:
    """NetworkX Karate Club graph with simple identity-like features."""

    graph = nx.karate_club_graph()
    edges = [(int(src), int(dst)) for src, dst in graph.edges()]
    clubs = [graph.nodes[node]["club"] for node in graph.nodes()]
    y = torch.tensor([0 if club == "Mr. Hi" else 1 for club in clubs], dtype=torch.long)
    x = torch.eye(graph.number_of_nodes(), dtype=torch.float32)
    masks = train_val_test_masks(graph.number_of_nodes(), train_ratio=0.5, val_ratio=0.25, seed=3)
    return GraphData(
        x=x,
        edge_index=edge_list_to_edge_index(edges, undirected=True),
        y=y,
        train_mask=masks[0],
        val_mask=masks[1],
        test_mask=masks[2],
        name="karate_club",
    )


def suspicious_accounts_graph() -> GraphData:
    """Synthetic account-device-transaction proxy for safe risk classification."""

    edges = [
        (0, 4),
        (1, 4),
        (1, 5),
        (2, 5),
        (2, 6),
        (3, 6),
        (3, 7),
        (0, 7),
        (8, 4),
        (8, 5),
        (9, 6),
        (9, 7),
    ]
    # First four rows are normal accounts, 8-9 are suspicious accounts, 4-7 are devices.
    x = torch.tensor(
        [
            [1.0, 0.0, 0.1],
            [1.0, 0.0, 0.2],
            [1.0, 0.0, 0.2],
            [1.0, 0.0, 0.1],
            [0.0, 1.0, 0.3],
            [0.0, 1.0, 0.4],
            [0.0, 1.0, 0.4],
            [0.0, 1.0, 0.3],
            [1.0, 0.0, 0.9],
            [1.0, 0.0, 0.8],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.long)
    masks = train_val_test_masks(num_nodes=10, train_ratio=0.5, val_ratio=0.2, seed=11)
    return GraphData(
        x=x,
        edge_index=edge_list_to_edge_index(edges, undirected=True),
        y=y,
        train_mask=masks[0],
        val_mask=masks[1],
        test_mask=masks[2],
        name="suspicious_accounts",
    )


def bipartite_recommendation_graph() -> GraphData:
    """Tiny user-item graph for link-prediction examples."""

    # Users: 0-3, items: 4-8.
    edges = [(0, 4), (0, 5), (1, 5), (1, 6), (2, 6), (2, 7), (3, 7), (3, 8)]
    x = torch.eye(9, dtype=torch.float32)
    return GraphData(
        x=x,
        edge_index=edge_list_to_edge_index(edges, undirected=True),
        name="bipartite_recommendation",
    )


def tiny_graph_classification_dataset() -> list[GraphData]:
    """Two toy graph classes: chains and triangles."""

    graphs: list[GraphData] = []
    for idx, edges in enumerate(([(0, 1), (1, 2)], [(0, 1), (1, 2), (2, 0)])):
        x = torch.ones((3, 1), dtype=torch.float32)
        y = torch.tensor([idx], dtype=torch.long)
        graphs.append(
            GraphData(
                x=x,
                edge_index=edge_list_to_edge_index(edges, undirected=True),
                y=y,
                name=f"toy_graph_{idx}",
            )
        )
    return graphs
