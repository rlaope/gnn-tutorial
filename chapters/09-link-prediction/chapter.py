"""Tiny user-item link prediction lab."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from graph_tutorial.datasets import bipartite_recommendation_graph
from graph_tutorial.graph_utils import edge_index_to_edge_list, edge_list_to_edge_index
from graph_tutorial.metrics import binary_link_metrics
from graph_tutorial.negative_sampling import sample_bipartite_negative_edges

USERS = range(0, 4)
ITEMS = range(4, 9)


class DotProductLinkPredictor(nn.Module):
    """One embedding table with dot-product edge scores."""

    def __init__(self, num_nodes: int, embedding_dim: int = 8) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        return score_edges(self.embeddings.weight, edge_index)


def score_edges(embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Score edges with source-target dot products."""

    src, dst = edge_index
    return (embeddings[src] * embeddings[dst]).sum(dim=1)


def unique_user_item_edges(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    """Return observed user-item pairs as undirected product edges."""

    return {
        (min(src, dst), max(src, dst))
        for src, dst in edge_index_to_edge_list(edge_index)
        if min(src, dst) in USERS and max(src, dst) in ITEMS
    }


def candidate_edges(existing_edge_index: torch.Tensor) -> torch.Tensor:
    """Return absent user-item pairs to rank as recommendations."""

    observed = unique_user_item_edges(existing_edge_index)
    candidates = [(user, item) for user in USERS for item in ITEMS if (user, item) not in observed]
    return edge_list_to_edge_index(candidates)


def train_link_predictor() -> dict[str, object]:
    """Train a tiny dot-product model and rank missing user-item edges."""

    torch.manual_seed(51)
    graph = bipartite_recommendation_graph()
    positive_edges = edge_list_to_edge_index(sorted(unique_user_item_edges(graph.edge_index)))
    train_pos = positive_edges[:, :-2]
    test_pos = positive_edges[:, -2:]
    train_neg = sample_bipartite_negative_edges(
        left_nodes=USERS,
        right_nodes=ITEMS,
        positive_edge_index=graph.edge_index,
        num_samples=train_pos.shape[1],
        seed=52,
    )
    test_neg = sample_bipartite_negative_edges(
        left_nodes=USERS,
        right_nodes=ITEMS,
        positive_edge_index=graph.edge_index,
        num_samples=test_pos.shape[1],
        seed=53,
    )
    train_edges = torch.cat([train_pos, train_neg], dim=1)
    train_labels = torch.cat([torch.ones(train_pos.shape[1]), torch.zeros(train_neg.shape[1])])

    model = DotProductLinkPredictor(graph.num_nodes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08, weight_decay=1e-3)
    losses: list[float] = []
    for _ in range(120):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(train_edges), train_labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    test_edges = torch.cat([test_pos, test_neg], dim=1)
    test_labels = torch.cat([torch.ones(test_pos.shape[1]), torch.zeros(test_neg.shape[1])])
    with torch.no_grad():
        test_scores = model(test_edges)
        metrics = binary_link_metrics(torch.sigmoid(test_scores), test_labels)
        candidates = candidate_edges(graph.edge_index)
        candidate_scores = torch.sigmoid(model(candidates))
    ranked = sorted(
        [
            (int(src), int(dst), float(score))
            for (src, dst), score in zip(candidates.t().tolist(), candidate_scores, strict=True)
        ],
        key=lambda row: row[2],
        reverse=True,
    )
    return {
        "loss_first": losses[0],
        "loss_final": losses[-1],
        "metrics": metrics,
        "top_recommendations": ranked[:4],
    }


def main() -> None:
    result = train_link_predictor()
    print("# Link prediction recommendation trace")
    print(f"loss: first={result['loss_first']:.4f} final={result['loss_final']:.4f}")
    print("metrics:", result["metrics"])
    print("top missing user-item edges:")
    for user, item, score in result["top_recommendations"]:
        print(f"user {user} -> item {item}: {score:.4f}")


if __name__ == "__main__":
    main()
