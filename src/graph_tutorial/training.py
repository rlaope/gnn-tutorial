"""Small training loops shared by chapter scripts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from graph_tutorial.datasets import GraphData
from graph_tutorial.metrics import accuracy


@dataclass(frozen=True)
class NodeTrainingResult:
    """Final logits and compact training diagnostics."""

    logits: torch.Tensor
    losses: list[float]
    train_accuracy: float
    val_accuracy: float | None
    test_accuracy: float | None


def train_node_classifier(
    model: nn.Module,
    graph: GraphData,
    *,
    epochs: int = 80,
    lr: float = 0.05,
    weight_decay: float = 0.0,
) -> NodeTrainingResult:
    """Train a tiny node classifier on the graph's train mask."""

    if graph.y is None:
        raise ValueError("graph.y is required for node classification")
    if graph.train_mask is None:
        train_mask = torch.ones(graph.num_nodes, dtype=torch.bool)
    else:
        train_mask = graph.train_mask

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(graph.x, graph.edge_index)
        loss = F.cross_entropy(logits[train_mask], graph.y[train_mask])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        logits = model(graph.x, graph.edge_index)
    return NodeTrainingResult(
        logits=logits,
        losses=losses,
        train_accuracy=_masked_accuracy(logits, graph.y, train_mask),
        val_accuracy=_masked_accuracy(logits, graph.y, graph.val_mask),
        test_accuracy=_masked_accuracy(logits, graph.y, graph.test_mask),
    )


def _masked_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None,
) -> float | None:
    if mask is None or int(mask.sum().item()) == 0:
        return None
    return accuracy(logits[mask], labels[mask])
