"""Metrics kept tiny and explicit for tutorial scripts."""

from __future__ import annotations

import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Classification accuracy for logits or class ids."""

    preds = logits.argmax(dim=-1) if logits.ndim > 1 else logits
    return float((preds == labels).float().mean().item())


def binary_link_metrics(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Return ROC-AUC and average precision for binary link prediction."""

    y_true = labels.detach().cpu().numpy()
    y_score = scores.detach().cpu().numpy()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
    }
