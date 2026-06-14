"""Negative edge sampling helpers for link-prediction chapters."""

from __future__ import annotations

import random

import torch

from graph_tutorial.graph_utils import Edge, edge_index_to_edge_list, edge_list_to_edge_index


def sample_negative_edges(
    *,
    num_nodes: int,
    positive_edge_index: torch.Tensor,
    num_samples: int,
    undirected: bool = True,
    exclude_self_loops: bool = True,
    seed: int = 0,
) -> torch.Tensor:
    """Sample node pairs absent from `positive_edge_index`."""

    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    blocked = _blocked_edges(positive_edge_index, undirected=undirected)
    candidates: list[Edge] = []
    for src in range(num_nodes):
        for dst in range(num_nodes):
            if exclude_self_loops and src == dst:
                continue
            key = (min(src, dst), max(src, dst)) if undirected else (src, dst)
            if key not in blocked:
                candidates.append((src, dst))
    if undirected:
        candidates = [(src, dst) for src, dst in candidates if src < dst]
    if num_samples > len(candidates):
        raise ValueError("num_samples exceeds available negative edges")
    rng = random.Random(seed)
    return edge_list_to_edge_index(rng.sample(candidates, num_samples))


def _blocked_edges(edge_index: torch.Tensor, *, undirected: bool) -> set[Edge]:
    blocked: set[Edge] = set()
    for src, dst in edge_index_to_edge_list(edge_index):
        key = (min(src, dst), max(src, dst)) if undirected else (src, dst)
        blocked.add(key)
    return blocked
