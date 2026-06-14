"""Small graph utilities used across tutorial chapters."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

Edge = tuple[int, int]


@dataclass(frozen=True)
class EdgeSplit:
    """Positive edge split for link-prediction labs."""

    train_pos: torch.Tensor
    val_pos: torch.Tensor
    test_pos: torch.Tensor


def edge_list_to_edge_index(
    edges: Iterable[Edge],
    *,
    undirected: bool = False,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert `(source, target)` pairs to a `[2, num_edges]` tensor."""

    pairs = [(int(src), int(dst)) for src, dst in edges]
    if undirected:
        pairs = pairs + [(dst, src) for src, dst in pairs if src != dst]
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()


def edge_index_to_edge_list(edge_index: torch.Tensor) -> list[Edge]:
    """Convert a `[2, num_edges]` tensor back to Python edge pairs."""

    _validate_edge_index(edge_index)
    return [(int(src), int(dst)) for src, dst in edge_index.t().tolist()]


def infer_num_nodes(edge_index: torch.Tensor, x: torch.Tensor | None = None) -> int:
    """Infer node count from features or the largest node id in `edge_index`."""

    if x is not None:
        return int(x.shape[0])
    if edge_index.numel() == 0:
        return 0
    return int(edge_index.max().item()) + 1


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Append one self-loop per node."""

    _validate_edge_index(edge_index)
    loops = torch.arange(num_nodes, device=edge_index.device, dtype=torch.long)
    loops = loops.unsqueeze(0).repeat(2, 1)
    return torch.cat([edge_index, loops], dim=1)


def degree(edge_index: torch.Tensor, num_nodes: int, *, incoming: bool = True) -> torch.Tensor:
    """Return in-degree by default, or out-degree when `incoming=False`."""

    _validate_edge_index(edge_index)
    node_ids = edge_index[1 if incoming else 0]
    out = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    if node_ids.numel() > 0:
        out.index_add_(0, node_ids, torch.ones_like(node_ids, dtype=torch.float32))
    return out


def edge_index_to_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create a dense adjacency matrix where `adj[src, dst] = weight`."""

    _validate_edge_index(edge_index)
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=edge_index.device)
    if edge_index.numel() == 0:
        return adj
    values = weights
    if values is None:
        values = torch.ones(edge_index.shape[1], dtype=torch.float32, device=edge_index.device)
    adj[edge_index[0], edge_index[1]] = values.float()
    return adj


def normalized_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    include_self_loops: bool = True,
) -> torch.Tensor:
    """Return dense symmetric GCN normalization: `D^-1/2 A D^-1/2`."""

    if include_self_loops:
        edge_index = add_self_loops(edge_index, num_nodes)
    adj = edge_index_to_adjacency(edge_index, num_nodes)
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.clamp(min=1).pow(-0.5)
    return deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]


def mean_aggregate(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int | None = None,
) -> torch.Tensor:
    """Average incoming neighbor features for each target node."""

    _validate_edge_index(edge_index)
    if num_nodes is None:
        num_nodes = infer_num_nodes(edge_index, x)
    out = torch.zeros((num_nodes, x.shape[1]), dtype=x.dtype, device=x.device)
    src, dst = edge_index
    if edge_index.numel() > 0:
        out.index_add_(0, dst, x[src])
    deg = degree(edge_index, num_nodes).clamp(min=1).to(x.dtype)
    return out / deg.unsqueeze(1)


def train_val_test_masks(
    num_nodes: int,
    *,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create boolean node masks with deterministic shuffling."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(num_nodes, generator=generator)
    train_end = int(num_nodes * train_ratio)
    val_end = train_end + int(num_nodes * val_ratio)
    masks = []
    for ids in (order[:train_end], order[train_end:val_end], order[val_end:]):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[ids] = True
        masks.append(mask)
    return tuple(masks)  # type: ignore[return-value]


def split_positive_edges(
    edge_index: torch.Tensor,
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 0,
    undirected: bool = True,
) -> EdgeSplit:
    """Split positive edges while de-duplicating undirected reverse pairs."""

    edges = _unique_edges(edge_index, undirected=undirected)
    if len(edges) < 3:
        raise ValueError("Need at least three positive edges to split")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(edges), generator=generator)
    shuffled = [edges[int(i)] for i in order]
    test_count = max(1, int(len(edges) * test_ratio))
    val_count = max(1, int(len(edges) * val_ratio))
    train = shuffled[: len(edges) - val_count - test_count]
    val = shuffled[len(edges) - val_count - test_count : len(edges) - test_count]
    test = shuffled[len(edges) - test_count :]
    return EdgeSplit(
        train_pos=edge_list_to_edge_index(train),
        val_pos=edge_list_to_edge_index(val),
        test_pos=edge_list_to_edge_index(test),
    )


def _unique_edges(edge_index: torch.Tensor, *, undirected: bool) -> list[Edge]:
    _validate_edge_index(edge_index)
    seen: set[Edge] = set()
    for src, dst in edge_index_to_edge_list(edge_index):
        key = (min(src, dst), max(src, dst)) if undirected else (src, dst)
        if src == dst:
            continue
        seen.add(key)
    return sorted(seen)


def _validate_edge_index(edge_index: torch.Tensor) -> None:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
