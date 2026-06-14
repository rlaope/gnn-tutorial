"""Random-walk and walk-embedding helpers for small graph labs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from graph_tutorial.graph_utils import edge_index_to_edge_list, infer_num_nodes


@dataclass(frozen=True)
class WalkEmbeddingResult:
    """Walk corpus plus a small matrix-factorized embedding."""

    walks: list[list[int]]
    counts: torch.Tensor
    embeddings: torch.Tensor


def adjacency_lists(edge_index: torch.Tensor, num_nodes: int | None = None) -> dict[int, list[int]]:
    """Return sorted neighbors for every node."""

    if num_nodes is None:
        num_nodes = infer_num_nodes(edge_index)
    neighbors: dict[int, set[int]] = {node: set() for node in range(num_nodes)}
    for src, dst in edge_index_to_edge_list(edge_index):
        if src != dst:
            neighbors[src].add(dst)
    return {node: sorted(values) for node, values in neighbors.items()}


def random_walk(
    edge_index: torch.Tensor,
    *,
    start: int,
    walk_length: int,
    num_nodes: int | None = None,
    seed: int = 0,
) -> list[int]:
    """Sample a uniform random walk with `walk_length` visited nodes."""

    if walk_length < 1:
        raise ValueError("walk_length must be at least 1")
    neighbors = adjacency_lists(edge_index, num_nodes)
    generator = torch.Generator().manual_seed(seed)
    current = int(start)
    walk = [current]
    for _ in range(walk_length - 1):
        candidates = neighbors[current]
        if not candidates:
            break
        next_id = int(torch.randint(len(candidates), (1,), generator=generator).item())
        current = candidates[next_id]
        walk.append(current)
    return walk


def generate_walks(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    walk_length: int,
    walks_per_node: int = 2,
    seed: int = 0,
) -> list[list[int]]:
    """Generate uniform random walks from every node."""

    walks: list[list[int]] = []
    for repeat in range(walks_per_node):
        for node in range(num_nodes):
            walks.append(
                random_walk(
                    edge_index,
                    start=node,
                    walk_length=walk_length,
                    num_nodes=num_nodes,
                    seed=seed + repeat * num_nodes + node,
                )
            )
    return walks


def node2vec_transition_weights(
    edge_index: torch.Tensor,
    *,
    previous: int | None,
    current: int,
    p: float,
    q: float,
    num_nodes: int | None = None,
) -> tuple[list[int], torch.Tensor]:
    """Return Node2Vec candidate neighbors and unnormalized transition weights."""

    if p <= 0 or q <= 0:
        raise ValueError("p and q must be positive")
    neighbors = adjacency_lists(edge_index, num_nodes)
    candidates = neighbors[current]
    if previous is None:
        return candidates, torch.ones(len(candidates), dtype=torch.float32)

    previous_neighbors = set(neighbors[previous])
    weights = []
    for candidate in candidates:
        if candidate == previous:
            weights.append(1.0 / p)
        elif candidate in previous_neighbors:
            weights.append(1.0)
        else:
            weights.append(1.0 / q)
    return candidates, torch.tensor(weights, dtype=torch.float32)


def biased_random_walk(
    edge_index: torch.Tensor,
    *,
    start: int,
    walk_length: int,
    p: float = 1.0,
    q: float = 1.0,
    num_nodes: int | None = None,
    seed: int = 0,
) -> list[int]:
    """Sample a Node2Vec-style random walk."""

    if walk_length < 1:
        raise ValueError("walk_length must be at least 1")
    if num_nodes is None:
        num_nodes = infer_num_nodes(edge_index)
    generator = torch.Generator().manual_seed(seed)
    previous: int | None = None
    current = int(start)
    walk = [current]
    for _ in range(walk_length - 1):
        candidates, weights = node2vec_transition_weights(
            edge_index,
            previous=previous,
            current=current,
            p=p,
            q=q,
            num_nodes=num_nodes,
        )
        if not candidates:
            break
        choice = int(torch.multinomial(weights, 1, generator=generator).item())
        previous, current = current, candidates[choice]
        walk.append(current)
    return walk


def generate_biased_walks(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    walk_length: int,
    walks_per_node: int = 2,
    p: float = 1.0,
    q: float = 1.0,
    seed: int = 0,
) -> list[list[int]]:
    """Generate Node2Vec-style walks from every node."""

    walks: list[list[int]] = []
    for repeat in range(walks_per_node):
        for node in range(num_nodes):
            walks.append(
                biased_random_walk(
                    edge_index,
                    start=node,
                    walk_length=walk_length,
                    p=p,
                    q=q,
                    num_nodes=num_nodes,
                    seed=seed + repeat * num_nodes + node,
                )
            )
    return walks


def cooccurrence_counts(
    walks: list[list[int]],
    *,
    num_nodes: int,
    window_size: int,
) -> torch.Tensor:
    """Count node-context pairs within a sliding window over walks."""

    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    counts = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for walk in walks:
        for center_pos, center in enumerate(walk):
            left = max(0, center_pos - window_size)
            right = min(len(walk), center_pos + window_size + 1)
            for context_pos in range(left, right):
                if context_pos == center_pos:
                    continue
                counts[center, walk[context_pos]] += 1.0
    return counts


def ppmi_matrix(counts: torch.Tensor, *, eps: float = 1e-9) -> torch.Tensor:
    """Convert co-occurrence counts into positive PMI values."""

    total = counts.sum()
    if float(total.item()) == 0.0:
        return torch.zeros_like(counts)
    row_sum = counts.sum(dim=1, keepdim=True)
    col_sum = counts.sum(dim=0, keepdim=True)
    numerator = counts * total
    denominator = row_sum @ col_sum
    pmi = torch.log((numerator + eps) / (denominator + eps))
    return torch.where(counts > 0, pmi.clamp(min=0), torch.zeros_like(pmi))


def factorize_ppmi(counts: torch.Tensor, *, embedding_dim: int = 2) -> torch.Tensor:
    """Create small embeddings by truncated SVD of the PPMI matrix."""

    if embedding_dim < 1:
        raise ValueError("embedding_dim must be at least 1")
    ppmi = ppmi_matrix(counts)
    if float(ppmi.sum().item()) == 0.0:
        return torch.zeros((counts.shape[0], embedding_dim), dtype=torch.float32)
    u, singular_values, _ = torch.linalg.svd(ppmi)
    dim = min(embedding_dim, singular_values.shape[0])
    embeddings = u[:, :dim] * singular_values[:dim].sqrt().unsqueeze(0)
    if dim < embedding_dim:
        padding = torch.zeros((counts.shape[0], embedding_dim - dim), dtype=embeddings.dtype)
        embeddings = torch.cat([embeddings, padding], dim=1)
    return embeddings


def deepwalk_embeddings(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    embedding_dim: int = 2,
    walk_length: int = 6,
    walks_per_node: int = 4,
    window_size: int = 2,
    seed: int = 0,
) -> WalkEmbeddingResult:
    """Build a DeepWalk-style context matrix and factorize it."""

    walks = generate_walks(
        edge_index,
        num_nodes=num_nodes,
        walk_length=walk_length,
        walks_per_node=walks_per_node,
        seed=seed,
    )
    counts = cooccurrence_counts(walks, num_nodes=num_nodes, window_size=window_size)
    return WalkEmbeddingResult(walks=walks, counts=counts, embeddings=factorize_ppmi(counts))


def node2vec_embeddings(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    embedding_dim: int = 2,
    walk_length: int = 6,
    walks_per_node: int = 4,
    window_size: int = 2,
    p: float = 1.0,
    q: float = 1.0,
    seed: int = 0,
) -> WalkEmbeddingResult:
    """Build Node2Vec-style biased-walk embeddings."""

    walks = generate_biased_walks(
        edge_index,
        num_nodes=num_nodes,
        walk_length=walk_length,
        walks_per_node=walks_per_node,
        p=p,
        q=q,
        seed=seed,
    )
    counts = cooccurrence_counts(walks, num_nodes=num_nodes, window_size=window_size)
    embeddings = factorize_ppmi(counts, embedding_dim=embedding_dim)
    return WalkEmbeddingResult(walks=walks, counts=counts, embeddings=embeddings)


def nearest_neighbors(
    embeddings: torch.Tensor,
    *,
    node: int,
    top_k: int = 2,
) -> list[tuple[int, float]]:
    """Return nearest nodes by cosine similarity."""

    sims = F.normalize(embeddings, dim=1) @ F.normalize(embeddings, dim=1).t()
    sims[node, node] = -2.0
    limit = min(top_k, embeddings.shape[0] - 1)
    values, indices = torch.topk(sims[node], k=limit)
    return [(int(index), float(value)) for index, value in zip(indices, values, strict=True)]
