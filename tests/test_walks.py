from __future__ import annotations

import torch

from graph_tutorial.datasets import toy_social_graph
from graph_tutorial.graph_utils import edge_list_to_edge_index
from graph_tutorial.walks import (
    adjacency_lists,
    biased_random_walk,
    cooccurrence_counts,
    deepwalk_embeddings,
    generate_walks,
    nearest_neighbors,
    node2vec_transition_weights,
    ppmi_matrix,
    random_walk,
)


def test_random_walk_is_deterministic_and_valid() -> None:
    graph = toy_social_graph()
    walk = random_walk(graph.edge_index, start=0, walk_length=5, num_nodes=graph.num_nodes, seed=1)
    assert walk == random_walk(
        graph.edge_index,
        start=0,
        walk_length=5,
        num_nodes=graph.num_nodes,
        seed=1,
    )
    neighbors = adjacency_lists(graph.edge_index, graph.num_nodes)
    assert len(walk) == 5
    assert all(dst in neighbors[src] for src, dst in zip(walk[:-1], walk[1:], strict=True))


def test_generate_walks_and_counts_have_expected_shapes() -> None:
    graph = toy_social_graph()
    walks = generate_walks(graph.edge_index, num_nodes=graph.num_nodes, walk_length=4)
    counts = cooccurrence_counts(walks, num_nodes=graph.num_nodes, window_size=2)
    assert len(walks) == graph.num_nodes * 2
    assert tuple(counts.shape) == (graph.num_nodes, graph.num_nodes)
    assert float(counts.sum().item()) > 0
    assert torch.all(ppmi_matrix(counts) >= 0)


def test_node2vec_transition_weights_reflect_return_and_explore_bias() -> None:
    edge_index = edge_list_to_edge_index([(0, 1), (1, 2), (1, 3)], undirected=True)
    candidates, weights = node2vec_transition_weights(
        edge_index,
        previous=0,
        current=1,
        p=0.5,
        q=2.0,
        num_nodes=4,
    )
    by_candidate = dict(zip(candidates, weights.tolist(), strict=True))
    assert by_candidate[0] == 2.0
    assert by_candidate[2] == 0.5
    assert by_candidate[3] == 0.5


def test_biased_walk_and_deepwalk_embeddings_are_finite() -> None:
    graph = toy_social_graph()
    walk = biased_random_walk(
        graph.edge_index,
        start=0,
        walk_length=5,
        p=1.0,
        q=0.5,
        num_nodes=graph.num_nodes,
        seed=2,
    )
    result = deepwalk_embeddings(graph.edge_index, num_nodes=graph.num_nodes, embedding_dim=2)
    assert len(walk) == 5
    assert tuple(result.embeddings.shape) == (graph.num_nodes, 2)
    assert torch.isfinite(result.embeddings).all()
    assert len(nearest_neighbors(result.embeddings, node=0, top_k=2)) == 2
