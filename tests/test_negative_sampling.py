from graph_tutorial.graph_utils import edge_list_to_edge_index
from graph_tutorial.negative_sampling import sample_bipartite_negative_edges, sample_negative_edges


def test_negative_sampling_avoids_positive_edges_and_self_loops() -> None:
    positives = edge_list_to_edge_index([(0, 1), (1, 2)], undirected=True)
    negatives = sample_negative_edges(
        num_nodes=4,
        positive_edge_index=positives,
        num_samples=2,
        seed=3,
    )
    positive_set = {tuple(sorted(edge)) for edge in positives.t().tolist()}
    for src, dst in negatives.t().tolist():
        assert src != dst
        assert tuple(sorted((src, dst))) not in positive_set


def test_sample_bipartite_negative_edges_stays_across_partitions() -> None:
    positives = edge_list_to_edge_index([(0, 3), (1, 4)], undirected=True)
    negatives = sample_bipartite_negative_edges(
        left_nodes=range(0, 2),
        right_nodes=range(3, 5),
        positive_edge_index=positives,
        num_samples=2,
        seed=1,
    )
    sampled = {(int(src), int(dst)) for src, dst in negatives.t().tolist()}
    assert sampled == {(0, 4), (1, 3)}
