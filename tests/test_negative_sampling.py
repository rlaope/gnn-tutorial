from graph_tutorial.graph_utils import edge_list_to_edge_index
from graph_tutorial.negative_sampling import sample_negative_edges


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
