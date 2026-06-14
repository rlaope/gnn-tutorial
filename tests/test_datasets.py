from graph_tutorial.datasets import (
    bipartite_recommendation_graph,
    karate_club_graph,
    suspicious_accounts_graph,
    tiny_graph_classification_dataset,
    toy_social_graph,
)


def test_toy_datasets_have_expected_shapes() -> None:
    for graph in [toy_social_graph(), karate_club_graph(), suspicious_accounts_graph()]:
        assert graph.x.shape[0] == graph.num_nodes
        assert graph.edge_index.shape[0] == 2
        assert graph.y is not None


def test_bipartite_recommendation_graph_is_unlabeled() -> None:
    graph = bipartite_recommendation_graph()
    assert graph.num_nodes == 9
    assert graph.y is None


def test_graph_classification_dataset_has_graph_labels() -> None:
    graphs = tiny_graph_classification_dataset()
    assert len(graphs) == 2
    assert {int(graph.y.item()) for graph in graphs if graph.y is not None} == {0, 1}
