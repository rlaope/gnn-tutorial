import torch

from graph_tutorial.graph_utils import (
    add_self_loops,
    edge_index_to_adjacency,
    edge_list_to_edge_index,
    mean_aggregate,
    normalized_adjacency,
    split_positive_edges,
    train_val_test_masks,
)


def test_edge_list_to_edge_index_adds_reverse_edges() -> None:
    edge_index = edge_list_to_edge_index([(0, 1), (1, 2)], undirected=True)
    assert edge_index.tolist() == [[0, 1, 1, 2], [1, 2, 0, 1]]


def test_self_loops_and_adjacency() -> None:
    edge_index = edge_list_to_edge_index([(0, 1)])
    with_loops = add_self_loops(edge_index, num_nodes=2)
    adj = edge_index_to_adjacency(with_loops, num_nodes=2)
    assert torch.equal(adj, torch.tensor([[1.0, 1.0], [0.0, 1.0]]))


def test_normalized_adjacency_has_self_signal() -> None:
    edge_index = edge_list_to_edge_index([(0, 1), (1, 0)])
    norm = normalized_adjacency(edge_index, num_nodes=2)
    assert norm.shape == (2, 2)
    assert torch.all(norm.diag() > 0)


def test_mean_aggregate_uses_incoming_neighbors() -> None:
    x = torch.tensor([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]])
    edge_index = edge_list_to_edge_index([(0, 2), (1, 2)])
    out = mean_aggregate(x, edge_index, num_nodes=3)
    assert torch.allclose(out[2], torch.tensor([2.0, 1.0]))


def test_train_val_test_masks_partition_nodes() -> None:
    train, val, test = train_val_test_masks(10, train_ratio=0.5, val_ratio=0.2, seed=1)
    assert int(train.sum()) == 5
    assert int(val.sum()) == 2
    assert int(test.sum()) == 3
    assert not torch.any(train & val)
    assert not torch.any(train & test)
    assert not torch.any(val & test)


def test_split_positive_edges_has_no_overlap() -> None:
    edge_index = edge_list_to_edge_index([(0, 1), (1, 2), (2, 3), (3, 4)], undirected=True)
    split = split_positive_edges(edge_index, seed=4)
    groups = [
        {tuple(edge) for edge in split.train_pos.t().tolist()},
        {tuple(edge) for edge in split.val_pos.t().tolist()},
        {tuple(edge) for edge in split.test_pos.t().tolist()},
    ]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
