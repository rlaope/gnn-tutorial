from __future__ import annotations

import torch
from torch import nn

from graph_tutorial.datasets import suspicious_accounts_graph, toy_social_graph
from graph_tutorial.gnn_layers import GCNLayer, GraphSAGELayer, SingleHeadGATLayer
from graph_tutorial.training import train_node_classifier


def test_gcn_and_graphsage_layers_keep_node_axis() -> None:
    graph = toy_social_graph()
    assert tuple(GCNLayer(2, 4)(graph.x, graph.edge_index).shape) == (graph.num_nodes, 4)
    assert tuple(GraphSAGELayer(2, 4)(graph.x, graph.edge_index).shape) == (
        graph.num_nodes,
        4,
    )


def test_gat_attention_sums_to_one_per_target_node() -> None:
    graph = toy_social_graph()
    layer = SingleHeadGATLayer(2, 3)
    output, edge_index, weights = layer(graph.x, graph.edge_index, return_attention=True)
    assert tuple(output.shape) == (graph.num_nodes, 3)
    for node in range(graph.num_nodes):
        incoming = edge_index[1] == node
        assert torch.isclose(weights[incoming].sum(), torch.tensor(1.0), atol=1e-5)


def test_train_node_classifier_returns_logits_and_losses() -> None:
    class TinyGCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = GCNLayer(3, 2)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            return self.layer(x, edge_index)

    torch.manual_seed(0)
    graph = suspicious_accounts_graph()
    result = train_node_classifier(TinyGCN(), graph, epochs=5, lr=0.01)
    assert tuple(result.logits.shape) == (graph.num_nodes, 2)
    assert len(result.losses) == 5
    assert result.losses[-1] <= result.losses[0] + 1e-4
