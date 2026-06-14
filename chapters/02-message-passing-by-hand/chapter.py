"""One round of message passing with inspectable tensors."""

from __future__ import annotations

import torch

from graph_tutorial.datasets import toy_social_graph
from graph_tutorial.graph_utils import (
    add_self_loops,
    mean_aggregate,
    normalized_adjacency,
)


def run_message_passing() -> dict[str, torch.Tensor]:
    """Compute neighbor-only, self-loop, and normalized aggregation."""

    graph = toy_social_graph()
    edge_index_with_self = add_self_loops(graph.edge_index, graph.num_nodes)
    norm_adj = normalized_adjacency(graph.edge_index, graph.num_nodes)
    return {
        "original": graph.x,
        "neighbor_mean": mean_aggregate(graph.x, graph.edge_index, graph.num_nodes),
        "self_loop_mean": mean_aggregate(graph.x, edge_index_with_self, graph.num_nodes),
        "normalized_once": norm_adj @ graph.x,
    }


def main() -> None:
    outputs = run_message_passing()
    print("# Node 0 trace")
    for name, tensor in outputs.items():
        print(f"{name}: {tensor[0].round(decimals=4).tolist()}")
    print("\n# Full self-loop mean aggregation")
    print(outputs["self_loop_mean"].round(decimals=4))


if __name__ == "__main__":
    main()
