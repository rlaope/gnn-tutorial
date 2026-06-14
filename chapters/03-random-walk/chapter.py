"""Random walks as neighborhood samplers."""

from __future__ import annotations

import torch

from graph_tutorial.datasets import toy_social_graph
from graph_tutorial.walks import adjacency_lists, cooccurrence_counts, generate_walks, random_walk


def run_walk_lab() -> dict[str, object]:
    """Create a deterministic walk corpus for the toy graph."""

    graph = toy_social_graph()
    walks = generate_walks(
        graph.edge_index,
        num_nodes=graph.num_nodes,
        walk_length=5,
        walks_per_node=1,
        seed=4,
    )
    return {
        "neighbors": adjacency_lists(graph.edge_index, graph.num_nodes),
        "walk_from_0": random_walk(
            graph.edge_index,
            start=0,
            walk_length=6,
            num_nodes=graph.num_nodes,
            seed=3,
        ),
        "walks": walks,
        "counts": cooccurrence_counts(walks, num_nodes=graph.num_nodes, window_size=1),
    }


def main() -> None:
    result = run_walk_lab()
    print("# Random walk neighborhood trace")
    print("neighbors:", result["neighbors"])
    print("walk from node 0:", result["walk_from_0"])
    print("training walks:", result["walks"])
    print("window=1 co-occurrence counts:")
    print(result["counts"].to(torch.int64))


if __name__ == "__main__":
    main()
