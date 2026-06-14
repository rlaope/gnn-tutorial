"""Solution for chapter 05 Node2Vec q bias."""

from __future__ import annotations

from graph_tutorial.datasets import toy_social_graph
from graph_tutorial.walks import node2vec_embeddings


def run_node2vec_variant(q: float):
    """Return a Node2Vec embedding result for the requested q value."""

    graph = toy_social_graph()
    return node2vec_embeddings(
        graph.edge_index,
        num_nodes=graph.num_nodes,
        walk_length=5,
        walks_per_node=2,
        p=1.0,
        q=q,
        seed=8,
    )


def main() -> None:
    for q in (2.0, 0.5):
        result = run_node2vec_variant(q)
        print(f"q={q}: first walk={result.walks[0]}")


if __name__ == "__main__":
    main()
