"""Node2Vec biased walks for tuning graph similarity."""

from __future__ import annotations

from graph_tutorial.datasets import toy_social_graph
from graph_tutorial.walks import nearest_neighbors, node2vec_embeddings


def summarize_variant(q: float) -> dict[str, object]:
    """Run a Node2Vec-style variant and summarize the result."""

    graph = toy_social_graph()
    result = node2vec_embeddings(
        graph.edge_index,
        num_nodes=graph.num_nodes,
        embedding_dim=2,
        walk_length=6,
        walks_per_node=4,
        window_size=2,
        p=1.0,
        q=q,
        seed=6,
    )
    return {
        "q": q,
        "first_walk": result.walks[0],
        "node_0_neighbors": nearest_neighbors(result.embeddings, node=0, top_k=2),
    }


def main() -> None:
    print("# Node2Vec bias trace")
    for variant in (summarize_variant(q=2.0), summarize_variant(q=0.5)):
        print(variant)


if __name__ == "__main__":
    main()
