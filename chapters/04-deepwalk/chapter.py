"""DeepWalk-style embeddings from a tiny walk corpus."""

from __future__ import annotations

from graph_tutorial.datasets import toy_social_graph
from graph_tutorial.walks import deepwalk_embeddings, nearest_neighbors


def run_deepwalk_lab() -> dict[str, object]:
    """Build embeddings and nearest-neighbor summaries."""

    graph = toy_social_graph()
    result = deepwalk_embeddings(
        graph.edge_index,
        num_nodes=graph.num_nodes,
        embedding_dim=2,
        walk_length=6,
        walks_per_node=4,
        window_size=2,
        seed=5,
    )
    return {
        "walk_count": len(result.walks),
        "counts_shape": tuple(result.counts.shape),
        "embedding_shape": tuple(result.embeddings.shape),
        "node_0_neighbors": nearest_neighbors(result.embeddings, node=0, top_k=2),
        "node_3_neighbors": nearest_neighbors(result.embeddings, node=3, top_k=2),
    }


def main() -> None:
    result = run_deepwalk_lab()
    print("# DeepWalk-style embedding trace")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
