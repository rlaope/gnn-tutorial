"""Product-to-graph task mapping for the opening chapter."""

from __future__ import annotations

from dataclasses import dataclass

from graph_tutorial.datasets import (
    bipartite_recommendation_graph,
    suspicious_accounts_graph,
    tiny_graph_classification_dataset,
    toy_social_graph,
)
from graph_tutorial.graph_utils import edge_index_to_edge_list


@dataclass(frozen=True)
class ProductQuestion:
    product_question: str
    graph_object: str
    prediction_target: str
    chapter: str


def task_catalog() -> list[ProductQuestion]:
    """Return product-shaped graph tasks used across the tutorial."""

    return [
        ProductQuestion(
            "Which account looks risky because of shared devices?",
            "node",
            "node classification",
            "06-gcn / 07-graphsage",
        ),
        ProductQuestion(
            "Which user-item relation should we recommend next?",
            "edge",
            "link prediction",
            "09-link-prediction",
        ),
        ProductQuestion(
            "Which neighbor most influenced this score?",
            "edge neighborhood",
            "attention / influence inspection",
            "08-gat",
        ),
        ProductQuestion(
            "What label belongs to this whole molecule or workflow graph?",
            "whole graph",
            "graph classification",
            "10-graph-classification",
        ),
    ]


def graph_snapshots() -> list[tuple[str, int, int, str]]:
    """Summarize the tiny graphs that later chapters reuse."""

    datasets = [
        toy_social_graph(),
        suspicious_accounts_graph(),
        bipartite_recommendation_graph(),
        *tiny_graph_classification_dataset(),
    ]
    rows: list[tuple[str, int, int, str]] = []
    for graph in datasets:
        undirected_edges = {
            (min(src, dst), max(src, dst))
            for src, dst in edge_index_to_edge_list(graph.edge_index)
            if src != dst
        }
        label_state = "labels" if graph.y is not None else "no labels"
        rows.append((graph.name, graph.num_nodes, len(undirected_edges), label_state))
    return rows


def format_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    """Render a tiny markdown-style table without extra dependencies."""

    table = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    table.extend(" | ".join(str(value) for value in row) for row in rows)
    return "\n".join(table)


def main() -> None:
    catalog_rows = [
        (
            item.product_question,
            item.graph_object,
            item.prediction_target,
            item.chapter,
        )
        for item in task_catalog()
    ]
    print("# Product questions become graph tasks")
    print(format_table(("product question", "object", "target", "later chapter"), catalog_rows))
    print()
    print("# Tiny graph snapshots")
    print(format_table(("dataset", "nodes", "undirected edges", "labels"), graph_snapshots()))


if __name__ == "__main__":
    main()
