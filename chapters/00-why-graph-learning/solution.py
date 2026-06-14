"""Solution for chapter 00 task mapping."""

from __future__ import annotations

SCENARIOS = [
    ("suspicious account scoring", "account"),
    ("friend or item recommendation", "missing relationship"),
    ("molecule property prediction", "whole graph"),
]


def choose_graph_task(_product_area: str, target_object: str) -> str:
    """Map a product target object to the graph learning task family."""

    target = target_object.lower()
    if "missing relationship" in target or "edge" in target:
        return "link prediction"
    if "whole graph" in target or "molecule" in target:
        return "graph classification"
    return "node classification"


def main() -> None:
    for product_area, target_object in SCENARIOS:
        task = choose_graph_task(product_area, target_object)
        print(f"{product_area}: target={target_object!r} -> {task}")


if __name__ == "__main__":
    main()
