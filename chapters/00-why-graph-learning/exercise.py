"""Exercise scaffold for mapping product questions to graph tasks."""

from __future__ import annotations

SCENARIOS = [
    ("suspicious account scoring", "account"),
    ("friend or item recommendation", "missing relationship"),
    ("molecule property prediction", "whole graph"),
]


def choose_graph_task(_product_area: str, _target_object: str) -> str:
    """Return node classification, link prediction, or graph classification.

    TODO: Replace the placeholder with a rule based on the target object.
    """

    return "TODO"


def main() -> None:
    print("Fill choose_graph_task so each row returns a graph task.")
    for product_area, target_object in SCENARIOS:
        task = choose_graph_task(product_area, target_object)
        print(f"{product_area}: target={target_object!r} -> {task}")


if __name__ == "__main__":
    main()
