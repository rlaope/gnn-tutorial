"""Canonical curriculum metadata for docs, tests, and orientation scripts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chapter:
    """One tutorial chapter entry."""

    chapter_id: str
    product_lens: str
    core_idea: str


@dataclass(frozen=True)
class ProductScope:
    """Product-shaped capability and explicit v1 boundary."""

    capability: str
    primary_chapter: str
    boundary: str


@dataclass(frozen=True)
class StudyTrack:
    """A suggested path through chapter ids."""

    name: str
    chapter_ids: tuple[str, ...]


CURRICULUM: tuple[Chapter, ...] = (
    Chapter(
        "00-why-graph-learning",
        "Recognize connected-data product problems",
        "Graph task taxonomy",
    ),
    Chapter(
        "01-graph-basics",
        "Turn product records into nodes and edges",
        "Edge lists, adjacency, features",
    ),
    Chapter(
        "02-message-passing-by-hand",
        "See relational signal flow",
        "Aggregation, self-loops, normalization",
    ),
    Chapter("03-random-walk", "Explore neighborhoods", "Random walks over graph structure"),
    Chapter("04-deepwalk", "Find similar nodes", "Walks plus skip-gram-style training"),
    Chapter("05-node2vec", "Tune similarity behavior", "BFS/DFS-biased walks"),
    Chapter("06-gcn", "Classify suspicious/high-risk nodes", "Normalized neighbor aggregation"),
    Chapter("07-graphsage", "Handle new users/items/accounts", "Sampling and inductive learning"),
    Chapter("08-gat", "Inspect influential neighbors", "Attention over edges"),
    Chapter(
        "09-link-prediction",
        "Recommend and recover missing relations",
        "Edge split, negative sampling, AUC/AP",
    ),
    Chapter(
        "10-graph-classification",
        "Classify whole graphs",
        "Pooling and graph-level prediction",
    ),
)

PRODUCT_SCOPE: tuple[ProductScope, ...] = (
    ProductScope(
        "Recommendation / connection suggestion",
        "09-link-prediction",
        "No ranking service or online serving.",
    ),
    ProductScope(
        "Suspicious-node or risk classification",
        "06-gcn, 07-graphsage",
        "No real transaction data, compliance workflow, or fraud system claim.",
    ),
    ProductScope(
        "Neighbor influence inspection",
        "08-gat",
        "Attention visualization, not formal explainability guarantees.",
    ),
    ProductScope(
        "Missing-relation search foundation",
        "09-link-prediction",
        "No production Graph RAG or knowledge graph search readiness.",
    ),
    ProductScope(
        "Graph-level classification",
        "10-graph-classification",
        "Small graph examples, not large domain benchmarks.",
    ),
)

STUDY_TRACKS: tuple[StudyTrack, ...] = (
    StudyTrack(
        "Three-Hour Quick Tour",
        (
            "00-why-graph-learning",
            "01-graph-basics",
            "02-message-passing-by-hand",
            "09-link-prediction",
        ),
    ),
    StudyTrack(
        "From-Scratch Path",
        (
            "01-graph-basics",
            "02-message-passing-by-hand",
            "03-random-walk",
            "04-deepwalk",
            "05-node2vec",
            "06-gcn",
            "07-graphsage",
            "08-gat",
        ),
    ),
    StudyTrack(
        "Product Path",
        (
            "00-why-graph-learning",
            "06-gcn",
            "07-graphsage",
            "08-gat",
            "09-link-prediction",
            "10-graph-classification",
        ),
    ),
)


def chapter_ids() -> tuple[str, ...]:
    """Return chapter ids in canonical order."""

    return tuple(chapter.chapter_id for chapter in CURRICULUM)
