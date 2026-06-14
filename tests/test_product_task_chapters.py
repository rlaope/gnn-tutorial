from __future__ import annotations

import torch

from tests.chapter_test_utils import ROOT, load_module, run_script

PRODUCT_TASK_CHAPTERS = ["09-link-prediction", "10-graph-classification"]


def test_product_task_chapters_have_required_files() -> None:
    for chapter in PRODUCT_TASK_CHAPTERS:
        chapter_dir = ROOT / "chapters" / chapter
        for filename in ("README.md", "chapter.py", "exercise.py", "solution.py"):
            assert (chapter_dir / filename).is_file(), f"{chapter}/{filename} missing"


def test_product_task_chapter_scripts_smoke() -> None:
    expected_markers = {
        "09-link-prediction": "Link prediction recommendation trace",
        "10-graph-classification": "Graph classification pooling trace",
    }
    for chapter, marker in expected_markers.items():
        output = run_script(ROOT / "chapters" / chapter / "chapter.py")
        assert marker in output


def test_product_task_solution_scripts_smoke() -> None:
    for chapter in PRODUCT_TASK_CHAPTERS:
        output = run_script(ROOT / "chapters" / chapter / "solution.py")
        assert "TODO" not in output


def test_chapter_09_solution_scores_edges_by_dot_product() -> None:
    solution = load_module(
        ROOT / "chapters/09-link-prediction/solution.py",
        "chapter_09_solution",
    )
    embeddings = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    edge_index = torch.tensor([[0, 0], [1, 2]])
    assert solution.score_edges(embeddings, edge_index).tolist() == [0.5, 0.0]


def test_chapter_09_train_test_edges_are_disjoint() -> None:
    chapter = load_module(
        ROOT / "chapters/09-link-prediction/chapter.py",
        "chapter_09_chapter",
    )
    graph = chapter.bipartite_recommendation_graph()
    split = chapter.build_link_prediction_split(graph.edge_index)

    def as_edges(edge_index: torch.Tensor) -> set[tuple[int, int]]:
        return {(int(src), int(dst)) for src, dst in edge_index.t().tolist()}

    train_pos = as_edges(split.train_pos)
    train_neg = as_edges(split.train_neg)
    test_pos = as_edges(split.test_pos)
    test_neg = as_edges(split.test_neg)

    assert train_pos.isdisjoint(test_pos)
    assert train_neg.isdisjoint(test_neg)
    assert train_pos.isdisjoint(train_neg)
    assert train_pos.isdisjoint(test_neg)
    assert test_pos.isdisjoint(train_neg)
    assert test_pos.isdisjoint(test_neg)


def test_chapter_10_solution_mean_pool_keeps_batch_axis() -> None:
    solution = load_module(
        ROOT / "chapters/10-graph-classification/solution.py",
        "chapter_10_solution",
    )
    node_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    pooled = solution.mean_pool(node_embeddings)
    assert tuple(pooled.shape) == (1, 2)
    assert torch.allclose(pooled, torch.tensor([[2.0 / 3.0, 2.0 / 3.0]]))
