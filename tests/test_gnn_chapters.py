from __future__ import annotations

import torch

from tests.chapter_test_utils import ROOT, load_module, run_script

GNN_CHAPTERS = ["06-gcn", "07-graphsage", "08-gat"]


def test_gnn_chapters_have_required_files() -> None:
    for chapter in GNN_CHAPTERS:
        chapter_dir = ROOT / "chapters" / chapter
        for filename in ("README.md", "chapter.py", "exercise.py", "solution.py"):
            assert (chapter_dir / filename).is_file(), f"{chapter}/{filename} missing"


def test_gnn_chapter_scripts_smoke() -> None:
    expected_markers = {
        "06-gcn": "GCN suspicious-node trace",
        "07-graphsage": "GraphSAGE new-node trace",
        "08-gat": "GAT neighbor-attention trace",
    }
    for chapter, marker in expected_markers.items():
        output = run_script(ROOT / "chapters" / chapter / "chapter.py")
        assert marker in output


def test_gnn_solution_scripts_smoke() -> None:
    for chapter in GNN_CHAPTERS:
        output = run_script(ROOT / "chapters" / chapter / "solution.py")
        assert "TODO" not in output


def test_chapter_06_solution_changes_features_with_gcn_aggregation() -> None:
    solution = load_module(ROOT / "chapters/06-gcn/solution.py", "chapter_06_solution")
    graph_module = load_module(ROOT / "chapters/06-gcn/chapter.py", "chapter_06_chapter")
    graph = graph_module.suspicious_accounts_graph()
    aggregated = solution.apply_gcn_aggregation(graph.x, graph.edge_index)
    assert tuple(aggregated.shape) == tuple(graph.x.shape)
    assert not torch.allclose(aggregated, graph.x)


def test_chapter_07_solution_samples_bounded_neighbors() -> None:
    solution = load_module(ROOT / "chapters/07-graphsage/solution.py", "chapter_07_solution")
    graph_module = load_module(ROOT / "chapters/07-graphsage/chapter.py", "chapter_07_chapter")
    graph = graph_module.suspicious_accounts_graph()
    assert solution.sample_neighbors(graph.edge_index, node=1, max_neighbors=2) == [4, 5]


def test_chapter_08_solution_sorts_attention_weights() -> None:
    solution = load_module(ROOT / "chapters/08-gat/solution.py", "chapter_08_solution")
    edge_index = torch.tensor([[0, 1, 2], [2, 2, 1]])
    weights = torch.tensor([0.2, 0.8, 1.0])
    assert solution.top_incoming_attention(edge_index, weights, target_node=2) == [
        (1, 2, 0.800000011920929),
        (0, 2, 0.20000000298023224),
    ]
