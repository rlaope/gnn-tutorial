from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FOUNDATION_CHAPTERS = [
    "00-why-graph-learning",
    "01-graph-basics",
    "02-message-passing-by-hand",
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(path: Path) -> str:
    env = os.environ.copy()
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_foundation_chapters_have_required_files() -> None:
    for chapter in FOUNDATION_CHAPTERS:
        chapter_dir = ROOT / "chapters" / chapter
        for filename in ("README.md", "chapter.py", "exercise.py", "solution.py"):
            assert (chapter_dir / filename).is_file(), f"{chapter}/{filename} missing"


def test_foundation_chapter_scripts_smoke() -> None:
    expected_markers = {
        "00-why-graph-learning": "Product questions become graph tasks",
        "01-graph-basics": "Tensor trace",
        "02-message-passing-by-hand": "Node 0 trace",
    }
    for chapter, marker in expected_markers.items():
        output = run_script(ROOT / "chapters" / chapter / "chapter.py")
        assert marker in output


def test_foundation_solution_scripts_smoke() -> None:
    for chapter in FOUNDATION_CHAPTERS:
        output = run_script(ROOT / "chapters" / chapter / "solution.py")
        assert "TODO" not in output


def test_chapter_00_solution_maps_task_families() -> None:
    solution = load_module(
        ROOT / "chapters/00-why-graph-learning/solution.py",
        "chapter_00_solution",
    )
    assert solution.choose_graph_task("risk", "account") == "node classification"
    assert solution.choose_graph_task("recommendation", "missing relationship") == "link prediction"
    assert solution.choose_graph_task("chemistry", "whole graph") == "graph classification"


def test_chapter_01_solution_builds_undirected_edges() -> None:
    solution = load_module(
        ROOT / "chapters/01-graph-basics/solution.py",
        "chapter_01_solution",
    )
    edge_index = solution.build_missing_edge_index([(0, 1), (1, 2)])
    assert tuple(edge_index.shape) == (2, 4)
    assert {(int(src), int(dst)) for src, dst in edge_index.t().tolist()} == {
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
    }


def test_chapter_02_solution_keeps_self_signal() -> None:
    solution = load_module(
        ROOT / "chapters/02-message-passing-by-hand/solution.py",
        "chapter_02_solution",
    )
    graph_module = load_module(
        ROOT / "chapters/02-message-passing-by-hand/chapter.py",
        "chapter_02_chapter",
    )
    graph = graph_module.toy_social_graph()
    updated = solution.aggregate_with_self_loops(graph.x, graph.edge_index, graph.num_nodes)
    neighbor_only = graph_module.mean_aggregate(graph.x, graph.edge_index, graph.num_nodes)
    assert tuple(updated.shape) == tuple(graph.x.shape)
    assert torch.linalg.vector_norm(updated[0] - graph.x[0]) < torch.linalg.vector_norm(
        neighbor_only[0] - graph.x[0]
    )
