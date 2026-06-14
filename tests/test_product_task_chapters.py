from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PRODUCT_TASK_CHAPTERS = ["09-link-prediction", "10-graph-classification"]


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


def test_chapter_10_solution_mean_pool_keeps_batch_axis() -> None:
    solution = load_module(
        ROOT / "chapters/10-graph-classification/solution.py",
        "chapter_10_solution",
    )
    node_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    pooled = solution.mean_pool(node_embeddings)
    assert tuple(pooled.shape) == (1, 2)
    assert torch.allclose(pooled, torch.tensor([[2.0 / 3.0, 2.0 / 3.0]]))
