from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_CHAPTERS = ["03-random-walk", "04-deepwalk", "05-node2vec"]


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


def test_embedding_chapters_have_required_files() -> None:
    for chapter in EMBEDDING_CHAPTERS:
        chapter_dir = ROOT / "chapters" / chapter
        for filename in ("README.md", "chapter.py", "exercise.py", "solution.py"):
            assert (chapter_dir / filename).is_file(), f"{chapter}/{filename} missing"


def test_embedding_chapter_scripts_smoke() -> None:
    expected_markers = {
        "03-random-walk": "Random walk neighborhood trace",
        "04-deepwalk": "DeepWalk-style embedding trace",
        "05-node2vec": "Node2Vec bias trace",
    }
    for chapter, marker in expected_markers.items():
        output = run_script(ROOT / "chapters" / chapter / "chapter.py")
        assert marker in output


def test_embedding_solution_scripts_smoke() -> None:
    for chapter in EMBEDDING_CHAPTERS:
        output = run_script(ROOT / "chapters" / chapter / "solution.py")
        assert "TODO" not in output


def test_chapter_03_solution_collects_walks_from_every_node() -> None:
    solution = load_module(ROOT / "chapters/03-random-walk/solution.py", "chapter_03_solution")
    graph_module = load_module(ROOT / "chapters/03-random-walk/chapter.py", "chapter_03_chapter")
    graph = graph_module.toy_social_graph()
    walks = solution.collect_training_walks(
        graph.edge_index,
        num_nodes=graph.num_nodes,
        walk_length=4,
    )
    assert len(walks) == graph.num_nodes
    assert [walk[0] for walk in walks] == list(range(graph.num_nodes))
    assert all(len(walk) == 4 for walk in walks)


def test_chapter_04_solution_builds_nonzero_context_counts() -> None:
    solution = load_module(ROOT / "chapters/04-deepwalk/solution.py", "chapter_04_solution")
    graph_module = load_module(ROOT / "chapters/04-deepwalk/chapter.py", "chapter_04_chapter")
    graph = graph_module.toy_social_graph()
    walks = graph_module.deepwalk_embeddings(graph.edge_index, num_nodes=graph.num_nodes).walks
    counts = solution.build_context_counts(walks, num_nodes=graph.num_nodes, window_size=2)
    assert tuple(counts.shape) == (graph.num_nodes, graph.num_nodes)
    assert float(counts.sum().item()) > 0


def test_chapter_05_solution_passes_q_into_sampler() -> None:
    solution = load_module(ROOT / "chapters/05-node2vec/solution.py", "chapter_05_solution")
    local = solution.run_node2vec_variant(2.0)
    explore = solution.run_node2vec_variant(0.5)
    assert tuple(local.embeddings.shape) == tuple(explore.embeddings.shape)
    assert torch.isfinite(local.embeddings).all()
    assert local.walks != explore.walks
