from __future__ import annotations

import json
import re
from pathlib import Path

from graph_tutorial.curriculum import (
    BUILDER_OUTCOMES,
    CURRICULUM,
    PRODUCT_SCOPE,
    STUDY_TRACKS,
    chapter_ids,
)

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = chapter_ids()


def test_readme_curriculum_points_to_existing_chapters() -> None:
    readme = (ROOT / "README.md").read_text()
    for chapter in CHAPTERS:
        chapter_dir = ROOT / "chapters" / chapter
        notebook_path = ROOT / "notebooks" / f"{chapter}.ipynb"
        assert chapter in readme
        assert f"notebooks/{chapter}.ipynb" in readme
        assert chapter_dir.is_dir()
        assert notebook_path.is_file(), f"{chapter}.ipynb missing"
        for filename in ("README.md", "chapter.py", "exercise.py", "solution.py"):
            assert (chapter_dir / filename).is_file(), f"{chapter}/{filename} missing"
            assert f"chapters/{chapter}/{filename}" in readme


def test_readme_curriculum_matches_manifest_order() -> None:
    readme = (ROOT / "README.md").read_text()
    curriculum_section = readme.split("## Curriculum", maxsplit=1)[1].split(
        "## Repository Layout",
        maxsplit=1,
    )[0]
    rows = re.findall(r"\| `([^`]+)` \| ([^|]+) \| ([^|]+) \|", curriculum_section)
    assert rows == [
        (chapter.chapter_id, chapter.product_lens, chapter.core_idea)
        for chapter in CURRICULUM
    ]


def test_readme_product_scope_matches_manifest() -> None:
    readme = (ROOT / "README.md").read_text()
    scope_section = readme.split("## v1 Scope Promise", maxsplit=1)[1].split(
        "## Builder Outcome",
        maxsplit=1,
    )[0]
    rows = re.findall(r"\| ([^|`][^|]+?) \| (.+?) \| ([^|]+?) \|", scope_section)
    scope_rows = [
        (
            capability.strip(),
            re.sub(r"`([^`]+)`", r"\1", primary_chapter).strip(),
            boundary.strip(),
        )
        for capability, primary_chapter, boundary in rows
        if capability.strip() != "Product-shaped capability"
    ][: len(PRODUCT_SCOPE)]
    assert scope_rows == [
        (scope.capability, scope.primary_chapter, scope.boundary)
        for scope in PRODUCT_SCOPE
    ]


def test_study_tracks_reference_only_existing_chapters() -> None:
    study_doc = (ROOT / "HOW_TO_STUDY.md").read_text()
    references = set(re.findall(r"\b\d{2}-[a-z0-9-]+\b", study_doc))
    assert references
    assert references <= set(CHAPTERS)
    for track in STUDY_TRACKS:
        for chapter_id in track.chapter_ids:
            assert chapter_id in study_doc


def test_builder_outcomes_are_documented_and_reference_existing_chapters() -> None:
    guide = (ROOT / "PRODUCT_BUILDER_GUIDE.md").read_text()
    readme = (ROOT / "README.md").read_text()
    study_doc = (ROOT / "HOW_TO_STUDY.md").read_text()

    seen_chapters: list[str] = []
    for outcome in BUILDER_OUTCOMES:
        assert outcome.goal_id in readme
        assert outcome.goal_id in study_doc
        assert outcome.goal_id in guide
        assert outcome.role in guide
        assert outcome.artifact in guide
        assert outcome.product_capability in guide
        for chapter_id in outcome.chapter_ids:
            assert chapter_id in CHAPTERS
            assert chapter_id in guide
            seen_chapters.append(chapter_id)

    assert tuple(seen_chapters) == CHAPTERS


def test_public_docs_do_not_reference_local_agent_artifacts() -> None:
    public_docs = [
        ROOT / "README.md",
        ROOT / "HOW_TO_STUDY.md",
        ROOT / "PRODUCT_BUILDER_GUIDE.md",
        ROOT / "NOTEBOOK_POLICY.md",
    ]
    for path in public_docs:
        assert ".omx" not in path.read_text()


def test_notebooks_are_lightweight_chapter_wrappers() -> None:
    for chapter in CHAPTERS:
        notebook_path = ROOT / "notebooks" / f"{chapter}.ipynb"
        notebook = json.loads(notebook_path.read_text())
        assert notebook["nbformat"] == 4
        assert notebook["metadata"]["kernelspec"]["language"] == "python"

        source_text = "\n".join(
            "".join(cell.get("source", ""))
            for cell in notebook["cells"]
        )
        assert chapter in source_text
        assert f"chapters/{chapter}/README.md" in source_text
        assert "runpy.run_path" in source_text
        assert "chapter.py" in source_text
        assert "exercise.py" in source_text
        assert "solution.py" in source_text

        for cell in notebook["cells"]:
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None


def test_completed_chapters_do_not_need_gitkeep_placeholders() -> None:
    for chapter in CHAPTERS:
        assert not (ROOT / "chapters" / chapter / ".gitkeep").exists()
