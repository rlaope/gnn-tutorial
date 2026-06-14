from __future__ import annotations

import re
from pathlib import Path

from graph_tutorial.curriculum import CURRICULUM, PRODUCT_SCOPE, STUDY_TRACKS, chapter_ids

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = chapter_ids()


def test_readme_curriculum_points_to_existing_chapters() -> None:
    readme = (ROOT / "README.md").read_text()
    for chapter in CHAPTERS:
        chapter_dir = ROOT / "chapters" / chapter
        assert chapter in readme
        assert chapter_dir.is_dir()
        for filename in ("README.md", "chapter.py", "exercise.py", "solution.py"):
            assert (chapter_dir / filename).is_file(), f"{chapter}/{filename} missing"


def test_readme_curriculum_matches_manifest_order() -> None:
    readme = (ROOT / "README.md").read_text()
    rows = re.findall(r"\| `([^`]+)` \| ([^|]+) \| ([^|]+) \|", readme)
    assert rows == [
        (chapter.chapter_id, chapter.product_lens, chapter.core_idea)
        for chapter in CURRICULUM
    ]


def test_readme_product_scope_matches_manifest() -> None:
    readme = (ROOT / "README.md").read_text()
    rows = re.findall(r"\| ([^|`][^|]+?) \| (.+?) \| ([^|]+?) \|", readme)
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


def test_completed_chapters_do_not_need_gitkeep_placeholders() -> None:
    for chapter in CHAPTERS:
        assert not (ROOT / "chapters" / chapter / ".gitkeep").exists()
