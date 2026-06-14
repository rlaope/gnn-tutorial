"""Generate README tutorial links and chapter notebook wrappers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_tutorial.curriculum import (  # noqa: E402
    TUTORIAL_SECTIONS,
    chapter_ids,
    tutorial_link_by_id,
)

README_START = "<!-- tutorial-links:start -->"
README_END = "<!-- tutorial-links:end -->"


def render_tutorial_links() -> str:
    """Render the generated README tutorial links block."""

    links = tutorial_link_by_id()
    lines: list[str] = []
    for section in TUTORIAL_SECTIONS:
        lines.append(f"#### {section.title}")
        lines.append("")
        for chapter_id in section.chapter_ids:
            link = links[chapter_id]
            chapter_dir = f"chapters/{chapter_id}"
            notebook_path = f"notebooks/{chapter_id}.ipynb"
            chapter_number = chapter_id.split("-", maxsplit=1)[0]
            lines.append(
                f"- {chapter_number}. [{link.title}]({chapter_dir}/README.md) - {link.summary}"
            )
            lines.append(f"  - Code - [chapter.py]({chapter_dir}/chapter.py)")
            lines.append(f"  - Notebook - [Jupyter]({notebook_path})")
            lines.append(
                "  - Practice - "
                f"[exercise.py]({chapter_dir}/exercise.py), "
                f"[solution.py]({chapter_dir}/solution.py)"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _markdown_cell(source: list[str], cell_id: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source,
    }


def _code_cell(source: list[str], cell_id: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def render_notebook(chapter_id: str) -> dict[str, object]:
    """Render one lightweight notebook wrapper."""

    link = tutorial_link_by_id()[chapter_id]
    return {
        "cells": [
            _markdown_cell(
                [
                    f"# {chapter_id} - {link.title}\n",
                    "\n",
                    f"{link.summary}\n",
                    "\n",
                    "- Chapter guide: "
                    f"[`chapters/{chapter_id}/README.md`](../chapters/{chapter_id}/README.md)\n",
                    f"- Code: [`chapter.py`](../chapters/{chapter_id}/chapter.py)\n",
                    f"- Exercise: [`exercise.py`](../chapters/{chapter_id}/exercise.py)\n",
                    f"- Solution: [`solution.py`](../chapters/{chapter_id}/solution.py)\n",
                ],
                f"{chapter_id}-intro",
            ),
            _markdown_cell(
                [
                    "This notebook is a lightweight companion to the versioned Python scripts. ",
                    "Run the chapter first, inspect the output, then try the exercise before "
                    "opening the solution.\n",
                ],
                f"{chapter_id}-workflow",
            ),
            _code_cell(
                [
                    "import runpy\n",
                    "from pathlib import Path\n",
                    "\n",
                    "REPO_ROOT = Path.cwd()\n",
                    "if not (REPO_ROOT / 'chapters').exists():\n",
                    "    REPO_ROOT = REPO_ROOT.parent\n",
                    f"CHAPTER_DIR = REPO_ROOT / 'chapters/{chapter_id}'\n",
                    "CHAPTER_DIR\n",
                ],
                f"{chapter_id}-setup",
            ),
            _markdown_cell(["## Run the chapter\n"], f"{chapter_id}-run-heading"),
            _code_cell(
                ["runpy.run_path(str(CHAPTER_DIR / 'chapter.py'), run_name='__main__')\n"],
                f"{chapter_id}-run",
            ),
            _markdown_cell(["## Try the exercise\n"], f"{chapter_id}-exercise-heading"),
            _code_cell(
                ["runpy.run_path(str(CHAPTER_DIR / 'exercise.py'), run_name='__main__')\n"],
                f"{chapter_id}-exercise",
            ),
            _markdown_cell(
                ["## Compare with the solution\n"],
                f"{chapter_id}-solution-heading",
            ),
            _code_cell(
                ["runpy.run_path(str(CHAPTER_DIR / 'solution.py'), run_name='__main__')\n"],
                f"{chapter_id}-solution",
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render_notebook_text(chapter_id: str) -> str:
    """Render one notebook as stable JSON text."""

    return json.dumps(render_notebook(chapter_id), indent=1) + "\n"


def sync_readme(root: Path = ROOT) -> None:
    """Replace the generated tutorial links block in README."""

    readme_path = root / "README.md"
    readme = readme_path.read_text()
    before, rest = readme.split(README_START, maxsplit=1)
    _, after = rest.split(README_END, maxsplit=1)
    readme_path.write_text(
        before + README_START + "\n" + render_tutorial_links() + README_END + after
    )


def sync_notebooks(root: Path = ROOT) -> None:
    """Write chapter notebook wrappers."""

    notebooks_dir = root / "notebooks"
    notebooks_dir.mkdir(exist_ok=True)
    for chapter_id in chapter_ids():
        (notebooks_dir / f"{chapter_id}.ipynb").write_text(render_notebook_text(chapter_id))


def main() -> None:
    sync_readme()
    sync_notebooks()


if __name__ == "__main__":
    main()
