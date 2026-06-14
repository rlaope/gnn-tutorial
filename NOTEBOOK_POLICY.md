# Notebook Policy

Notebooks are optional companions to the Python scripts. Chapter notebooks live in
`notebooks/` and mirror the matching `chapters/<chapter-id>/` files.

## Rules

- Every chapter must have runnable Python code first.
- A notebook may mirror the chapter script, add visual explanation, or show a
  guided walkthrough.
- Notebook code should call the versioned chapter scripts instead of duplicating
  the tutorial implementation.
- When chapter metadata changes, run `python tools/sync_tutorial_assets.py` so
  README links and notebook wrappers stay aligned with the curriculum manifest.
- Notebook outputs should be lightweight. Avoid committing large generated data.
- Quick-tour notebooks should run on CPU.
- If a notebook is too expensive or too flaky for automated execution, the chapter
  README must say how it was manually verified.

## Verification

The default automated checks are:

```bash
python -m compileall .
pytest
```

Notebook execution can be added later as a focused CI job once the chapter scripts
are stable.
