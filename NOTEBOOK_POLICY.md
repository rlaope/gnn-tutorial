# Notebook Policy

Notebooks are optional companions to the Python scripts.

## Rules

- Every chapter must have runnable Python code first.
- A notebook may mirror the chapter script, add visual explanation, or show a
  guided walkthrough.
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
