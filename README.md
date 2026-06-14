# gnn-tutorial

`gnn-tutorial` is a hands-on graph learning tutorial for building AI product features
with connected data: recommendation, suspicious-node classification, missing-relation
search foundations, link prediction, neighbor-influence inspection, and graph
classification.

The project follows the spirit of compact educational repositories such as
`graykode/nlp-tutorial`: each chapter focuses on one core idea, gives you runnable
code, and keeps the implementation close enough to the math that you can inspect it.
The topic is graph learning rather than NLP, and the target outcome is product
intuition: when you finish a chapter, you should know both what the model did and
what kind of product capability it could support.

## What This Is

This repository is a study path, not a production graph ML platform. v1 is made of
small learning labs. Each lab asks you to:

1. see the graph or tensor,
2. run a tiny example,
3. trace the important shapes and intermediate values,
4. break one modeling assumption,
5. fix it,
6. compare the idea with practical tooling where useful,
7. extend the task with a small exercise.

The first implementation path is from scratch with PyTorch, NetworkX, NumPy, and
small synthetic datasets. PyTorch Geometric belongs in optional comparison notes
after the mechanics are visible.

## v1 Scope Promise

v1 covers product-shaped learning labs, not production applications. The product
mapping is deliberately explicit so the tutorial does not overclaim.

| Product-shaped capability | Primary chapter | Boundary |
|---|---|---|
| Recommendation / connection suggestion | `09-link-prediction` | No ranking service or online serving. |
| Suspicious-node or risk classification | `06-gcn`, `07-graphsage` | No real transaction data, compliance workflow, or fraud system claim. |
| Neighbor influence inspection | `08-gat` | Attention visualization, not formal explainability guarantees. |
| Missing-relation search foundation | `09-link-prediction` | No production Graph RAG or knowledge graph search readiness. |
| Graph-level classification | `10-graph-classification` | Small graph examples, not large domain benchmarks. |

## Curriculum

| Chapter | Product lens | Core idea |
|---|---|---|
| `00-why-graph-learning` | Recognize connected-data product problems | Graph task taxonomy |
| `01-graph-basics` | Turn product records into nodes and edges | Edge lists, adjacency, features |
| `02-message-passing-by-hand` | See relational signal flow | Aggregation, self-loops, normalization |
| `03-random-walk` | Explore neighborhoods | Random walks over graph structure |
| `04-deepwalk` | Find similar nodes | Walks plus skip-gram-style training |
| `05-node2vec` | Tune similarity behavior | BFS/DFS-biased walks |
| `06-gcn` | Classify suspicious/high-risk nodes | Normalized neighbor aggregation |
| `07-graphsage` | Handle new users/items/accounts | Sampling and inductive learning |
| `08-gat` | Inspect influential neighbors | Attention over edges |
| `09-link-prediction` | Recommend and recover missing relations | Edge split, negative sampling, AUC/AP |
| `10-graph-classification` | Classify whole graphs | Pooling and graph-level prediction |

## Repository Layout

```text
gnn-tutorial/
  README.md
  HOW_TO_STUDY.md
  NOTEBOOK_POLICY.md
  pyproject.toml
  data/
  assets/images/
  src/graph_tutorial/
  chapters/
  tests/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,notebook]"
pytest
```

All default examples are designed to run on CPU. The quick tour should not require
large downloads or a GPU.

## Development Status

The implementation is driven by the durable plan in
`.omx/ultragoal/goals.json`. The approved planning artifacts live under `.omx/`.
The tutorial itself lives in top-level docs, `src/graph_tutorial`, `chapters`, and
`tests`.
