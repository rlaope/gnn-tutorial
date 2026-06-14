# gnn-tutorial

`gnn-tutorial` is a hands-on graph learning tutorial for building AI product features
with connected data: recommendation, suspicious-node classification, missing-relation
search foundations, link prediction, neighbor-influence inspection, and graph
classification.

Each chapter focuses on one core idea, links directly to runnable code and notebooks,
and keeps the implementation close enough to the math that you can inspect it.

## Overview

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

The target outcome is product intuition: when you finish a chapter, you should know
both what the model did and what kind of product capability it could support.

## Tutorial Links

#### 1. Graph Foundations

- 00. [Why Graph Learning](chapters/00-why-graph-learning/README.md) - Map product questions to graph tasks.
  - Code - [chapter.py](chapters/00-why-graph-learning/chapter.py)
  - Notebook - [Jupyter](notebooks/00-why-graph-learning.ipynb)
  - Practice - [exercise.py](chapters/00-why-graph-learning/exercise.py), [solution.py](chapters/00-why-graph-learning/solution.py)
- 01. [Graph Basics](chapters/01-graph-basics/README.md) - Turn records into `edge_index`, features, labels, and adjacency.
  - Code - [chapter.py](chapters/01-graph-basics/chapter.py)
  - Notebook - [Jupyter](notebooks/01-graph-basics.ipynb)
  - Practice - [exercise.py](chapters/01-graph-basics/exercise.py), [solution.py](chapters/01-graph-basics/solution.py)
- 02. [Message Passing By Hand](chapters/02-message-passing-by-hand/README.md) - Trace aggregation, self-loops, and normalization.
  - Code - [chapter.py](chapters/02-message-passing-by-hand/chapter.py)
  - Notebook - [Jupyter](notebooks/02-message-passing-by-hand.ipynb)
  - Practice - [exercise.py](chapters/02-message-passing-by-hand/exercise.py), [solution.py](chapters/02-message-passing-by-hand/solution.py)

#### 2. Walks and Embeddings

- 03. [Random Walk](chapters/03-random-walk/README.md) - Sample local neighborhoods as sequences.
  - Code - [chapter.py](chapters/03-random-walk/chapter.py)
  - Notebook - [Jupyter](notebooks/03-random-walk.ipynb)
  - Practice - [exercise.py](chapters/03-random-walk/exercise.py), [solution.py](chapters/03-random-walk/solution.py)
- 04. [DeepWalk](chapters/04-deepwalk/README.md) - Convert walk contexts into node embeddings.
  - Code - [chapter.py](chapters/04-deepwalk/chapter.py)
  - Notebook - [Jupyter](notebooks/04-deepwalk.ipynb)
  - Practice - [exercise.py](chapters/04-deepwalk/exercise.py), [solution.py](chapters/04-deepwalk/solution.py)
- 05. [Node2Vec](chapters/05-node2vec/README.md) - Tune local-vs-outward similarity with biased walks.
  - Code - [chapter.py](chapters/05-node2vec/chapter.py)
  - Notebook - [Jupyter](notebooks/05-node2vec.ipynb)
  - Practice - [exercise.py](chapters/05-node2vec/exercise.py), [solution.py](chapters/05-node2vec/solution.py)

#### 3. Message-Passing GNNs

- 06. [GCN](chapters/06-gcn/README.md) - Train a suspicious-node classifier with normalized aggregation.
  - Code - [chapter.py](chapters/06-gcn/chapter.py)
  - Notebook - [Jupyter](notebooks/06-gcn.ipynb)
  - Practice - [exercise.py](chapters/06-gcn/exercise.py), [solution.py](chapters/06-gcn/solution.py)
- 07. [GraphSAGE](chapters/07-graphsage/README.md) - Score new nodes with sampled neighborhood features.
  - Code - [chapter.py](chapters/07-graphsage/chapter.py)
  - Notebook - [Jupyter](notebooks/07-graphsage.ipynb)
  - Practice - [exercise.py](chapters/07-graphsage/exercise.py), [solution.py](chapters/07-graphsage/solution.py)
- 08. [GAT](chapters/08-gat/README.md) - Inspect neighbor attention weights.
  - Code - [chapter.py](chapters/08-gat/chapter.py)
  - Notebook - [Jupyter](notebooks/08-gat.ipynb)
  - Practice - [exercise.py](chapters/08-gat/exercise.py), [solution.py](chapters/08-gat/solution.py)

#### 4. Product Graph Tasks

- 09. [Link Prediction](chapters/09-link-prediction/README.md) - Rank missing user-item relations as recommendation candidates.
  - Code - [chapter.py](chapters/09-link-prediction/chapter.py)
  - Notebook - [Jupyter](notebooks/09-link-prediction.ipynb)
  - Practice - [exercise.py](chapters/09-link-prediction/exercise.py), [solution.py](chapters/09-link-prediction/solution.py)
- 10. [Graph Classification](chapters/10-graph-classification/README.md) - Pool node embeddings into graph-level predictions.
  - Code - [chapter.py](chapters/10-graph-classification/chapter.py)
  - Notebook - [Jupyter](notebooks/10-graph-classification.ipynb)
  - Practice - [exercise.py](chapters/10-graph-classification/exercise.py), [solution.py](chapters/10-graph-classification/solution.py)

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

## Builder Outcome

Use [PRODUCT_BUILDER_GUIDE.md](PRODUCT_BUILDER_GUIDE.md) to connect the chapter
sequence to portfolio artifacts and product-builder roles. The intended progression
is:

| Goal slice | Builder role | Output |
|---|---|---|
| `G003` | Graph problem modeler | Graph task brief and tensor trace. |
| `G004` | Graph embedding candidate generator | Walk corpus, embeddings, and nearest-neighbor comparison. |
| `G005` | Message-passing GNN prototyper | Node classifier, new-node inference, and attention inspection. |
| `G006` | Graph product task builder | Recommendation candidates and graph-level classifier report. |

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
  PRODUCT_BUILDER_GUIDE.md
  NOTEBOOK_POLICY.md
  pyproject.toml
  data/
  assets/images/
  src/graph_tutorial/
  chapters/
  notebooks/
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

The versioned tutorial lives in top-level docs, `src/graph_tutorial`, `chapters`,
and `tests`. Local agent/runtime artifacts are intentionally excluded from git.
