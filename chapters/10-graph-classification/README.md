# 10 - Graph Classification

## Goal

Classify whole graphs by applying a small GCN layer, pooling node embeddings, and
predicting a graph-level label.

## Product Relevance

Graph classification supports product features where the object is a complete
connected structure: molecules, workflows, transaction subgraphs, code dependency
graphs, or user journey graphs. The boundary: this lab uses tiny synthetic graph
shapes, not benchmark-scale graph datasets.

## Concept

Node-level models output one representation per node. Graph classification adds a
readout step, such as mean pooling, to turn all node embeddings into one graph
embedding before classification.

## Run

```bash
python chapters/10-graph-classification/chapter.py
python chapters/10-graph-classification/exercise.py
python chapters/10-graph-classification/solution.py
```

## Trace

Inspect each graph's node count, label, predicted class, final loss, and graph-level
accuracy. Node features include normalized degree plus a repeated graph-density hint
so the readout step has visible graph-level signal.

## Break It

Use only the first node embedding as the graph representation. The classifier ignores
most of the graph.

## Fix It

Complete `mean_pool` in `exercise.py`.

## Compare

Real graph-classification pipelines may use global mean/max/add pooling, hierarchical
pooling, or domain-specific readouts. This chapter keeps mean pooling visible.

## Extend

Add a star graph to the dataset and decide whether it belongs with chains or closed
cycle-like graphs.

## Check Your Understanding

- Why does node classification not directly solve graph classification?
- What does pooling change about the tensor shape?
- What graph-level products need a whole-graph label?
- Why is this toy dataset not enough for a scientific claim?

## Files

- `chapter.py`
- `exercise.py`
- `solution.py`
- `README.md`

## Expected Result

The chapter script prints graph-level predictions for a tiny shape dataset.
