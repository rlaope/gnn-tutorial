# 01 - Graph Basics

## Goal

Turn product records into the minimum graph tensors used by the rest of the tutorial:
node features `x`, edge list `edge_index`, labels `y`, adjacency, and degree.

## Product Relevance

This chapter supports the data modeling step behind recommendation, risk scoring, and
relationship search. The boundary: the example is deliberately tiny and dense enough
to inspect by eye.

## Concept

A graph learning lab usually begins with three choices:

1. what counts as a node,
2. what counts as an edge,
3. what numeric features describe each node.

In this tutorial, edges are stored as a PyTorch tensor with shape `[2, num_edges]`.
The first row is source nodes; the second row is target nodes.

## Run

```bash
python chapters/01-graph-basics/chapter.py
python chapters/01-graph-basics/exercise.py
python chapters/01-graph-basics/solution.py
```

## Trace

Inspect `edge_index.shape`, `x.shape`, the adjacency matrix, and the degree vector.
Those four objects are enough to understand most later examples.

## Break It

Use only one direction for an undirected social edge. Node 1 may point to node 2, but
node 2 will not receive the reverse signal.

## Fix It

Complete `build_missing_edge_index` in `exercise.py` so every friendship is represented
in both directions.

## Compare

Libraries such as PyTorch Geometric also use `edge_index`, but this chapter keeps the
conversion visible before any library abstraction appears.

## Extend

Add one feature column for "days since signup" and check how `x.shape` changes.

## Check Your Understanding

- Which tensor stores relationships?
- Why does an undirected edge create two directed entries in `edge_index`?
- What does the degree vector count?
- What product mistake happens if node ids are inconsistent across tables?

## Files

- `chapter.py`
- `exercise.py`
- `solution.py`
- `README.md`

## Expected Result

The chapter script prints customer records, tensor shapes, an adjacency matrix, and
degree counts for a six-node toy social graph.
