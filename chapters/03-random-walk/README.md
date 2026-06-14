# 03 - Random Walk

## Goal

Use random walks to sample a node's local neighborhood and turn graph structure into
sequences that later embedding methods can learn from.

## Product Relevance

Random walks support exploration features such as "nearby users", "related items",
and candidate generation. The boundary: a walk is a sampling strategy, not a trained
recommendation model.

## Concept

A random walk starts at one node and repeatedly chooses a neighbor. The resulting
sequence is a local view of the graph. Nodes that appear in similar walk contexts are
often structurally or socially related.

## Run

```bash
python chapters/03-random-walk/chapter.py
python chapters/03-random-walk/exercise.py
python chapters/03-random-walk/solution.py
```

## Trace

Inspect the walk from node 0 and the co-occurrence counts produced by a small sliding
window. Those counts become the bridge to DeepWalk.

## Break It

Make the walk length 1. The walk never observes a neighbor, so no useful context is
created.

## Fix It

Complete `collect_training_walks` in `exercise.py` with one walk from every node.

## Compare

Graph databases and feature stores often have neighborhood sampling primitives. This
chapter keeps the sampler small enough to read.

## Extend

Start walks from only high-activity nodes and compare the context counts.

## Check Your Understanding

- What does one walk record?
- Why can walks turn graphs into sequence data?
- What happens when a node has no neighbors?
- Which product workflows need neighborhood exploration before ranking?

## Files

- `chapter.py`
- `exercise.py`
- `solution.py`
- `README.md`

## Expected Result

The chapter script prints adjacency lists, one deterministic walk, and a small
co-occurrence matrix.
