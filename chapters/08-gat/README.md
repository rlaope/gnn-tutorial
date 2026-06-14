# 08 - GAT

## Goal

Train a tiny Graph Attention Network and inspect which incoming neighbors receive the
largest attention weights.

## Product Relevance

GAT supports neighbor influence inspection: a product team can ask which linked
entities had more weight in a score. The boundary: attention is a useful inspection
signal, not a formal explanation guarantee.

## Concept

Instead of averaging all neighbors equally, GAT learns an attention weight per incoming
edge. Each target node receives a weighted sum of transformed source-node features.

## Run

```bash
python chapters/08-gat/chapter.py
python chapters/08-gat/exercise.py
python chapters/08-gat/solution.py
```

## Trace

Inspect the incoming attention edges for node 8. The self-loop is included because the
node can attend to its own features.

## Break It

Average all neighbors equally. The model loses the ability to highlight one incoming
source over another.

## Fix It

Complete `top_incoming_attention` in `exercise.py`.

## Compare

Production explainability requires stronger methods and domain validation. This lab
only shows where an attention layer placed higher weights.

## Extend

Inspect the top incoming attention edges for node 9 and compare them with node 8.

## Check Your Understanding

- What is normalized per target node?
- Why is attention not the same as guaranteed explanation?
- Which edges compete with each other?
- What product review question can attention help start?

## Files

- `chapter.py`
- `exercise.py`
- `solution.py`
- `README.md`

## Expected Result

The chapter script prints training diagnostics and the top incoming attention edges
for a suspicious account node.
