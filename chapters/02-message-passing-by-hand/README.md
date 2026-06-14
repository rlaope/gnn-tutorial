# 02 - Message Passing By Hand

## Goal

Compute one round of neighbor aggregation by hand and see why self-loops and
normalization change the representation each node receives.

## Product Relevance

This chapter supports relational signal flow: suspicious accounts can inherit signal
from shared devices, and recommendable items can inherit signal from users and nearby
items. The boundary: this lab performs one deterministic aggregation step, not a
trained neural network.

## Concept

Message passing has three small ingredients:

1. send a message from source nodes to target nodes,
2. aggregate incoming messages at each target,
3. update each target representation.

Self-loops let a node keep its own features. Normalization prevents high-degree nodes
from dominating every update.

## Run

```bash
python chapters/02-message-passing-by-hand/chapter.py
python chapters/02-message-passing-by-hand/exercise.py
python chapters/02-message-passing-by-hand/solution.py
```

## Trace

Compare node 0 before aggregation, after neighbor-only aggregation, after adding
self-loops, and after symmetric normalization.

## Break It

Remove self-loops. Node 0 then updates only from neighbors and loses its own feature
identity in that step.

## Fix It

Complete `aggregate_with_self_loops` in `exercise.py`.

## Compare

GCN later performs this same idea with learnable weights after normalized aggregation.
This chapter freezes the weights so the aggregation itself is visible.

## Extend

Change one edge in the toy social graph and inspect which node representations move.

## Check Your Understanding

- What does a self-loop preserve?
- Why do we divide by degree in mean aggregation?
- What shape stays the same after one aggregation step?
- What product signal could move across an account-device edge?

## Files

- `chapter.py`
- `exercise.py`
- `solution.py`
- `README.md`

## Expected Result

The chapter script prints the node 0 feature vector under three aggregation variants
and the full updated feature matrix.
