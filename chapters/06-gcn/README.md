# 06 - GCN

## Goal

Train a tiny Graph Convolutional Network for suspicious-node classification and trace
how normalized neighbor aggregation feeds a classifier.

## Product Relevance

GCN supports product features such as suspicious-account scoring when risk signals
move through shared devices, merchants, or sessions. The boundary: this lab uses safe
synthetic data and is not a fraud detection system.

## Concept

A GCN layer computes normalized neighbor aggregation and then applies a learnable
linear map. Compared with the hand-written message passing chapter, the model now
learns which aggregated feature combinations help a label.

## Run

```bash
python chapters/06-gcn/chapter.py
python chapters/06-gcn/exercise.py
python chapters/06-gcn/solution.py
```

## Trace

Inspect the first and final loss, train/test accuracy, predicted classes, and the
probability assigned to the suspicious account nodes.

## Break It

Remove normalization. High-degree nodes can dominate updates, which makes training
less stable and harder to reason about.

## Fix It

Complete `apply_gcn_aggregation` in `exercise.py`.

## Compare

PyTorch Geometric has `GCNConv`; this chapter implements the same core idea with dense
normalization so the tensor path remains visible.

## Extend

Change the risk feature for node 8 and observe how the predicted probability moves.

## Check Your Understanding

- What does normalized adjacency do?
- Which labels are used during training?
- Why is this still only a toy risk classifier?
- What product evidence can move across account-device edges?

## Files

- `chapter.py`
- `exercise.py`
- `solution.py`
- `README.md`

## Expected Result

The chapter script prints compact training diagnostics and class probabilities for
the suspicious-account toy graph.
