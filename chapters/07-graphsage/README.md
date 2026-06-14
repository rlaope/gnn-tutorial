# 07 - GraphSAGE

## Goal

Train a tiny GraphSAGE classifier and run inference for a new account node that was
not present during training.

## Product Relevance

GraphSAGE supports inductive product workflows: new users, items, sellers, devices, or
accounts arrive after training but still need a prediction. The boundary: this lab
uses a hand-built new account, not a streaming feature pipeline.

## Concept

GraphSAGE combines a node's own features with an aggregate of sampled neighbor
features. Because the layer is a reusable function of features and neighbors, it can
score a newly added node when its local neighborhood is known.

## Run

```bash
python chapters/07-graphsage/chapter.py
python chapters/07-graphsage/exercise.py
python chapters/07-graphsage/solution.py
```

## Trace

Inspect the new account's neighbors and its predicted suspicious probability.

## Break It

Use no neighbors for the new account. The model then sees only the account's own
features and loses shared-device context.

## Fix It

Complete `sample_neighbors` in `exercise.py`.

## Compare

Large GraphSAGE systems sample a bounded number of neighbors to keep inference cheap.
This chapter samples by sorted node ids so the mechanics stay deterministic.

## Extend

Connect the new account to a different device and compare the probability.

## Check Your Understanding

- Why is GraphSAGE called inductive?
- What is concatenated before the linear layer?
- What breaks when a new node has no edges?
- Which product entities commonly arrive after training?

## Files

- `chapter.py`
- `exercise.py`
- `solution.py`
- `README.md`

## Expected Result

The chapter script prints training diagnostics and a new-account probability.
