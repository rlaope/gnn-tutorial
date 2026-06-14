# 09 - Link Prediction

## Goal

Train a tiny dot-product link predictor and use it to rank missing user-item relations
as recommendation candidates.

## Product Relevance

Link prediction is the graph learning shape behind recommendations, follow
suggestions, missing relation recovery, and candidate generation. The boundary: this
lab ranks toy candidates; it is not an online recommender or ranking service.

## Concept

The model learns one embedding per node. Existing user-item edges are positive
examples, sampled absent user-item pairs are negative examples, and the dot product
between two embeddings becomes an edge score.

## Run

```bash
python chapters/09-link-prediction/chapter.py
python chapters/09-link-prediction/exercise.py
python chapters/09-link-prediction/solution.py
```

## Trace

Inspect the positive edge split, negative samples, final loss, ROC-AUC/AP, and top
missing user-item recommendations.

## Break It

Sample negative edges from any node pair. The model may learn that user-user pairs are
negative, which is not the product question.

## Fix It

Complete `score_edges` in `exercise.py`.

## Compare

Production recommenders combine graph candidates with ranking models, business rules,
freshness, and serving infrastructure. This lab only teaches the graph candidate step.

## Extend

Score recommendations for user 2 after hiding one of its existing edges.

## Check Your Understanding

- What is a positive edge?
- Why should negative edges stay inside the user-item product space?
- What does a dot product score?
- Why is link prediction not the same as a complete recommender?

## Files

- `chapter.py`
- `exercise.py`
- `solution.py`
- `README.md`

## Expected Result

The chapter script prints link-prediction metrics and the highest-scoring missing
user-item edges.
