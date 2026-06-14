# 04 - DeepWalk

## Goal

Turn random-walk contexts into node embeddings and use cosine similarity to find
nearby nodes.

## Product Relevance

DeepWalk-style embeddings support similar-user, similar-item, and candidate discovery
features. The boundary: this chapter factorizes a tiny co-occurrence matrix instead
of training the full skip-gram neural objective used in the original paper.

## Concept

DeepWalk treats random walks like short sentences. A node is similar to another node
when they appear in similar walk contexts. This lab builds a positive PMI context
matrix and factorizes it so the idea is visible without a long training loop.

## Run

```bash
python chapters/04-deepwalk/chapter.py
python chapters/04-deepwalk/exercise.py
python chapters/04-deepwalk/solution.py
```

## Trace

Inspect the walk corpus size, co-occurrence matrix shape, embedding shape, and nearest
neighbors by cosine similarity.

## Break It

Set the context window too small or use too few walks. The embedding has little
evidence and nearest neighbors become unstable.

## Fix It

Complete `build_context_counts` in `exercise.py` with a nonzero window.

## Compare

Production embedding jobs usually train on many walks and use negative sampling. This
chapter keeps the matrix view because it exposes the same local-context assumption.

## Extend

Increase `walks_per_node` and see whether node 0's nearest neighbors change.

## Check Your Understanding

- Why do walks become context windows?
- What does cosine similarity compare?
- Why is this still not a supervised classifier?
- Which product candidates might use graph embeddings before ranking?

## Files

- `chapter.py`
- `exercise.py`
- `solution.py`
- `README.md`

## Expected Result

The chapter script prints a two-dimensional embedding shape and nearest-neighbor
lists for two nodes.
