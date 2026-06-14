# 05 - Node2Vec

## Goal

Compare uniform DeepWalk-style walks with Node2Vec's biased walks, where `p` controls
return behavior and `q` controls local-vs-outward exploration.

## Product Relevance

Node2Vec helps tune what "similar" should mean for a product: close community members,
structurally similar accounts, or exploratory candidates. The boundary: this chapter
changes sampling behavior on a toy graph; it does not tune an offline recommender.

## Concept

Node2Vec remembers the previous node while walking. Returning to the previous node,
staying near the same neighborhood, or exploring outward can receive different
probabilities.

- Low `p` encourages returning.
- High `q` favors local, BFS-like exploration.
- Low `q` favors outward, DFS-like exploration.

## Run

```bash
python chapters/05-node2vec/chapter.py
python chapters/05-node2vec/exercise.py
python chapters/05-node2vec/solution.py
```

## Trace

Inspect the first walk and nearest neighbors for local (`q=2.0`) and exploratory
(`q=0.5`) variants.

## Break It

Set every bias to 1.0. The sampler collapses back toward DeepWalk-style uniform walks.

## Fix It

Complete `run_node2vec_variant` in `exercise.py` so the provided `q` value actually
changes the sampler.

## Compare

Full Node2Vec implementations precompute transition tables and train embeddings with
negative sampling. This lab keeps the bias formula visible.

## Extend

Try `p=0.25` and inspect whether walks bounce back more often.

## Check Your Understanding

- What does `q` change?
- Why does the sampler need the previous node?
- Which product would prefer local community similarity?
- Which product might prefer exploratory structural similarity?

## Files

- `chapter.py`
- `exercise.py`
- `solution.py`
- `README.md`

## Expected Result

The chapter script prints two walk variants and their nearest-neighbor summaries.
