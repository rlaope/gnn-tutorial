# 00 - Why Graph Learning

## Goal

Learn when a product question is really a graph learning question, and map that
question to the graph object being predicted: a node, an edge, or a whole graph.

## Product Relevance

This chapter supports product scoping. A graph model is useful when the thing you
want to predict depends on relationships, not only on independent rows. The boundary:
this lab chooses the graph task; it does not train a production model.

## Concept

Most graph learning projects start with ordinary product records: users follow users,
accounts share devices, customers click items, repositories depend on packages. The
graph appears when those records are viewed as connected objects.

- Node prediction asks, "What label or score belongs to this entity?"
- Edge prediction asks, "Which relationship is missing or likely to appear?"
- Graph prediction asks, "What label belongs to this entire connected structure?"

## Run

```bash
python chapters/00-why-graph-learning/chapter.py
python chapters/00-why-graph-learning/exercise.py
python chapters/00-why-graph-learning/solution.py
```

## Trace

Inspect the task table printed by `chapter.py`. The important trace is the shift from
product wording to graph wording: entity, relationship, prediction target, and later
chapter.

## Break It

Pretend every product record is independent. Recommendation becomes a row-ranking
problem without candidate relationships, and suspicious-account scoring loses shared
device or shared-merchant evidence.

## Fix It

Complete `choose_graph_task` in `exercise.py` so each product question maps to the
right graph task.

## Compare

Production systems often combine graph features with non-graph models. This tutorial
starts with the graph side so you can tell when that extra machinery is warranted.

## Extend

Pick one product you know and write the node type, edge type, and one prediction
target before choosing a model.

## Check Your Understanding

- What object is predicted in node classification?
- Why is recommendation often framed as edge prediction?
- When would graph classification be more natural than node classification?
- What product claim would be too strong after this chapter?

## Files

- `chapter.py`
- `exercise.py`
- `solution.py`
- `README.md`

## Expected Result

The chapter script prints a compact task catalog and tiny graph snapshots for the
datasets used later in the tutorial.
