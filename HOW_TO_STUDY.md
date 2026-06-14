# How to Study This Tutorial

This repository is easiest to use as a set of labs. Do not start by memorizing
model names. Start by asking what kind of connected-data product problem you are
trying to understand.

## Track A: Three-Hour Quick Tour

Use this when you want the shape of the field before going deep.

1. `00-why-graph-learning`
2. `01-graph-basics`
3. `02-message-passing-by-hand`
4. `09-link-prediction`

You should finish with a concrete sense of why graphs are useful and how missing
relations become recommendations or search candidates.

## Track B: From-Scratch Path

Use this when you want to understand the mechanics before touching higher-level
libraries.

1. `01-graph-basics`
2. `02-message-passing-by-hand`
3. `03-random-walk`
4. `04-deepwalk`
5. `05-node2vec`
6. `06-gcn`
7. `07-graphsage`
8. `08-gat`

This path emphasizes shapes, edge lists, sampling, aggregation, and the difference
between embedding methods and message-passing GNNs.

## Track C: Product Path

Use this when you care about what a graph learner can build.

1. `00-why-graph-learning`
2. `06-gcn`
3. `07-graphsage`
4. `08-gat`
5. `09-link-prediction`
6. `10-graph-classification`

This path maps graph learning to suspicious-node classification, new-node
predictions, neighbor influence, recommendation, missing-relation recovery, and
whole-graph classification.

## Track D: Research Reading Path

Use this after you can run the code.

1. DeepWalk
2. Node2Vec
3. GCN
4. GraphSAGE
5. GAT

Read each paper after finishing the matching chapter. That order keeps the paper
connected to an experiment you already ran.

## The Lab Loop

Each complete chapter follows the same learning loop:

1. **See it**: inspect a graph, plot, tensor, or prediction table.
2. **Run it**: execute a small CPU-friendly script.
3. **Trace it**: print shapes and intermediate values.
4. **Break it**: remove self-loops, change sampling, or corrupt a split.
5. **Fix it**: restore the missing idea through an exercise.
6. **Compare it**: relate the from-scratch code to practical tooling.
7. **Extend it**: answer a small product-shaped question.

If a chapter feels too abstract, go back to the trace and break/fix sections. That
is where graph learning usually becomes tangible.
