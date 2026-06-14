# Product Builder Guide

This tutorial is for developers who want to turn connected data into small,
inspectable product prototypes. It does not make you a production graph ML platform
engineer by itself. It does give you the working vocabulary and runnable mechanics
needed to collaborate on graph-powered recommendation, risk, search, and graph-level
prediction features.

## What You Become Able To Build

| Goal slice | Builder role | Chapters | Portfolio artifact | Product capability |
|---|---|---|---|---|
| `G003` | Graph problem modeler | `00-why-graph-learning`, `01-graph-basics`, `02-message-passing-by-hand` | A graph task brief plus tensor trace for nodes, edges, features, and one message pass. | Translate connected product records into inspectable graph-learning inputs. |
| `G004` | Graph embedding candidate generator | `03-random-walk`, `04-deepwalk`, `05-node2vec` | A walk corpus, embedding table, and nearest-neighbor comparison. | Prototype similar-user/item exploration before a ranking system exists. |
| `G005` | Message-passing GNN prototyper | `06-gcn`, `07-graphsage`, `08-gat` | A node classifier, new-node inference path, and attention inspection table. | Build toy suspicious-node, cold-start, and neighbor-influence workflows. |
| `G006` | Graph product task builder | `09-link-prediction`, `10-graph-classification` | A recommendation candidate table plus a graph-level classifier report. | Frame missing-relation and whole-graph prediction as product-facing prototypes. |

## How To Use Each Goal Slice

Use `G003` when the problem is still vague. Your output should be a one-page brief:
node types, edge types, feature columns, prediction target, and one tensor trace that
proves the graph can be represented without hidden magic.

Use `G004` when the product needs candidates before it needs final decisions. Your
output should compare at least two neighborhood assumptions: local community
similarity and outward structural similarity.

Use `G005` when labels and local neighborhoods matter. Your output should show which
nodes were labeled for training, how a new node can be scored, and which neighbors
received higher attention in one inspection case.

Use `G006` when the product target is a missing relation or a whole connected object.
Your output should separate graph candidate generation from ranking/serving claims,
then explain what extra system work would be needed before production.

## Portfolio Project Prompts

- Build a "people you may know" candidate table from a small user graph.
- Build a suspicious-account toy classifier from accounts, devices, and merchants.
- Build a cold-start item classifier where the item appears after training.
- Build a neighbor-attention inspection table for one model decision.
- Build a graph classifier for tiny workflow, molecule-like, or dependency graphs.

Each prompt should include the same evidence: graph schema, tensor shapes, train/test
split, metric or inspection output, and one paragraph naming what the prototype does
not prove yet.

## Production Boundary

After this tutorial, you should be able to prototype and critique graph-learning
features. Production systems still need data contracts, leakage checks, temporal
splits, monitoring, fairness/security review, scalable sampling, model serving,
ranking integration, and domain validation.
