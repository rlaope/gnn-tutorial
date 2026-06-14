"""Lightweight visualization helpers for graphs and embeddings."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import torch

from graph_tutorial.graph_utils import edge_index_to_edge_list


def to_networkx(edge_index: torch.Tensor, *, num_nodes: int) -> nx.Graph:
    """Convert `edge_index` to an undirected NetworkX graph."""

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edge_index_to_edge_list(edge_index))
    return graph


def plot_graph(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    labels: torch.Tensor | None = None,
    path: str | Path | None = None,
) -> None:
    """Plot a small graph and optionally save it."""

    graph = to_networkx(edge_index, num_nodes=num_nodes)
    pos = nx.spring_layout(graph, seed=0)
    colors = labels.tolist() if labels is not None else "#73a9ff"
    nx.draw_networkx(graph, pos=pos, node_color=colors, cmap="coolwarm", with_labels=True)
    _finish_plot(path)


def plot_embeddings(embeddings: torch.Tensor, *, path: str | Path | None = None) -> None:
    """Plot the first two embedding dimensions."""

    emb = embeddings.detach().cpu()
    if emb.shape[1] < 2:
        raise ValueError("Need at least two embedding dimensions to plot")
    plt.scatter(emb[:, 0], emb[:, 1])
    for idx, (x_coord, y_coord) in enumerate(emb[:, :2].tolist()):
        plt.text(x_coord, y_coord, str(idx))
    _finish_plot(path)


def _finish_plot(path: str | Path | None) -> None:
    plt.tight_layout()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
    plt.close()
