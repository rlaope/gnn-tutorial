"""Tiny GNN layers implemented with plain PyTorch tensors."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from graph_tutorial.graph_utils import add_self_loops, mean_aggregate, normalized_adjacency


class GCNLayer(nn.Module):
    """One graph convolution layer: normalized aggregation followed by a linear map."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        norm_adj = normalized_adjacency(edge_index, x.shape[0])
        return self.linear(norm_adj @ x)


class GraphSAGELayer(nn.Module):
    """Mean GraphSAGE layer using self features plus neighbor mean features."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        neighbor_mean = mean_aggregate(x, edge_index, x.shape[0])
        return self.linear(torch.cat([x, neighbor_mean], dim=1))


class SingleHeadGATLayer(nn.Module):
    """Single-head graph attention layer for small inspectable graphs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.attention = nn.Parameter(torch.empty(out_channels * 2))
        self.negative_slope = negative_slope
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attention.unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_nodes = x.shape[0]
        edge_index = add_self_loops(edge_index, num_nodes)
        h = self.linear(x)
        src, dst = edge_index
        pair_features = torch.cat([h[src], h[dst]], dim=1)
        logits = F.leaky_relu(
            (pair_features * self.attention).sum(dim=1),
            negative_slope=self.negative_slope,
        )
        weights = torch.zeros_like(logits)
        for node in range(num_nodes):
            mask = dst == node
            if bool(mask.any()):
                weights[mask] = torch.softmax(logits[mask], dim=0)

        out = torch.zeros_like(h)
        out.index_add_(0, dst, h[src] * weights.unsqueeze(1))
        if return_attention:
            return out, edge_index, weights
        return out
