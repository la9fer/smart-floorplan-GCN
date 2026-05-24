"""
gcn_model.py
------------
Graph Convolutional Network for per-node congestion prediction.

Architecture:
    Input (6 features) → GCNConv → ReLU → Dropout
                       → GCNConv → ReLU → Dropout
                       → GCNConv → ReLU
                       → Linear  → output (2 classes: congested / not)

Why GCN for this problem?
    In a chip floorplan, a macro's congestion is NOT independent —
    it depends on its neighbours' placement and connectivity.
    A standard MLP on individual nodes ignores this relational structure.
    GCN naturally propagates information along net edges, so each macro
    "sees" its neighbours' features before making a prediction.

    Analogy: Think of it like PageRank — a node's importance (congestion)
    is influenced by who it's connected to.

Reference: Kipf & Welling, "Semi-Supervised Classification with
           Graph Convolutional Networks" (ICLR 2017)

Author: [Your Name]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class FloorplanGCN(nn.Module):
    """
    3-layer GCN for node-level binary classification.

    Args:
        in_channels  : number of input node features (default 6)
        hidden_dim   : width of hidden layers (default 64)
        out_channels : number of classes (default 2)
        dropout      : dropout probability during training (default 0.3)

    Example:
        >>> model = FloorplanGCN(in_channels=6, hidden_dim=64)
        >>> out = model(data.x, data.edge_index)
        >>> out.shape   # (n_nodes, 2)  — logits over [not congested, congested]
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_dim: int = 64,
        out_channels: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # Layer 1: aggregate 1-hop neighbourhood
        self.conv1 = GCNConv(in_channels, hidden_dim)

        # Layer 2: aggregate 2-hop neighbourhood (captures macro clusters)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Layer 3: refine
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)

        # Final classifier head
        self.classifier = nn.Linear(hidden_dim // 2, out_channels)

        # Batch norm to stabilise training
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x          : node feature matrix, shape (N, in_channels)
            edge_index : COO edge list, shape (2, E)

        Returns:
            logits of shape (N, 2)
        """
        # ── Layer 1 ──────────────────────────────────────────────────────
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ── Layer 2 ──────────────────────────────────────────────────────
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ── Layer 3 ──────────────────────────────────────────────────────
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # ── Classification head ──────────────────────────────────────────
        logits = self.classifier(x)  # (N, 2)
        return logits

    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Returns class probabilities via softmax.

        Example:
            >>> probs = model.predict_proba(data.x, data.edge_index)
            >>> probs[:, 1]   # congestion probability per macro
        """
        logits = self.forward(x, edge_index)
        return F.softmax(logits, dim=-1)


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters.

    Example:
        >>> count_parameters(FloorplanGCN())
        12354  (approx, depending on hidden_dim)
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
