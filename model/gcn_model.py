
"""
Graph Convolutional Network for node-level congestion prediction
in synthetic chip floorplans.

Architecture:
    Input (6 node features)
        -> GCNConv -> BatchNorm -> ReLU -> Dropout
        -> GCNConv -> BatchNorm -> ReLU -> Dropout
        -> GCNConv -> ReLU
        -> Linear classifier
        -> Output (2 class logits)

The model uses node features and graph connectivity (edge_index).
Edge attributes, such as Manhattan distances, are not currently
used by the GCN layers.

The three graph convolutional layers allow information to propagate
through neighbourhoods up to three hops away.

Reference:
    Kipf, T. N. and Welling, M. (2017).
    Semi-Supervised Classification with Graph Convolutional Networks.
    International Conference on Learning Representations (ICLR).

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class FloorplanGCN(nn.Module):
    """
    Three-layer GCN for node-level binary classification.

    Args:
        in_channels (int): Number of input node features.
            Default: 6.

        hidden_dim (int): Hidden layer dimensionality.
            Default: 64.

        out_channels (int): Number of output classes.
            Default: 2.

        dropout (float): Dropout probability during training.
            Default: 0.3.

    Input:
        x:
            Node feature matrix of shape (N, in_channels).

        edge_index:
            Graph connectivity in COO format,
            with shape (2, E).

    Output:
        Logits of shape (N, out_channels).

        For the default binary classification setup:
            Class 0 = Not congested
            Class 1 = Congested

    Example:
        model = FloorplanGCN(in_channels=6, hidden_dim=64)
        logits = model(data.x, data.edge_index)
        probabilities = model.predict_proba(
            data.x, data.edge_index
        )
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

        # First graph convolution:
        # Aggregate information from one-hop neighbours.
        self.conv1 = GCNConv(in_channels, hidden_dim)

        # Second graph convolution:
        # Extend information propagation to two-hop neighbours.
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Third graph convolution:
        # Extend information propagation to three-hop neighbours.
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)

        # Batch normalization for the first two hidden layers.
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Node-level classification head.
        self.classifier = nn.Linear(
            hidden_dim // 2,
            out_channels
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute node-level class logits.

        Args:
            x: Node features, shape (N, in_channels).
            edge_index: COO graph connectivity, shape (2, E).

        Returns:
            Tensor of shape (N, out_channels).

        Note:
            Only graph connectivity is used. Edge attributes
            are not passed to the convolutional layers.
        """

        # Graph convolution 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        # Graph convolution 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        # Graph convolution 3
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Per-node classification
        logits = self.classifier(x)

        return logits

    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert model logits into class probabilities.

        Returns:
            Tensor of shape (N, 2), containing probabilities
            for [not congested, congested].

        Note:
            Call model.eval() before using this method for inference.
            Use torch.no_grad() when gradients are not required.
        """

        logits = self.forward(x, edge_index)

        return F.softmax(logits, dim=-1)


def count_parameters(model: nn.Module) -> int:
    """
    Return the number of trainable model parameters.

    For the default FloorplanGCN configuration, this is 7,010.
    """
    
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
