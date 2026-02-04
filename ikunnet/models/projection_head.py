"""Projection Head for contrastive learning."""

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    Projection Head for SimCLR-style contrastive learning.

    Maps encoder features to embedding space where contrastive loss is applied.

    Architecture: 2-layer MLP
    feature_dim -> hidden_dim -> embedding_dim
    """

    def __init__(self, feature_dim: int = 256, hidden_dim: int = 128, embedding_dim: int = 128):
        """
        Args:
            feature_dim: Input feature dimension from encoder
            hidden_dim: Hidden layer dimension
            embedding_dim: Final embedding dimension
        """
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features from encoder (B, feature_dim)

        Returns:
            embeddings: (B, embedding_dim)
        """
        # Project to embedding space
        z = self.projection(x)

        # No normalization - VICReg needs variance
        return z


class PredictionHead(nn.Module):
    """
    Prediction Head for BYOL-style contrastive learning.

    Optional: Only needed if using BYOL instead of SimCLR.
    """

    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 64):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)
