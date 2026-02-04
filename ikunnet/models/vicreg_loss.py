"""VICReg-style loss for compression representation learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegLoss(nn.Module):
    """
    Variance-Invariance-Covariance Regularization Loss.

    Simplified version: only variance and covariance regularization.
    No invariance term (no positive pairs needed).

    Reference:
        Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization", 2022
        https://inria.hal.science/hal-03541297/document
    """

    def __init__(
        self,
        variance_weight: float = 1.0,
        covariance_weight: float = 1.0,
        std_target: float = 1.0
    ):
        """
        Args:
            variance_weight: Weight for variance loss term
            covariance_weight: Weight for covariance loss term
            std_target: Target standard deviation for each dimension
        """
        super().__init__()
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.std_target = std_target

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute VICReg loss (variance + covariance).

        Args:
            embeddings: (B, D) embeddings (not necessarily L2-normalized)

        Returns:
            loss: scalar loss value
        """
        batch_size, embedding_dim = embeddings.shape

        # Variance loss: encourage each dimension to have sufficient variance
        std = embeddings.std(dim=0) + 1e-6
        variance_loss = torch.mean(F.relu(self.std_target - std))

        # Covariance loss: decorrelate different dimensions
        # Center embeddings
        embeddings_centered = embeddings - embeddings.mean(dim=0, keepdim=True)

        # Compute covariance matrix
        cov = torch.mm(embeddings_centered.t(), embeddings_centered) / (batch_size - 1)

        # Remove diagonal (variance is already handled)
        cov_off_diagonal = cov - torch.diag(torch.diag(cov))

        # Minimize off-diagonal elements (decorrelation)
        covariance_loss = cov_off_diagonal.pow(2).sum() / (embedding_dim ** 2)

        # Total loss
        loss = self.variance_weight * variance_loss + self.covariance_weight * covariance_loss

        return loss
