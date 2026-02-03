"""Uniformity loss for unique mask embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UniformityLoss(nn.Module):
    """
    Uniformity Loss from Wang & Isola 2020.

    Encourages embeddings to be uniformly distributed on the unit hypersphere.
    Maximizes distance between different embeddings.

    Loss = log(mean(exp(-t * sim(z_i, z_j))))
    where sim is cosine similarity and t is temperature.

    Reference:
        Wang et al., "Understanding Contrastive Learning", ICCV 2021
        https://arxiv.org/abs/2012.01321
    """

    def __init__(self, temperature: float = 2.0):
        """
        Args:
            temperature: Temperature parameter for scaling similarity.
                         Higher values = stronger penalty for similarity.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute uniformity loss.

        Args:
            embeddings: (B, D) L2-normalized embeddings

        Returns:
            loss: Scalar loss value (lower = more uniform distribution)
        """
        # Compute pairwise cosine similarity
        # sim[i, j] = z_i · z_j (since embeddings are L2 normalized)
        similarity = torch.mm(embeddings, embeddings.t())  # (B, B)

        # Remove diagonal (self-similarity is always 1.0)
        batch_size = embeddings.size(0)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        similarity = similarity[mask]

        # Uniformity: minimize similarity (maximize distance)
        # Lower loss = more uniform distribution
        loss = torch.log(torch.exp(-self.temperature * similarity).mean() + 1e-6)

        return loss
