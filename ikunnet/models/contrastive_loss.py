"""Contrastive loss functions for self-supervised learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).

    Used in SimCLR for contrastive learning.
    """

    def __init__(self, temperature: float = 0.5):
        """
        Args:
            temperature: Temperature parameter for scaling similarity
        """
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss.

        Args:
            z_i: Embeddings from view 1 (B, embedding_dim), L2 normalized
            z_j: Embeddings from view 2 (B, embedding_dim), L2 normalized

        Returns:
            loss: Scalar loss value
        """
        batch_size = z_i.shape[0]

        # Concatenate positive pairs
        z = torch.cat((z_i, z_j), dim=0)  # (2B, embedding_dim)

        # Compute similarity matrix
        # sim[i, j] = z[i] · z[j] / temperature
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        # Create positive pairs mask
        # Diagonal offset by batch_size: (i, i+B) and (i+B, i) are positive pairs
        mask = torch.eye(batch_size, dtype=torch.bool, device=z.device)
        mask = torch.cat((mask, mask), dim=1)  # (B, 2B)
        mask = torch.cat((mask, mask), dim=0)  # (2B, 2B)

        # Remove diagonal (self-similarity)
        logits = sim[~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)].view(2 * batch_size, -1)

        # Labels: positive pair is at index (i % batch_size)
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat((labels, labels), dim=0)

        # Compute loss
        loss = self.criterion(logits, labels)

        # Normalize by batch size
        loss = loss / (2 * batch_size)

        return loss


def contrastive_loss_simclr(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Functional version of NT-Xent loss.

    Args:
        z_i: Embeddings from view 1 (B, embedding_dim)
        z_j: Embeddings from view 2 (B, embedding_dim)
        temperature: Temperature parameter

    Returns:
        loss: Scalar loss value
    """
    batch_size = z_i.shape[0]

    # Concatenate
    z = torch.cat((z_i, z_j), dim=0)

    # Similarity matrix
    sim = torch.mm(z, z.t()) / self.temperature

    # Positive pairs mask
    positives = torch.diag(sim, batch_size)
    positives = torch.cat((positives, positives), dim=0)

    # Negatives (excluding diagonal and positives)
    mask = torch.ones((2 * batch_size, 2 * batch_size), device=z.device)
    mask = mask.fill_diagonal_(0)
    mask[range(batch_size), range(batch_size, 2 * batch_size)] = 0
    mask[range(batch_size, 2 * batch_size), range(batch_size)] = 0

    negatives = sim[mask].view(2 * batch_size, -1)

    # Log-sum-exp for stability
    logits = torch.cat((positives.unsqueeze(1), negatives), dim=1)
    loss = -torch.log(torch.exp(logits[:, 0]) / torch.sum(torch.exp(logits), dim=1))

    return loss.mean()


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss - another common contrastive loss variant.

    Similar to NT-Xent but with slightly different formulation.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: Embeddings from view 1 (B, embedding_dim)
            z_j: Embeddings from view 2 (B, embedding_dim)

        Returns:
            loss: Scalar loss value
        """
        batch_size = z_i.shape[0]

        # Compute similarities
        # Positive pairs: (z_i[i], z_j[i]) for each i
        pos_sim = torch.sum(z_i * z_j, dim=1) / self.temperature  # (B,)

        # Negative pairs: all other combinations
        z = torch.cat((z_i, z_j), dim=0)  # (2B, embedding_dim)

        # Similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        # For each sample, positive is at (i, i+B) or (i+B, i)
        # Negative is everything else (excluding diagonal)
        neg_sim = []

        for i in range(batch_size):
            # View 1 positives
            pos_idx = i + batch_size
            neg_indices = list(range(2 * batch_size))
            neg_indices.remove(i)
            neg_indices.remove(pos_idx)
            neg_sim.append(sim[i, neg_indices])

            # View 2 positives
            pos_idx = i
            neg_indices = list(range(2 * batch_size))
            neg_indices.remove(i + batch_size)
            neg_indices.remove(pos_idx)
            neg_sim.append(sim[i + batch_size, neg_indices])

        neg_sim = torch.cat(neg_sim)  # (2B * (2B-2),)

        # Concatenate positive and negative similarities
        logits = torch.cat([pos_sim, pos_sim, neg_sim], dim=0)
        logits = logits.reshape(2, batch_size, -1)  # (2, B, neg+1)

        # Labels: positive is at index 0 for each sample
        labels = torch.zeros(batch_size, dtype=torch.long, device=z_i.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits[0], labels)
        loss += F.cross_entropy(logits[1], labels)

        return loss / 2
