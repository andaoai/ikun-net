"""Query Pooling: learnable query with cross-attention for token aggregation."""

import torch
import torch.nn as nn
import math


class QueryPooling(nn.Module):
    """
    Learnable query pooling with multi-head cross-attention.

    Uses a learned query vector to attend over a variable-length sequence
    of tokens and produce a single aggregated embedding.

    This is inspired by:
        - Perceiver IO: Cross-attention for pooling
        - Set Transformer: Attention-based set pooling
        - BERT pooler: [CLS] token equivalent

    The key advantage: no special tokens needed, the learnable query
    automatically learns to extract relevant information from the input.

    Args:
        embed_dim: Dimension of query and tokens (default 256)
        num_heads: Number of attention heads (default 8)
        dropout: Dropout probability (default 0.1)

    Example:
        >>> pooler = QueryPooling(embed_dim=256, num_heads=8)
        >>> tokens = torch.rand(4, 10, 256)  # batch=4, seq_len=10
        >>> mask = torch.ones(4, 10, dtype=torch.bool)  # all valid
        >>> pooled = pooler(tokens, mask)  # (4, 256)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Learnable query vector
        self.query = nn.Parameter(torch.Tensor(embed_dim))
        nn.init.normal_(self.query, mean=0.0, std=0.02)

        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer norm and MLP for processing output
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate tokens using learnable query.

        Args:
            tokens: (B, N, D) token embeddings
            mask: (B, N) boolean mask, True = valid token, False = padding

        Returns:
            pooled: (B, D) aggregated embedding
        """
        B, N, D = tokens.shape

        # Prepare query: (1, 1, D) → (B, 1, D)
        q = self.query.unsqueeze(0).unsqueeze(0).expand(B, -1, -1)

        # Multi-head attention
        # key_padding_mask: True = ignore position (opposite of our mask)
        key_padding_mask = ~mask

        attn_output, attn_weights = self.cross_attn(
            query=q,
            key=tokens,
            value=tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )  # attn_output: (B, 1, D)

        # Squeeze and normalize
        pooled = attn_output.squeeze(1)  # (B, D)
        pooled = self.norm1(pooled)

        # MLP processing
        residual = pooled
        pooled = self.mlp(pooled)
        pooled = pooled + residual
        pooled = self.norm2(pooled)

        return pooled


class MultiQueryPooling(nn.Module):
    """
    Multiple learnable queries for multi-scale aggregation.

    Uses multiple query vectors to extract different aspects of the input.
    The outputs are concatenated or averaged.

    Args:
        embed_dim: Dimension of tokens
        num_queries: Number of query vectors (default 4)
        num_heads: Number of attention heads per query (default 8)
        aggregation: How to combine query outputs ('mean', 'max', 'concat')

    Example:
        >>> pooler = MultiQueryPooling(embed_dim=256, num_queries=4)
        >>> tokens = torch.rand(4, 10, 256)
        >>> mask = torch.ones(4, 10, dtype=torch.bool)
        >>> pooled = pooler(tokens, mask)  # (4, 256)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_queries: int = 4,
        num_heads: int = 8,
        aggregation: str = 'mean',
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.aggregation = aggregation

        # Multiple learnable queries
        self.queries = nn.Parameter(torch.Tensor(num_queries, embed_dim))
        nn.init.normal_(self.queries, mean=0.0, std=0.02)

        # Single attention that handles all queries at once
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate tokens using multiple learnable queries.

        Args:
            tokens: (B, N, D) token embeddings
            mask: (B, N) boolean mask

        Returns:
            pooled: (B, D) aggregated embedding
        """
        B, N, D = tokens.shape

        # Prepare queries: (num_queries, D) → (B, num_queries, D)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)

        # Multi-head attention
        key_padding_mask = ~mask

        attn_output, _ = self.cross_attn(
            query=q,
            key=tokens,
            value=tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )  # (B, num_queries, D)

        # Aggregate multiple query outputs
        if self.aggregation == 'mean':
            pooled = attn_output.mean(dim=1)  # (B, D)
        elif self.aggregation == 'max':
            pooled = attn_output.max(dim=1)[0]  # (B, D)
        elif self.aggregation == 'concat':
            pooled = attn_output.reshape(B, -1)  # (B, num_queries * D)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return self.norm(pooled)


class SetPool(nn.Module):
    """
    Simple set pooling without attention (baseline).

    Provides simple pooling strategies as baselines for comparison.

    Args:
        embed_dim: Dimension of tokens
        pooling_type: 'mean', 'max', or 'sum'

    Example:
        >>> pooler = SetPool(embed_dim=256, pooling_type='mean')
        >>> tokens = torch.rand(4, 10, 256)
        >>> mask = torch.ones(4, 10, dtype=torch.bool)
        >>> pooled = pooler(tokens, mask)  # (4, 256)
    """

    def __init__(self, embed_dim: int = 256, pooling_type: str = 'mean'):
        super().__init__()
        self.pooling_type = pooling_type

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool tokens using simple aggregation.

        Args:
            tokens: (B, N, D) token embeddings
            mask: (B, N) boolean mask

        Returns:
            pooled: (B, D) aggregated embedding
        """
        # Expand mask for broadcasting
        mask_expanded = mask.unsqueeze(-1)  # (B, N, 1)

        # Apply mask (set padding to 0)
        masked_tokens = tokens * mask_expanded

        # Compute valid counts
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)

        if self.pooling_type == 'mean':
            pooled = masked_tokens.sum(dim=1) / valid_counts
        elif self.pooling_type == 'max':
            # Use large negative for padding instead of 0
            masked_tokens = torch.where(
                mask_expanded,
                tokens,
                torch.tensor(-1e9, device=tokens.device)
            )
            pooled = masked_tokens.max(dim=1)[0]
        elif self.pooling_type == 'sum':
            pooled = masked_tokens.sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")

        return pooled
