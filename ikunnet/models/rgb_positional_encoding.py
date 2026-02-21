"""RGB Sinusoidal Positional Encoding for color tokens."""

import torch
import torch.nn as nn


class RGBPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for RGB color space.

    Encodes RGB values (normalized to [0, 1]) into a fixed-dimensional vector
    using sinusoidal functions at different frequencies.

    This treats RGB as a 3D coordinate space and applies encoding similar to
    Transformer positional encoding and NeRF-style positional encoding.

    Architecture:
        For each RGB channel:
        - Apply sin/cos at multiple frequencies
        - Concatenate all encodings

    Args:
        dim: Output embedding dimension (must be divisible by 6)
             6 = 3 channels × 2 (sin + cos)
        max_freq: Maximum frequency for encoding (default: 10000)

    Example:
        >>> color_enc = RGBPositionalEncoding(dim=256)
        >>> colors = torch.rand(10, 3)  # 10 colors, RGB [0, 1]
        >>> encoded = color_enc(colors)  # (10, 256)
    """

    def __init__(self, dim: int = 256, max_freq: float = 10000):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq

        # Each channel (R, G, B) uses freqs_per_channel frequencies for sin + cos
        # Total encoding dim = 3 channels × (freqs_per_channel × 2) = 6 × freqs_per_channel
        # Find closest multiple of 6 that is >= dim
        self.encoding_dim = ((dim + 5) // 6) * 6  # Round up to nearest multiple of 6
        self.freqs_per_channel = self.encoding_dim // 6

        # Add projection layer if needed to get exact output dimension
        if self.encoding_dim != dim:
            self.proj = nn.Linear(self.encoding_dim, dim)
        else:
            self.proj = nn.Identity()

    def forward(self, colors: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB colors to sinusoidal positional embeddings.

        Args:
            colors: (N, 3) tensor with RGB values in range [0, 1]
                    or (N, 3) with RGB values in range [0, 255] (will be normalized)

        Returns:
            encoded: (N, dim) sinusoidal positional encoding
        """
        # Normalize to [0, 1] if needed
        if colors.max() > 1.0:
            colors = colors / 255.0

        N = colors.shape[0]
        device = colors.device

        # Frequency bands: [1, max_freq^(1/(dim/6)), ..., max_freq]
        freqs = torch.arange(self.freqs_per_channel, device=device)
        freqs = self.max_freq ** (freqs / self.freqs_per_channel)  # (dim/6,)

        # Encode each channel separately
        encodings = []
        for channel_idx in range(3):
            # Get single channel: (N, 1)
            x = colors[:, channel_idx:channel_idx+1]

            # Compute angles: (N, dim/6)
            angles = x * freqs.unsqueeze(0)

            # Sinusoidal encoding: (N, 2 * dim/6)
            enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
            encodings.append(enc)

        # Concatenate all channels: (N, encoding_dim)
        encoded = torch.cat(encodings, dim=-1)

        # Project to target dimension if needed
        encoded = self.proj(encoded)

        return encoded

    def extra_repr(self) -> str:
        return f'dim={self.dim}, max_freq={self.max_freq}'


class RGBMLPEncoding(nn.Module):
    """
    Alternative: Simple MLP-based RGB encoding (learnable).

    This is a simpler alternative to sinusoidal encoding that learns
    the mapping from RGB to embedding space.

    Args:
        dim: Output embedding dimension
        hidden_dim: Hidden layer dimension

    Example:
        >>> color_enc = RGBMLPEncoding(dim=256, hidden_dim=128)
        >>> colors = torch.rand(10, 3)
        >>> encoded = color_enc(colors)  # (10, 256)
    """

    def __init__(self, dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, colors: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB colors using MLP.

        Args:
            colors: (N, 3) tensor with RGB values in range [0, 255]

        Returns:
            encoded: (N, dim) learned encoding
        """
        # Normalize to [0, 1]
        if colors.max() > 1.0:
            colors = colors / 255.0

        return self.mlp(colors)
