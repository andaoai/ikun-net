"""Color Token Encoder: combines mask encoder with RGB positional encoding."""

import torch
import torch.nn as nn
from typing import List, Tuple

from ikunnet.models.encoder import CNNEncoder
from ikunnet.models.rgb_positional_encoding import RGBPositionalEncoding


class ColorTokenEncoder(nn.Module):
    """
    Encodes color masks into tokens with positional information.

    For each color mask:
        1. Encode mask using CNN encoder → shape features
        2. Encode RGB color using sinusoidal encoding → position features
        3. Combine: token = shape_features + position_features

    This preserves both:
        - Shape information (from mask patterns)
        - Color/position information (from RGB values)

    Args:
        encoder_dim: Output dimension of CNN encoder (default 256)
        pos_dim: Dimension of positional encoding (default 256)
        use_sin_encoding: Use sinusoidal encoding (True) or MLP ( False)

    Example:
        >>> encoder = ColorTokenEncoder(encoder_dim=256, pos_dim=256)
        >>> masks = [torch.rand(5, 1, 224, 224)]  # 5 masks
        >>> colors = torch.rand(5, 3)  # 5 RGB colors
        >>> tokens = encoder(masks, colors)  # (5, 256)
    """

    def __init__(
        self,
        encoder_dim: int = 256,
        pos_dim: int = 256,
        use_sin_encoding: bool = True
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.pos_dim = pos_dim

        # Mask encoder (extracts shape features)
        self.mask_encoder = CNNEncoder(embedding_dim=encoder_dim)

        # Color positional encoder (encodes RGB position)
        if use_sin_encoding:
            self.color_encoder = RGBPositionalEncoding(dim=pos_dim)
        else:
            from ikunnet.models.rgb_positional_encoding import RGBMLPEncoding
            self.color_encoder = RGBMLPEncoding(dim=pos_dim)

        # Projection to match dimensions if needed
        if encoder_dim != pos_dim:
            self.proj = nn.Linear(pos_dim, encoder_dim)
        else:
            self.proj = nn.Identity()

    def forward(
        self,
        masks: torch.Tensor,
        colors: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode masks and colors into tokens.

        Args:
            masks: (N, 1, H, W) batch of binary masks
            colors: (N, 3) RGB colors for each mask, in range [0, 255]

        Returns:
            tokens: (N, encoder_dim) combined shape + position tokens
        """
        # Encode masks (shape information)
        h = self.mask_encoder(masks)  # (N, encoder_dim)

        # Encode colors (position information)
        p = self.color_encoder(colors)  # (N, pos_dim)
        p = self.proj(p)  # (N, encoder_dim)

        # Combine: shape + position
        tokens = h + p  # (N, encoder_dim)

        return tokens

    def forward_batch(
        self,
        masks_list: List[torch.Tensor],
        colors_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of images with variable numbers of masks.

        This handles the case where different images have different numbers
        of color masks (dynamic token count).

        Args:
            masks_list: List of (N_i, 1, H, W) masks for each image
            colors_list: List of (N_i, 3) colors for each image

        Returns:
            tokens: (B, max_N, encoder_dim) padded tokens
            mask: (B, max_N) boolean mask (True = valid, False = padding)
        """
        device = masks_list[0].device
        batch_size = len(masks_list)
        max_n = max(m.shape[0] for m in masks_list)

        # Initialize padded tensors
        tokens = torch.zeros(batch_size, max_n, self.encoder_dim, device=device)
        mask = torch.zeros(batch_size, max_n, dtype=torch.bool, device=device)

        # Encode each image's masks
        for i, (masks, colors) in enumerate(zip(masks_list, colors_list)):
            n = masks.shape[0]
            if n > 0:
                tokens[i, :n] = self.forward(masks, colors)
                mask[i, :n] = True

        return tokens, mask


def collate_color_tokens(
    tokens_list: List[torch.Tensor],
    pad_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for batching variable-length token sequences.

    Args:
        tokens_list: List of (N_i, D) token tensors
        pad_value: Value to use for padding

    Returns:
        tokens: (B, max_N, D) padded tokens
        mask: (B, max_N) boolean mask (True = valid, False = padding)
    """
    batch_size = len(tokens_list)
    max_n = max(t.shape[0] for t in tokens_list)
    dim = tokens_list[0].shape[1]

    device = tokens_list[0].device

    tokens = torch.full((batch_size, max_n, dim), pad_value, device=device)
    mask = torch.zeros(batch_size, max_n, dtype=torch.bool, device=device)

    for i, t in enumerate(tokens_list):
        n = t.shape[0]
        tokens[i, :n] = t
        mask[i, :n] = True

    return tokens, mask
