"""Color Transformer Classifier: end-to-end model for ImageNet classification."""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
from pathlib import Path

from ikunnet.color_separator import ColorSeparator
from ikunnet.models.color_token_encoder import ColorTokenEncoder
from ikunnet.models.query_pooling import QueryPooling
from ikunnet.models.vicreg_loss import VICRegLoss


class ColorTransformerClassifier(nn.Module):
    """
    End-to-end color-based transformer classifier.

    Architecture:
        1. ColorSeparator: Segment image into color masks
        2. ColorTokenEncoder: Encode each mask with shape + color position
        3. QueryPooling: Aggregate tokens into single embedding
        4. Classifier: Predict class

    Uses dual loss:
        - VICReg loss on encoder outputs (shape representation learning)
        - Cross-entropy loss on classifier (task learning)

    Args:
        num_classes: Number of output classes (default 1000 for ImageNet)
        encoder_dim: Encoder feature dimension (default 256)
        num_heads: Number of attention heads in pooling (default 8)
        use_sin_encoding: Use sinusoidal RGB encoding (default True)
        interval_size: Color interval size for ColorSeparator (default 40)
        min_pixel_count: Min pixels for color groups (default 100)

    Example:
        >>> model = ColorTransformerClassifier(num_classes=1000)
        >>> image = torch.rand(1, 3, 640, 640)
        >>> logits, loss_dict = model(image, labels=None)
        >>> logits, loss_dict = model(image, labels=torch.tensor([42]))
    """

    def __init__(
        self,
        num_classes: int = 1000,
        encoder_dim: int = 256,
        num_heads: int = 8,
        use_sin_encoding: bool = True,
        interval_size: int = 40,
        min_pixel_count: int = 100,
        device: str = 'cuda'
    ):
        super().__init__()
        self.num_classes = num_classes
        self.encoder_dim = encoder_dim
        self.device_name = device

        # Color separator (for online extraction)
        self.color_sep = ColorSeparator(
            interval_size=interval_size,
            min_pixel_count=min_pixel_count
        )

        # Token encoder (mask + color → token)
        self.token_encoder = ColorTokenEncoder(
            encoder_dim=encoder_dim,
            pos_dim=encoder_dim,
            use_sin_encoding=use_sin_encoding
        )

        # Query pooling (tokens → single embedding)
        self.query_pool = QueryPooling(
            embed_dim=encoder_dim,
            num_heads=num_heads
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Dropout(0.3),
            nn.Linear(encoder_dim, encoder_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(encoder_dim, num_classes)
        )

        # VICReg loss (for encoder training)
        self.vicreg_loss = VICRegLoss(
            variance_weight=1.0,
            covariance_weight=1.0,
            std_target=1.0
        )

    def extract_masks(
        self,
        images: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Extract color masks from images.

        Args:
            images: (B, C, H, W) batch of images in range [0, 1]

        Returns:
            masks_list: List of (N_i, 1, H, W) masks for each image
            colors_list: List of (N_i, 3) RGB colors for each image (normalized [0, 1])
            centers_list: List of (N_i, 3) RGB center colors (raw values [0, 255])
        """
        import numpy as np

        masks_list = []
        colors_list = []
        centers_list = []

        for img in images:
            # Convert from (C, H, W) to (H, W, C) and move to CPU
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype('uint8')

            # Preprocess image (pad to 640x640 if needed)
            img_prep, preprocess_info = self.color_sep.preprocess_image(img_np, target_size=640)

            # Analyze color groups
            color_groups = self.color_sep.analyze_color_groups(img_prep)

            # Create binary masks
            masks = self.color_sep.create_binary_masks(img_prep, color_groups, preprocess_info)

            if len(color_groups) > 0:
                # Convert masks to tensors
                mask_tensors = []
                center_colors = []

                for group in color_groups:
                    mask = masks[group.group_id]  # (H, W)
                    mask_tensor = torch.from_numpy(mask).float()

                    # Resize to 224x224
                    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                    mask_tensor = nn.functional.interpolate(
                        mask_tensor,
                        size=(224, 224),
                        mode='nearest'
                    ).squeeze(0)  # (1, 224, 224)

                    mask_tensors.append(mask_tensor)
                    center_colors.append(list(group.center_color))

                masks_list.append(torch.stack(mask_tensors))
                centers_list.append(torch.tensor(center_colors, dtype=torch.float32))

                # Normalize colors to [0, 1] for encoding
                colors_list.append(torch.tensor(center_colors, dtype=torch.float32) / 255.0)
            else:
                # No color groups found, create dummy
                masks_list.append(torch.zeros(1, 1, 224, 224))
                centers_list.append(torch.zeros(1, 3))
                colors_list.append(torch.zeros(1, 3))

        return masks_list, colors_list, centers_list

    def forward(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional loss computation.

        Args:
            images: (B, 3, H, W) batch of images (normalized [0, 1])
            labels: Optional (B,) class labels for loss computation
            return_features: Whether to return intermediate features

        Returns:
            dict with:
                - logits: (B, num_classes) class predictions
                - loss: total loss (only if labels provided)
                - loss_vicreg: VICReg loss (only if labels provided)
                - loss_clf: classification loss (only if labels provided)
                - tokens: (B, N, D) tokens (only if return_features)
                - pooled: (B, D) pooled embedding (only if return_features)
        """
        B = images.shape[0]
        device = images.device

        # Extract masks and colors
        masks_list, colors_list, centers_list = self.extract_masks(images)

        # Move masks to the correct device
        masks_list = [m.to(device) for m in masks_list]
        colors_list = [c.to(device) for c in colors_list]

        # Encode tokens
        tokens, mask = self.token_encoder.forward_batch(masks_list, colors_list)

        # Pool tokens
        pooled = self.query_pool(tokens, mask)

        # Classify
        logits = self.classifier(pooled)

        output = {'logits': logits}

        # Compute losses if labels provided
        if labels is not None:
            # Classification loss
            loss_clf = nn.functional.cross_entropy(logits, labels)

            # VICReg loss on all encoder outputs
            # Collect all tokens across batch (ignoring padding)
            all_tokens = []
            for i in range(B):
                valid_mask = mask[i]
                if valid_mask.any():
                    all_tokens.append(tokens[i, valid_mask])
            if all_tokens:
                all_tokens = torch.cat(all_tokens, dim=0)
                loss_vicreg = self.vicreg_loss(all_tokens)
            else:
                loss_vicreg = torch.tensor(0.0, device=images.device)

            # Total loss
            loss = loss_clf + loss_vicreg

            output['loss'] = loss
            output['loss_vicreg'] = loss_vicreg
            output['loss_clf'] = loss_clf

        # Return intermediate features if requested
        if return_features:
            output['tokens'] = tokens
            output['mask'] = mask
            output['pooled'] = pooled

        return output

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Make predictions without loss computation.

        Args:
            images: (B, 3, H, W) batch of images

        Returns:
            predictions: (B,) predicted class indices
        """
        with torch.no_grad():
            output = self.forward(images)
            logits = output['logits']
            predictions = logits.argmax(dim=-1)
        return predictions

    def load_encoder_checkpoint(self, checkpoint_path: Path):
        """
        Load pre-trained encoder weights.

        Args:
            checkpoint_path: Path to encoder checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'encoder_state_dict' in checkpoint:
            self.token_encoder.mask_encoder.load_state_dict(
                checkpoint['encoder_state_dict']
            )
            print(f"Loaded encoder from {checkpoint_path}")
        else:
            print(f"Warning: No encoder_state_dict found in {checkpoint_path}")

    def save_checkpoint(self, path: Path, epoch: int, **kwargs):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            **kwargs: Additional metadata to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'num_classes': self.num_classes,
            'encoder_dim': self.encoder_dim,
            **kwargs
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
