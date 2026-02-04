"""CNN Encoder for mask embedding."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block with Conv, BN, ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CNNEncoder(nn.Module):
    """
    CNN Encoder for binary mask embedding.

    Architecture:
        Input: (B, 1, 640, 640)
        ConvBlock x4 with pooling
        Global Average Pooling
        Output: (B, 256)
    """

    def __init__(self, embedding_dim: int = 128):
        """
        Args:
            embedding_dim: Final embedding dimension (default 128)
        """
        super().__init__()

        # Conv Block 1: 640x640 -> 160x160
        self.conv1 = ConvBlock(1, 32, kernel_size=7, stride=2)
        self.pool1 = nn.MaxPool2d(2, 2)  # 160x160 -> 80x80

        # Conv Block 2: 80x80 -> 20x20
        self.conv2 = ConvBlock(32, 64, kernel_size=5, stride=2)
        self.pool2 = nn.MaxPool2d(2, 2)  # 20x20 -> 10x10

        # Conv Block 3: 10x10 -> 3x3
        self.conv3 = ConvBlock(64, 128, kernel_size=3, stride=2)
        self.pool3 = nn.MaxPool2d(2, 2)  # 3x3 -> 不需要，已经很小了

        # Conv Block 4
        self.conv4 = ConvBlock(128, 256, kernel_size=3, stride=1)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Final feature dimension
        self.feature_dim = 256
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, 1, H, W) or (B, H, W)

        Returns:
            features: (B, 256) - L2 normalized features
        """
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Conv Blocks
        x = self.conv1(x)  # (B, 32, 320, 320)
        x = self.pool1(x)   # (B, 32, 160, 160)

        x = self.conv2(x)  # (B, 64, 80, 80)
        x = self.pool2(x)   # (B, 64, 40, 40)

        x = self.conv3(x)  # (B, 128, 20, 20)
        x = self.pool3(x)  # (B, 128, 10, 10)

        x = self.conv4(x)  # (B, 256, 10, 10)

        # Global Average Pooling
        x = self.gap(x)  # (B, 256, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (B, 256)

        # No normalization - VICReg needs variance
        return x


# ResNet-style Encoder (alternative)
class ResNetEncoder(nn.Module):
    """
    ResNet-style encoder for better performance.
    Simplified version without pretraining.
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()

        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # Res Blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Global pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.feature_dim = 512
        self.embedding_dim = embedding_dim

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gap(x)
        x = x.squeeze(-1).squeeze(-1)

        return F.normalize(x, dim=1)


class ResBlock(nn.Module):
    """Residual block."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
