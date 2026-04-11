"""A4 encoder architectures for cross-architecture replication.

Four encoder families with fundamentally different inductive biases:
  - CNN:      local spatial filters (Phase 1 baseline, imported from baseline_mlp)
  - MLP:      no spatial prior, every pixel treated equally
  - ViT:      global attention over patch tokens
  - CNN-Wide: same CNN family but 2x capacity (control for width vs architecture)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EncoderMLP(nn.Module):
    """Pure MLP encoder: flatten 32x32 then 3-layer MLP.

    No convolutional inductive bias — each pixel is an independent input feature.
    ~55K parameters.
    """

    def __init__(self, latent_dim: int = 32, input_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 32, 32)
        return self.net(x.flatten(1))


class EncoderViT(nn.Module):
    """Minimal Vision Transformer encoder.

    Splits 32x32 into 4x4 grid of 8x8 patches (16 tokens), uses learnable
    positional embeddings + CLS token, and 2-layer Transformer encoder.
    ~60K parameters.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        patch_size: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.patch_size = patch_size
        n_patches = (32 // patch_size) ** 2  # 16
        patch_dim = patch_size * patch_size   # 64

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, n_patches + 1, d_model) * 0.02
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=128,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.head = nn.Linear(d_model, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 32, 32)
        B = x.shape[0]
        ps = self.patch_size

        # Patchify: (B, 32, 32) -> (B, n_patches, patch_dim)
        x = x.unfold(1, ps, ps).unfold(2, ps, ps)  # (B, 4, 4, 8, 8)
        x = x.contiguous().view(B, -1, ps * ps)     # (B, 16, 64)

        x = self.patch_embed(x)  # (B, 16, d_model)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 17, d_model)
        x = x + self.pos_embed

        x = self.transformer(x)
        return self.head(x[:, 0, :])  # CLS token -> latent


class EncoderCNNWide(nn.Module):
    """Wide CNN: same topology as Phase 1 CNN but 2x filter count.

    Controls for model capacity within the same architecture family.
    ~150K parameters.
    """

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(64 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 32, 32)
        h = x.unsqueeze(1)  # (batch, 1, 32, 32)
        h = self.conv(h)
        return self.fc(h.flatten(1))


ENCODER_REGISTRY = {
    "mlp": EncoderMLP,
    "vit": EncoderViT,
    "cnn_wide": EncoderCNNWide,
}
