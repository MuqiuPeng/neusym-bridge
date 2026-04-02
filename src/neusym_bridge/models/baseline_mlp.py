"""CNN-based world model for next-step prediction of 2D heat equation dynamics.

Architecture: ConvEncoder → latent z → LatentPredictor → z_next
                                     → Decoder → reconstructed T
Training objective: reconstruction MSE + prediction MSE in latent space.
Reconstruction loss prevents representation collapse by forcing the latent
to retain enough information to reconstruct the full temperature field.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HeatEncoder(nn.Module):
    """Convolutional encoder: (batch, 1, 32, 32) → (batch, latent_dim)."""

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(4)  # → (batch, 32, 4, 4)
        self.fc = nn.Linear(32 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 32, 32)
        h = x.unsqueeze(1)  # → (batch, 1, 32, 32)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.pool(h)
        h = h.flatten(1)
        return self.fc(h)


class HeatDecoder(nn.Module):
    """Decoder: (batch, latent_dim) → (batch, 32, 32).

    Mirrors the encoder: FC → reshape → deconv layers.
    """

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 32 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),  # 4→16
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 16→32
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, 32, 4, 4)
        h = self.deconv(h)
        return h.squeeze(1)  # → (batch, 32, 32)


class LatentPredictor(nn.Module):
    """MLP predictor in latent space: z_t → z_{t+1}."""

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.hidden1 = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())
        self.hidden2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.output = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.hidden1(z)
        h = self.hidden2(h)
        return self.output(h)


class HeatWorldModel(nn.Module):
    """Full world model with reconstruction to prevent representation collapse.

    Loss = recon_loss(T_t) + recon_loss(T_t1) + pred_loss(z_pred, z_t1)
    """

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.encoder = HeatEncoder(latent_dim)
        self.decoder = HeatDecoder(latent_dim)
        self.predictor = LatentPredictor(latent_dim)

    def forward(self, T_t: torch.Tensor, T_t1: torch.Tensor):
        """Forward pass.

        Returns:
            z_pred: Predicted next latent.
            z_t1: Encoded next latent (target for prediction).
            T_t_recon: Reconstructed T_t.
            T_t1_recon: Reconstructed T_t1.
        """
        z_t = self.encoder(T_t)
        z_t1 = self.encoder(T_t1)
        z_pred = self.predictor(z_t)
        T_t_recon = self.decoder(z_t)
        T_t1_recon = self.decoder(z_t1)
        return z_pred, z_t1, T_t_recon, T_t1_recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


SEED_CONFIGS = {
    "model_a": 42,
    "model_b": 137,
    "model_c": 999,
}


def create_model(model_name: str, latent_dim: int = 32) -> HeatWorldModel:
    """Create a world model with deterministic initialization."""
    seed = SEED_CONFIGS[model_name]
    torch.manual_seed(seed)
    return HeatWorldModel(latent_dim=latent_dim)
