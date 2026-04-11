"""A4 generic world model with swappable encoder.

Mirrors HeatWorldModel interface exactly so that the existing train_model()
and collect_latents() work unchanged via duck typing.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from neusym_bridge.models.baseline_mlp import (
    HeatDecoder,
    HeatEncoder,
    LatentPredictor,
)
from experiments.a4.architectures import ENCODER_REGISTRY


# Include the original CNN encoder under the "cnn" key
ALL_ENCODERS = {
    "cnn": HeatEncoder,
    **ENCODER_REGISTRY,
}

SEEDS = [42, 137, 999]


class HeatWorldModelA4(nn.Module):
    """World model with interchangeable encoder architecture.

    Interface matches HeatWorldModel:
        forward(T_t, T_t1) -> (z_pred, z_t1, T_t_recon, T_t1_recon)
        encode(x) -> z
        decode(z) -> x_hat
    """

    def __init__(self, arch_name: str, latent_dim: int = 32):
        super().__init__()
        EncoderClass = ALL_ENCODERS[arch_name]
        self.encoder = EncoderClass(latent_dim=latent_dim)
        self.decoder = HeatDecoder(latent_dim=latent_dim)
        self.predictor = LatentPredictor(latent_dim=latent_dim)

    def forward(self, T_t: torch.Tensor, T_t1: torch.Tensor):
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


def create_a4_model(arch_name: str, seed: int, latent_dim: int = 32) -> HeatWorldModelA4:
    torch.manual_seed(seed)
    return HeatWorldModelA4(arch_name, latent_dim=latent_dim)


def save_a4_model(model: HeatWorldModelA4, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_a4_model(
    arch_name: str, path: str | Path, latent_dim: int = 32,
) -> HeatWorldModelA4:
    model = HeatWorldModelA4(arch_name, latent_dim=latent_dim)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def model_key(arch: str, seed: int) -> str:
    return f"{arch}_seed{seed}"


def ckpt_path(arch: str, seed: int) -> Path:
    return Path(f"experiments/a4/checkpoints/{arch}_seed{seed}.pt")
