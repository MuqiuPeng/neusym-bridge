"""Variant 1: Reconstruction AE (baseline, same as Phase 4 LeWM).

Loss = MSE(z_pred, z_t1) + MSE(s_recon, s_t)

This is the existing training objective. Reconstruction keeps latent
from collapsing but may force high-rank mixed representations that
degrade dynamical structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase4.models.lewm_tentacle import TentacleEncoder, TentaclePredictor, TentacleDecoder
from phase4.envs.tentacle_env import STATE_DIM, ACTION_DIM


class ReconstructionAE(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, latent_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.encoder = TentacleEncoder(state_dim, latent_dim)
        self.predictor = TentaclePredictor(latent_dim, action_dim)
        self.decoder = TentacleDecoder(latent_dim, state_dim)

    def forward(self, s_t, a_t, s_t1):
        z_t = self.encoder(s_t)
        z_t1 = self.encoder(s_t1)
        z_pred = self.predictor(z_t, a_t)
        s_recon = self.decoder(z_t)
        return z_pred, z_t1, s_recon

    def loss(self, s_t, a_t, s_t1, **kwargs):
        z_pred, z_t1, s_recon = self.forward(s_t, a_t, s_t1)
        loss_pred = F.mse_loss(z_pred, z_t1)
        loss_recon = F.mse_loss(s_recon, s_t)
        total = loss_pred + loss_recon
        return total, {"pred": loss_pred.item(), "recon": loss_recon.item()}

    def encode(self, s):
        return self.encoder(s)

    def predict(self, z, a):
        return self.predictor(z, a)

    def decode(self, z):
        return self.decoder(z)
