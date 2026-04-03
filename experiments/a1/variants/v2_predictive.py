"""Variant 2: One-step Predictive (no reconstruction).

Loss = MSE(z_pred, z_t1) + 0.1 * VICReg(z_t)

No decoder. Encoder is free to organize latent for dynamics prediction.
VICReg regularization prevents collapse without tying latent to state space.

Hypothesis: without reconstruction, latent dynamics become more regular
and SINDy-recoverable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase4.models.lewm_tentacle import TentacleEncoder, TentaclePredictor
from phase4.envs.tentacle_env import STATE_DIM, ACTION_DIM


class PredictiveModel(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, latent_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.encoder = TentacleEncoder(state_dim, latent_dim)
        self.predictor = TentaclePredictor(latent_dim, action_dim)

    def forward(self, s_t, a_t, s_t1):
        z_t = self.encoder(s_t)
        z_t1 = self.encoder(s_t1)
        z_pred = self.predictor(z_t, a_t)
        return z_pred, z_t1, z_t

    def vicreg_loss(self, z, gamma: float = 1.0, mu: float = 1.0, nu: float = 0.04):
        """VICReg regularization: variance + covariance terms.

        Prevents latent collapse without reconstruction.
        - Variance: each dimension should have std > gamma
        - Covariance: dimensions should be decorrelated
        """
        N, D = z.shape

        # Variance: hinge loss on per-dimension std
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        loss_var = F.relu(gamma - std).mean()

        # Covariance: off-diagonal elements of covariance matrix -> 0
        z_norm = z - z.mean(dim=0)
        cov = (z_norm.T @ z_norm) / (N - 1)
        loss_cov = (cov.pow(2).sum() - cov.diag().pow(2).sum()) / D

        return mu * loss_var + nu * loss_cov

    def loss(self, s_t, a_t, s_t1, **kwargs):
        z_pred, z_t1, z_t = self.forward(s_t, a_t, s_t1)
        loss_pred = F.mse_loss(z_pred, z_t1)
        loss_vicreg = self.vicreg_loss(z_t)
        total = loss_pred + 0.1 * loss_vicreg
        return total, {"pred": loss_pred.item(), "vicreg": loss_vicreg.item()}

    def encode(self, s):
        return self.encoder(s)

    def predict(self, z, a):
        return self.predictor(z, a)
