"""Variant 3: Temporal Contrastive learning.

Loss = MSE(z_pred, z_t1) + InfoNCE(z_t, z_t1, z_neg)

Adjacent timesteps are positive pairs; random states are negatives.
Contrastive objective directly optimizes temporal structure of latent,
which should benefit dynamical recoverability.

Projection head isolates contrastive loss from encoder representation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase4.models.lewm_tentacle import TentacleEncoder, TentaclePredictor
from phase4.envs.tentacle_env import STATE_DIM, ACTION_DIM


class TemporalContrastiveModel(nn.Module):
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        latent_dim: int = 64,
        proj_dim: int = 32,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.temperature = temperature

        self.encoder = TentacleEncoder(state_dim, latent_dim)
        self.predictor = TentaclePredictor(latent_dim, action_dim)

        # Projection head: contrastive loss computed in this space
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def temporal_nce_loss(self, z_t, z_t1, z_neg):
        """InfoNCE with temporal positive pairs.

        Args:
            z_t:   (B, latent_dim) anchor
            z_t1:  (B, latent_dim) positive (next timestep)
            z_neg: (B, K, latent_dim) negatives (random states)
        """
        q = F.normalize(self.projector(z_t), dim=-1)    # (B, D)
        k = F.normalize(self.projector(z_t1), dim=-1)   # (B, D)

        # Flatten negatives through projector
        B, K, _ = z_neg.shape
        n = F.normalize(
            self.projector(z_neg.reshape(B * K, -1)),
            dim=-1,
        ).reshape(B, K, -1)  # (B, K, D)

        # Positive similarity
        pos = (q * k).sum(dim=-1, keepdim=True) / self.temperature  # (B, 1)

        # Negative similarities
        neg = torch.bmm(n, q.unsqueeze(-1)).squeeze(-1) / self.temperature  # (B, K)

        logits = torch.cat([pos, neg], dim=-1)  # (B, K+1)
        labels = torch.zeros(B, dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def loss(self, s_t, a_t, s_t1, s_neg=None, **kwargs):
        z_t = self.encoder(s_t)
        z_t1 = self.encoder(s_t1)
        z_pred = self.predictor(z_t, a_t)

        loss_pred = F.mse_loss(z_pred, z_t1)

        if s_neg is not None:
            B, K, _ = s_neg.shape
            z_neg = self.encoder(s_neg.reshape(B * K, -1)).reshape(B, K, -1)
            loss_nce = self.temporal_nce_loss(z_t, z_t1, z_neg)
        else:
            # Fallback: use shuffled batch as negatives
            perm = torch.randperm(z_t.size(0), device=z_t.device)
            z_neg = z_t[perm].unsqueeze(1)  # (B, 1, D)
            loss_nce = self.temporal_nce_loss(z_t, z_t1, z_neg)

        total = loss_pred + loss_nce
        return total, {"pred": loss_pred.item(), "nce": loss_nce.item()}

    def encode(self, s):
        return self.encoder(s)

    def predict(self, z, a):
        return self.predictor(z, a)
