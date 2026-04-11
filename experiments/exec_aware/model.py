"""Execution-Aware Encoder: InfoNCE + execution consistency regularization.

The key insight: InfoNCE optimizes temporal similarity but doesn't guarantee
that lever execution outcomes are predictable in latent space. Adding an
execution consistency loss forces the encoder to produce representations
where same-lever transitions map to consistent latent displacements.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase4.models.lewm_tentacle import TentacleEncoder, TentaclePredictor
from phase4.envs.tentacle_env import STATE_DIM, ACTION_DIM
from experiments.lever_control.levers import LEVERS

N_LEVERS = len(LEVERS)
LEVER_TO_IDX = {l: i for i, l in enumerate(LEVERS)}


class ExecutionAwareEncoder(nn.Module):
    """Encoder with execution consistency regularization.

    loss = InfoNCE(z_t, z_{t+1}, z_neg) + lambda_exec * MSE(z_pred, z_after)

    The exec_predictor learns: given (z_before, lever_id) -> z_after,
    forcing the encoder to produce latent spaces where lever transitions
    are predictable (low drift after discretization).
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        latent_dim: int = 64,
        proj_dim: int = 32,
        temperature: float = 0.1,
        lambda_exec: float = 0.5,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.lambda_exec = lambda_exec

        # Shared encoder (same architecture as contrastive)
        self.encoder = TentacleEncoder(state_dim, latent_dim)

        # Standard predictor (for temporal prediction loss)
        self.predictor = TentaclePredictor(latent_dim, action_dim)

        # Contrastive projection head
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

        # Lever embedding + execution predictor (NEW)
        self.lever_embed = nn.Embedding(N_LEVERS, 16)
        self.exec_predictor = nn.Sequential(
            nn.Linear(latent_dim + 16, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def encode(self, s: torch.Tensor) -> torch.Tensor:
        return self.encoder(s)

    def predict(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.predictor(z, a)

    def forward_nce(
        self,
        s_t: torch.Tensor,
        s_t1: torch.Tensor,
        s_neg: torch.Tensor,
    ) -> torch.Tensor:
        """InfoNCE temporal contrastive loss."""
        z_t = self.encoder(s_t)
        z_t1 = self.encoder(s_t1)
        z_neg = self.encoder(s_neg)

        q = F.normalize(self.projector(z_t), dim=-1)
        k = F.normalize(self.projector(z_t1), dim=-1)
        n = F.normalize(self.projector(z_neg), dim=-1)

        pos = (q * k).sum(-1, keepdim=True) / self.temperature
        neg = (q * n).sum(-1, keepdim=True) / self.temperature

        logits = torch.cat([pos, neg], dim=-1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def forward_exec(
        self,
        s_before: torch.Tensor,
        lever_idx: torch.Tensor,
        s_after: torch.Tensor,
    ) -> torch.Tensor:
        """Execution consistency loss."""
        z_before = self.encoder(s_before)
        z_after = self.encoder(s_after)

        lever_emb = self.lever_embed(lever_idx)
        z_pred = self.exec_predictor(torch.cat([z_before, lever_emb], dim=-1))

        return F.mse_loss(z_pred, z_after.detach())

    def compute_loss(
        self,
        s_t: torch.Tensor,
        s_t1: torch.Tensor,
        s_neg: torch.Tensor,
        s_before: torch.Tensor,
        lever_idx: torch.Tensor,
        s_after: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        loss_nce = self.forward_nce(s_t, s_t1, s_neg)
        loss_exec = self.forward_exec(s_before, lever_idx, s_after)
        loss_total = loss_nce + self.lambda_exec * loss_exec

        return loss_total, {
            "nce": loss_nce.item(),
            "exec": loss_exec.item(),
            "total": loss_total.item(),
        }
