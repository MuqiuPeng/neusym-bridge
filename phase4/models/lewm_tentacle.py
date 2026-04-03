"""LeWM (Latent World Model) for the tentacle domain.

Architecture:
    Encoder:   state (140) -> latent z (64)
    Predictor: (z, action) -> z_next (64)
    Decoder:   z -> reconstructed state (140)

The decoder prevents representation collapse (Phase 1 lesson).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..envs.tentacle_env import STATE_DIM, ACTION_DIM


class TentacleEncoder(nn.Module):
    """Encode tentacle state into latent representation.

    Input:  (batch, 140) state vector
    Output: (batch, latent_dim)
    """

    def __init__(self, state_dim: int = STATE_DIM, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TentaclePredictor(nn.Module):
    """Predict next latent state given current latent and action.

    Input:  (batch, latent_dim + action_dim)
    Output: (batch, latent_dim)
    """

    def __init__(self, latent_dim: int = 64, action_dim: int = ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a], dim=-1))


class TentacleDecoder(nn.Module):
    """Decode latent back to state space (prevents representation collapse).

    Phase 1 finding: reconstruction loss is necessary to prevent
    the encoder from collapsing all states to the same latent.

    Input:  (batch, latent_dim)
    Output: (batch, state_dim)
    """

    def __init__(self, latent_dim: int = 64, state_dim: int = STATE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LeWMTentacle(nn.Module):
    """Full Latent World Model for tentacle control.

    Combines encoder, predictor, and decoder.
    Training loss = prediction_loss + reconstruction_loss.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.encoder = TentacleEncoder(state_dim=state_dim, latent_dim=latent_dim)
        self.predictor = TentaclePredictor(latent_dim=latent_dim, action_dim=action_dim)
        self.decoder = TentacleDecoder(latent_dim=latent_dim, state_dim=state_dim)

    def forward(
        self,
        s_t: torch.Tensor,
        a_t: torch.Tensor,
        s_t1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        Args:
            s_t:  (batch, state_dim) current state.
            a_t:  (batch, action_dim) action taken.
            s_t1: (batch, state_dim) next state (ground truth).

        Returns:
            z_pred:  predicted next latent.
            z_t1:    encoded next latent (prediction target).
            s_recon: reconstructed current state.
            s_t:     original current state (for loss computation).
        """
        z_t = self.encoder(s_t)
        z_t1 = self.encoder(s_t1)
        z_pred = self.predictor(z_t, a_t)
        s_recon = self.decoder(z_t)
        return z_pred, z_t1, s_recon, s_t

    def encode(self, s: torch.Tensor) -> torch.Tensor:
        return self.encoder(s)

    def predict(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.predictor(z, a)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
