"""Interface layer: learnable mapping from LeWM latent z to Relatum probabilistic facts.

Each predicate gets an independent probe (small MLP + sigmoid).
This modular design allows adding new predicates without retraining existing probes.

Predicates for the tentacle domain:
- curvature_high:    segment curvature exceeds safe threshold
- tension_saturated: cable tensions near maximum
- tip_deviation:     end effector far from target
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Predicate definitions for the tentacle domain
TENTACLE_PREDICATES = [
    "curvature_high",
    "tension_saturated",
    "tip_deviation",
]


class InterfaceLayer(nn.Module):
    """Learnable mapping from LeWM latent to Relatum probabilistic facts.

    Input:  (batch, latent_dim=64)
    Output: (batch, n_predicates) confidence values in [0, 1]
    """

    def __init__(
        self,
        latent_dim: int = 64,
        n_predicates: int | None = None,
        predicate_names: list[str] | None = None,
    ):
        super().__init__()

        if predicate_names is None:
            predicate_names = TENTACLE_PREDICATES
        if n_predicates is None:
            n_predicates = len(predicate_names)

        self.predicate_names = predicate_names
        self.n_predicates = n_predicates

        # Independent probe per predicate
        self.probes = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
            for _ in range(n_predicates)
        ])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute confidence for each predicate.

        Args:
            z: (batch, latent_dim) latent vectors.

        Returns:
            (batch, n_predicates) confidences in [0, 1].
        """
        confidences = torch.cat(
            [probe(z) for probe in self.probes],
            dim=-1,
        )
        return confidences

    def to_prob_facts(
        self,
        z: torch.Tensor,
        node_id: str,
    ) -> list[dict]:
        """Convert latent to Relatum-consumable probabilistic facts.

        Args:
            z: (latent_dim,) or (1, latent_dim) latent vector.
            node_id: Identifier for the entity (e.g., "seg_5", "current").

        Returns:
            List of dicts with keys: predicate, args, confidence.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)

        with torch.no_grad():
            confs = self.forward(z).squeeze(0)

        facts = []
        for i, (name, conf) in enumerate(
            zip(self.predicate_names, confs.tolist())
        ):
            facts.append({
                "predicate": name,
                "args": (node_id,),
                "confidence": conf,
            })

        return facts

    def to_relatum_assertions(
        self,
        z: torch.Tensor,
        node_id: str,
        ri,
    ) -> list[str]:
        """Assert probabilistic facts directly into a RelatumInterface.

        Args:
            z: Latent vector.
            node_id: Entity identifier.
            ri: RelatumInterface instance.

        Returns:
            List of asserted fact_ids.
        """
        facts = self.to_prob_facts(z, node_id)
        fact_ids = []
        for fact in facts:
            fid = ri.assert_probabilistic(
                predicate=fact["predicate"],
                args=fact["args"],
                confidence=fact["confidence"],
            )
            fact_ids.append(fid)
        return fact_ids
