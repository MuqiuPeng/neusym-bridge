"""LeWM (Latent World Model) for tentacle control."""

from .lewm_tentacle import (
    TentacleEncoder,
    TentaclePredictor,
    TentacleDecoder,
    LeWMTentacle,
)

__all__ = [
    "TentacleEncoder",
    "TentaclePredictor",
    "TentacleDecoder",
    "LeWMTentacle",
]
