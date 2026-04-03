"""Cable geometry for the tentacle simulation.

Each segment has 4 cables arranged at 90-degree intervals around
the rod cross-section. Cable tension produces bending forces.

Cable layout (looking down the rod axis):
    Cable 0: +x direction
    Cable 1: +z direction
    Cable 2: -x direction
    Cable 3: -z direction

The offset from rod center determines the moment arm for bending.
"""

from __future__ import annotations

import numpy as np


# Cable angular positions around the rod cross-section (radians)
CABLE_ANGLES = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])

# Offset from rod centerline (fraction of rod radius)
CABLE_OFFSET_RATIO = 0.8


def cable_offset(cable_idx: int, rod_radius: float = 0.01) -> np.ndarray:
    """Compute the offset vector of a cable from the rod centerline.

    Args:
        cable_idx: Cable index (0-3).
        rod_radius: Radius of the rod cross-section.

    Returns:
        (3,) offset vector in the rod's local frame (x, y=0, z).
    """
    angle = CABLE_ANGLES[cable_idx]
    r = rod_radius * CABLE_OFFSET_RATIO
    return np.array([r * np.cos(angle), 0.0, r * np.sin(angle)])


def cable_direction(seg_idx: int, cable_idx: int) -> np.ndarray:
    """Compute the unit direction of cable force for a given segment.

    The cable pulls the segment toward the rod base (negative y-direction),
    with a lateral component determined by the cable's angular position.
    This creates bending when cables on opposite sides have different tensions.

    Args:
        seg_idx: Segment index (0-19).
        cable_idx: Cable index (0-3).

    Returns:
        (3,) unit force direction vector.
    """
    angle = CABLE_ANGLES[cable_idx]

    # Axial component: cables pull toward base (negative y)
    axial = -0.3

    # Lateral component: determined by cable position
    lateral_x = np.cos(angle)
    lateral_z = np.sin(angle)

    # Segments further from base have stronger lateral effect
    # (longer moment arm)
    lateral_scale = 0.7 * (1.0 + 0.02 * seg_idx)

    direction = np.array([
        lateral_scale * lateral_x,
        axial,
        lateral_scale * lateral_z,
    ])

    # Normalize
    norm = np.linalg.norm(direction)
    if norm > 1e-10:
        direction /= norm

    return direction


def compute_all_cable_forces(
    tensions: np.ndarray,
    n_segments: int = 20,
    n_cables: int = 4,
) -> np.ndarray:
    """Compute force vectors for all cables on all segments.

    Args:
        tensions: (n_segments * n_cables,) tension magnitudes.
        n_segments: Number of rod segments.
        n_cables: Number of cables per segment.

    Returns:
        (3, n_segments) force array, summed over cables per segment.
    """
    forces = np.zeros((3, n_segments))

    for seg_idx in range(n_segments):
        for cable_idx in range(n_cables):
            t = tensions[seg_idx * n_cables + cable_idx]
            d = cable_direction(seg_idx, cable_idx)
            forces[:, seg_idx] += t * d

    return forces
