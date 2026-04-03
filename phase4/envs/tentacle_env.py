"""20-segment soft tentacle simulation environment.

Based on the Cosserat rod model from PyElastica.
Each segment has 4 cable actuators (pull only, T >= 0).

State space:
    Per segment: position (x, y, z) + velocity (vx, vy, vz) + curvature (1) = 7 dims
    20 segments -> total state dim = 140

Control space:
    4 cables per segment x 20 segments = 80 tension values
    Constraint: T_i >= 0 (cables can only pull, not push)
"""

from __future__ import annotations

import numpy as np

try:
    import elastica as ea
    from elastica.wrappers import (
        BaseSystemCollection,
        Connections,
        Forcing,
        CallBacks,
    )
    HAS_ELASTICA = True
except ImportError:
    HAS_ELASTICA = False

from .cable_geometry import cable_direction, compute_all_cable_forces


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SEGMENTS = 20
N_CABLES = 4
STATE_DIM = 140  # 7 per segment x 20
ACTION_DIM = N_SEGMENTS * N_CABLES  # 80
MAX_TENSION = 5.0

# Rod physical parameters
ROD_LENGTH = 1.0
ROD_RADIUS = 0.01
ROD_DENSITY = 1000.0
ROD_YOUNGS_MODULUS = 1e6
ROD_SHEAR_MODULUS = 1e4


# ---------------------------------------------------------------------------
# Environment (with PyElastica backend)
# ---------------------------------------------------------------------------

if HAS_ELASTICA:
    class TentacleEnv(BaseSystemCollection, Connections, Forcing, CallBacks):
        """20-segment soft tentacle simulation environment.

        Uses PyElastica Cosserat rod model for physically accurate
        soft-body dynamics.
        """
        pass
else:
    class TentacleEnv:
        """Fallback tentacle environment using simplified dynamics.

        Used when PyElastica is not installed. Implements a simplified
        Euler-Bernoulli beam model for testing and development.
        """

        def __init__(self):
            self.rods = []

        def append(self, rod):
            self.rods.append(rod)


class SimplifiedRod:
    """Simplified rod model for when PyElastica is unavailable.

    Tracks positions, velocities, and directors for each node.
    Uses a simplified elastic model for dynamics.
    """

    def __init__(
        self,
        n_elements: int = N_SEGMENTS,
        base_length: float = ROD_LENGTH,
        base_radius: float = ROD_RADIUS,
        density: float = ROD_DENSITY,
        youngs_modulus: float = ROD_YOUNGS_MODULUS,
    ):
        self.n_elements = n_elements
        self.n_nodes = n_elements + 1
        self.base_length = base_length
        self.element_length = base_length / n_elements
        self.base_radius = base_radius
        self.density = density
        self.youngs_modulus = youngs_modulus

        # Mass per node
        volume = np.pi * base_radius**2 * self.element_length
        self.node_mass = density * volume

        # Stiffness
        I = np.pi * base_radius**4 / 4  # noqa: E741
        self.bending_stiffness = youngs_modulus * I

        # State arrays: (3, n_nodes) for positions and velocities
        self.position_collection = np.zeros((3, self.n_nodes))
        self.velocity_collection = np.zeros((3, self.n_nodes))
        self.external_forces = np.zeros((3, self.n_nodes))

        # Director collection: (3, 3, n_elements) rotation matrices
        self.director_collection = np.zeros((3, 3, n_elements))
        for i in range(n_elements):
            self.director_collection[:, :, i] = np.eye(3)

        # Initialize straight rod along y-axis
        for i in range(self.n_nodes):
            self.position_collection[1, i] = i * self.element_length

        # Damping coefficient
        self.damping = 0.05

    def compute_internal_forces(self) -> np.ndarray:
        """Compute elastic restoring forces from bending."""
        forces = np.zeros((3, self.n_nodes))

        for i in range(1, self.n_nodes - 1):
            # Curvature-based restoring force
            d1 = self.position_collection[:, i] - self.position_collection[:, i - 1]
            d2 = self.position_collection[:, i + 1] - self.position_collection[:, i]

            # Bending force proportional to curvature
            curvature_vec = d2 - d1
            force = self.bending_stiffness * curvature_vec / (self.element_length**2)
            forces[:, i] += force

        return forces

    def step(self, dt: float = 1e-4) -> None:
        """Advance one timestep using semi-implicit Euler."""
        internal = self.compute_internal_forces()
        total_forces = internal + self.external_forces

        # Damping
        total_forces -= self.damping * self.velocity_collection

        # Update velocities (skip node 0: fixed base)
        for i in range(1, self.n_nodes):
            self.velocity_collection[:, i] += (
                total_forces[:, i] / self.node_mass * dt
            )

        # Update positions
        for i in range(1, self.n_nodes):
            self.position_collection[:, i] += self.velocity_collection[:, i] * dt

        # Clear external forces for next step
        self.external_forces = np.zeros((3, self.n_nodes))


def make_tentacle(
    n_segments: int = N_SEGMENTS,
    n_cables: int = N_CABLES,
) -> tuple:
    """Construct a tentacle system.

    Returns:
        Tuple of (env, rod).
    """
    env = TentacleEnv()

    if HAS_ELASTICA:
        rod = ea.CosseratRod.straight_rod(
            n_elements=n_segments,
            start=np.array([0.0, 0.0, 0.0]),
            direction=np.array([0.0, 1.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            base_length=ROD_LENGTH,
            base_radius=ROD_RADIUS,
            density=ROD_DENSITY,
            youngs_modulus=ROD_YOUNGS_MODULUS,
            shear_modulus=ROD_SHEAR_MODULUS,
        )
    else:
        rod = SimplifiedRod(
            n_elements=n_segments,
            base_length=ROD_LENGTH,
            base_radius=ROD_RADIUS,
            density=ROD_DENSITY,
            youngs_modulus=ROD_YOUNGS_MODULUS,
        )

    env.append(rod)
    return env, rod


def step(
    env: TentacleEnv,
    rod,
    tensions: np.ndarray,
    dt: float = 1e-4,
    n_steps: int = 100,
) -> tuple[np.ndarray, float]:
    """Apply cable tensions and simulate n_steps forward.

    Args:
        env: The tentacle environment.
        rod: The Cosserat rod or SimplifiedRod.
        tensions: (80,) tension vector, 4 cables per segment.
        dt: Timestep for integration.
        n_steps: Number of integration steps.

    Returns:
        Tuple of (new_state (140,), total_energy_consumed).
    """
    # Enforce non-negative tensions (cables can only pull)
    tensions = np.clip(tensions, 0.0, MAX_TENSION)

    total_work = 0.0

    for _ in range(n_steps):
        # Apply cable forces
        cable_forces = compute_all_cable_forces(tensions, N_SEGMENTS, N_CABLES)

        # Map to rod nodes (n_nodes = n_segments + 1, distribute to nearest)
        n_nodes = rod.position_collection.shape[1]
        for seg_idx in range(min(N_SEGMENTS, n_nodes)):
            rod.external_forces[:, seg_idx] += cable_forces[:, min(seg_idx, N_SEGMENTS - 1)]

        if HAS_ELASTICA:
            ea.integrate(ea.PositionVerlet(), env, dt, 1)
        else:
            rod.step(dt)

        # Compute instantaneous power = |F . v|
        v = rod.velocity_collection
        f = rod.external_forces
        # Element-wise product summed over spatial dims
        power = np.abs(np.sum(f * v[:, :f.shape[1]]))
        total_work += power * dt

    state = extract_state(rod)
    return state, total_work


def extract_state(rod) -> np.ndarray:
    """Extract state vector from rod.

    Returns:
        (140,) vector: per-segment [x, y, z, vx, vy, vz, curvature].
    """
    pos = rod.position_collection  # (3, n_nodes)
    vel = rod.velocity_collection  # (3, n_nodes)

    n_nodes = pos.shape[1]
    n_seg = min(N_SEGMENTS, n_nodes - 1)

    state_parts = []
    for i in range(n_seg):
        # Position of segment center
        p = 0.5 * (pos[:, i] + pos[:, i + 1])
        # Velocity of segment center
        v = 0.5 * (vel[:, i] + vel[:, i + 1])
        # Local curvature (approximate)
        if 0 < i < n_seg - 1:
            d1 = pos[:, i] - pos[:, i - 1]
            d2 = pos[:, i + 1] - pos[:, i]
            cross = np.cross(d1, d2)
            curvature = np.linalg.norm(cross) / (
                np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-10
            )
        else:
            curvature = 0.0

        # Normalize
        seg_state = np.concatenate([
            p / ROD_LENGTH,      # normalize to rod length
            v / 0.1,             # normalize to typical velocity
            [curvature / 10.0],  # normalize curvature
        ])
        state_parts.append(seg_state)

    state = np.concatenate(state_parts)

    # Pad or truncate to STATE_DIM
    if len(state) < STATE_DIM:
        state = np.pad(state, (0, STATE_DIM - len(state)))
    else:
        state = state[:STATE_DIM]

    return state


def set_state(rod, state: np.ndarray) -> None:
    """Set rod state from a state vector.

    Inverse of extract_state: distributes state vector
    back to rod position and velocity arrays.
    """
    n_nodes = rod.position_collection.shape[1]
    n_seg = min(N_SEGMENTS, n_nodes - 1)

    for i in range(n_seg):
        offset = i * 7
        if offset + 6 >= len(state):
            break

        # Un-normalize position
        px, py, pz = state[offset:offset + 3] * ROD_LENGTH
        # Un-normalize velocity
        vx, vy, vz = state[offset + 3:offset + 6] * 0.1

        # Set node positions (distribute to both nodes of segment)
        rod.position_collection[:, i] = np.array([px, py, pz])
        if i + 1 < n_nodes:
            rod.position_collection[:, i + 1] = np.array([px, py, pz])

        # Set velocities
        rod.velocity_collection[:, i] = np.array([vx, vy, vz])
        if i + 1 < n_nodes:
            rod.velocity_collection[:, i + 1] = np.array([vx, vy, vz])


def random_valid_state(seed: int | None = None) -> np.ndarray:
    """Generate a random physically plausible tentacle state.

    Creates states by simulating from rest with random cable tensions
    for a short duration, ensuring physical validity.
    """
    rng = np.random.RandomState(seed)

    env, rod = make_tentacle()

    # Apply random tensions for a few steps to get a non-trivial state
    n_warmup = 50
    for _ in range(n_warmup):
        tensions = rng.exponential(0.3, size=ACTION_DIM)
        tensions = np.clip(tensions, 0.0, MAX_TENSION)

        cable_forces = compute_all_cable_forces(tensions)
        n_nodes = rod.position_collection.shape[1]
        for seg_idx in range(min(N_SEGMENTS, n_nodes)):
            rod.external_forces[:, seg_idx] += cable_forces[
                :, min(seg_idx, N_SEGMENTS - 1)
            ]

        if HAS_ELASTICA:
            ea.integrate(ea.PositionVerlet(), env, 1e-4, 1)
        else:
            rod.step(1e-4)

    state = extract_state(rod)
    del env, rod
    return state
