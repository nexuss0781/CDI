"""
CDI Configuration
=================

All parameters map to mathematical objects in the CDI specification.
Uses dataclass for immutability and type safety.
"""
from dataclasses import dataclass, field
from typing import Tuple, Optional
import torch


@dataclass
class CDIConfig:
    """Configuration for the Cohomodynamic Intelligence engine.

    Every parameter corresponds to a mathematical quantity in the
    CDI Mathematical Specification v1.0.
    """

    # ═══════════════════════════════════════════════════════════════
    # §1  Cognitive Manifold (M, g)
    # ═══════════════════════════════════════════════════════════════
    manifold_dim: int = 3
    """Dimension *d* of the cognitive manifold M."""

    n_points: int = 64
    """Number of discretisation points on M."""

    # ═══════════════════════════════════════════════════════════════
    # §2  Good Cover & Observation Sheaf
    # ═══════════════════════════════════════════════════════════════
    cover_k: int = 8
    """k for k-nearest-neighbour cover construction."""

    observation_dim: int = 3
    """Dimension of the observation space (input)."""

    output_dim: int = 3
    """Dimension of the output prediction space."""

    # ═══════════════════════════════════════════════════════════════
    # §3  Belief Complex B^•
    # ═══════════════════════════════════════════════════════════════
    motor_depth: int = 1
    """m — number of negative-degree (motor) sheaves."""

    abstraction_height: int = 3
    """n — number of positive-degree (abstraction) sheaves."""

    belief_dims: Tuple[int, ...] = (16, 32, 32, 16, 8)
    """Dimensions (dim B_{-m}, ..., dim B_0, ..., dim B_n).
    Length MUST equal motor_depth + abstraction_height + 1."""

    # ═══════════════════════════════════════════════════════════════
    # §6  Heat-Equation Dynamics
    # ═══════════════════════════════════════════════════════════════
    heat_dt: float = 0.01
    """Time-step Δt for heat-equation integration."""

    heat_steps: int = 50
    """Number of heat-equation steps per learning iteration."""

    # ═══════════════════════════════════════════════════════════════
    # Training hyper-parameters
    # ═══════════════════════════════════════════════════════════════
    learning_rate: float = 1e-3
    """Learning rate for parameter-space gradient updates."""

    consistency_weight: float = 10.0
    """Weight λ_c for the δ²=0 consistency penalty."""

    energy_weight: float = 1.0
    """Weight λ_e for the cognitive-energy regulariser."""

    bianchi_weight: float = 1.0
    """Weight for the Bianchi-identity penalty ‖d_A F_A‖²."""

    epochs: int = 100
    """Number of training epochs."""

    batch_size: int = 32
    """Mini-batch size."""

    finetune_interval: int = 10
    """Run fine-tuning every N epochs."""

    eval_interval: int = 5
    """Run evaluation every N epochs."""

    # ═══════════════════════════════════════════════════════════════
    # System / runtime
    # ═══════════════════════════════════════════════════════════════
    device: str = "cpu"
    """Compute device ('cpu' or 'cuda')."""

    dtype_str: str = "float64"
    """String name of the torch dtype (for serialisation)."""

    seed: int = 42
    """Random seed for reproducibility."""

    spectral_cutoff: int = 0
    """Keep only this many eigenvalues (0 → keep all)."""

    # ───────────────────────────────────────────────────────────────
    # Derived / computed properties
    # ───────────────────────────────────────────────────────────────
    @property
    def dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype_str)

    @property
    def n_degrees(self) -> int:
        """Total number of belief degrees: m + n + 1."""
        return self.motor_depth + self.abstraction_height + 1

    @property
    def degree_range(self) -> range:
        """Integer range [−m … n]."""
        return range(-self.motor_depth, self.abstraction_height + 1)

    @property
    def total_belief_dim(self) -> int:
        """Σ_k dim(B_k)."""
        return sum(self.belief_dims)

    @property
    def spinor_dim(self) -> int:
        """dim(S) = 2^⌊d/2⌋."""
        return 2 ** (self.manifold_dim // 2)

    @property
    def twisted_bundle_dim(self) -> int:
        """dim(𝔹) = dim(S) × Σ_k dim(B_k)."""
        return self.spinor_dim * self.total_belief_dim

    @property
    def total_state_dim(self) -> int:
        """Total discrete state-vector length: n × dim(𝔹)."""
        return self.n_points * self.twisted_bundle_dim

    def belief_dim(self, degree: int) -> int:
        """Dimension of belief sheaf at a given degree k ∈ [−m, n]."""
        idx = degree + self.motor_depth
        return self.belief_dims[idx]

    def belief_offset(self, degree: int) -> int:
        """Offset into the concatenated belief vector for degree k."""
        idx = degree + self.motor_depth
        return sum(self.belief_dims[:idx])

    # ───────────────────────────────────────────────────────────────
    # Validation
    # ───────────────────────────────────────────────────────────────
    def validate(self) -> None:
        """Raise AssertionError on inconsistent settings."""
        assert len(self.belief_dims) == self.n_degrees, (
            f"belief_dims has {len(self.belief_dims)} entries but "
            f"n_degrees = {self.n_degrees}"
        )
        assert self.manifold_dim >= 1, "manifold_dim must be ≥ 1"
        assert self.n_points >= 2, "n_points must be ≥ 2"
        assert 0 < self.cover_k < self.n_points, (
            f"cover_k={self.cover_k} out of range (0, {self.n_points})"
        )
        assert self.heat_dt > 0, "heat_dt must be positive"
        assert all(d > 0 for d in self.belief_dims), (
            "Every belief_dims entry must be positive"
        )

    # ───────────────────────────────────────────────────────────────
    # Convenience constructors
    # ───────────────────────────────────────────────────────────────
    @classmethod
    def tiny(cls) -> "CDIConfig":
        """Minimal config for unit tests (fast)."""
        return cls(
            manifold_dim=2,
            n_points=8,
            cover_k=3,
            motor_depth=1,
            abstraction_height=2,
            belief_dims=(4, 8, 8, 4),
            observation_dim=2,
            output_dim=2,
            heat_steps=10,
        )

    @classmethod
    def small(cls) -> "CDIConfig":
        """Small config for integration tests."""
        return cls(
            manifold_dim=2,
            n_points=16,
            cover_k=5,
            motor_depth=1,
            abstraction_height=2,
            belief_dims=(8, 16, 16, 8),
            observation_dim=2,
            output_dim=2,
            heat_steps=20,
        )

    @classmethod
    def medium(cls) -> "CDIConfig":
        """Medium config for training experiments."""
        return cls(
            manifold_dim=3,
            n_points=64,
            cover_k=8,
            motor_depth=1,
            abstraction_height=3,
            belief_dims=(16, 32, 32, 16, 8),
            observation_dim=3,
            output_dim=3,
            heat_steps=50,
        )
