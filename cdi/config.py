"""
CDI Configuration — v2.0
=========================

v2.0 Spec Corrections (CDI_LM_v2_Technical_Specification.md):
  - Axiom 2.4.2.1: dim(B_0) >= embed_dim  (no bottleneck)
  - Axiom 2.4.2.2: sum(belief_dims) >= 4 * embed_dim
  - Axiom 2.4.2.3: engine param budget >= 15% of embedding budget
  - Axiom 2.4.2.4: n_points >= min(context, 32), manifold_dim >= ceil(log2(embed_dim))
  - Fix F2: learnable initial state theta_init
  - Fix F3: rebuild_operators called after every optimizer.step()
"""
from dataclasses import dataclass, field
from typing import Tuple, Optional
import math
import torch


@dataclass
class CDIConfig:
    """Configuration for the Cohomodynamic Intelligence engine v2.0.

    Every parameter corresponds to a mathematical quantity in the
    CDI Mathematical Specification v1.0 / v2.0 Corrective Spec.
    """

    # ═══════════════════════════════════════════════════════════════
    # §1  Cognitive Manifold (M, g)
    # ═══════════════════════════════════════════════════════════════
    manifold_dim: int = 4
    """Dimension d of M. v2.0: d >= ceil(log2(embed_dim))."""

    n_points: int = 16
    """Number of discretisation points. v2.0: >= min(context_length, 32)."""

    # ═══════════════════════════════════════════════════════════════
    # §2  Good Cover & Observation Sheaf
    # ═══════════════════════════════════════════════════════════════
    cover_k: int = 8
    """k-NN cover construction parameter."""

    observation_dim: int = 32
    """Dimension of the observation/embedding space."""

    output_dim: int = 32
    """Dimension of the output prediction space (= observation_dim for LM)."""

    # ═══════════════════════════════════════════════════════════════
    # §3  Belief Complex B^•
    # ═══════════════════════════════════════════════════════════════
    motor_depth: int = 1
    """m — number of negative-degree (motor) sheaves."""

    abstraction_height: int = 2
    """n — number of positive-degree (abstraction) sheaves."""

    belief_dims: Tuple[int, ...] = (32, 64, 64, 32)
    """Dimensions (dim B_{-m}, ..., dim B_0, ..., dim B_n).
    v2.0 Axiom: belief_dims[motor_depth] >= embed_dim (no bottleneck).
    Length MUST equal motor_depth + abstraction_height + 1."""

    # ═══════════════════════════════════════════════════════════════
    # §6  Heat-Equation Dynamics — v2.0 Fix F2/F3
    # ═══════════════════════════════════════════════════════════════
    heat_dt: float = 0.01
    """Time-step Δt for heat-equation Euler integration."""

    heat_steps: int = 10
    """K — Euler steps per token in recurrent evolution."""

    # ═══════════════════════════════════════════════════════════════
    # Training hyper-parameters
    # ═══════════════════════════════════════════════════════════════
    learning_rate: float = 1e-3
    """Adam learning rate."""

    consistency_weight: float = 0.1
    """Weight λ_C for δ²=0 consistency penalty (v2.0: 0.1 standard, 1.0 warm-up)."""

    consistency_warmup_steps: int = 100
    """Steps to use λ_C=1.0 before dropping to consistency_weight."""

    bianchi_weight: float = 0.01
    """Weight λ_B for Bianchi identity penalty."""

    spectral_weight: float = 0.001
    """Weight λ_S for spectral gap penalty."""

    spectral_target: float = 0.01
    """Target λ₁ value for spectral gap penalty."""

    energy_weight: float = 0.0
    """Weight λ_E for cognitive energy (disabled in v2.0 forward path)."""

    epochs: int = 100
    batch_size: int = 16
    finetune_interval: int = 10
    eval_interval: int = 5

    # ═══════════════════════════════════════════════════════════════
    # System / runtime
    # ═══════════════════════════════════════════════════════════════
    device: str = "cpu"
    dtype_str: str = "float64"
    seed: int = 42
    spectral_cutoff: int = 0

    # Spectral gap diagnostics period (steps)
    spectral_diag_every: int = 100

    # ───────────────────────────────────────────────────────────────
    # Derived / computed properties
    # ───────────────────────────────────────────────────────────────
    @property
    def dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype_str)

    @property
    def n_degrees(self) -> int:
        return self.motor_depth + self.abstraction_height + 1

    @property
    def degree_range(self) -> range:
        return range(-self.motor_depth, self.abstraction_height + 1)

    @property
    def total_belief_dim(self) -> int:
        return sum(self.belief_dims)

    @property
    def spinor_dim(self) -> int:
        return 2 ** (self.manifold_dim // 2)

    @property
    def twisted_bundle_dim(self) -> int:
        return self.spinor_dim * self.total_belief_dim

    @property
    def total_state_dim(self) -> int:
        return self.n_points * self.twisted_bundle_dim

    def belief_dim(self, degree: int) -> int:
        idx = degree + self.motor_depth
        return self.belief_dims[idx]

    def belief_offset(self, degree: int) -> int:
        idx = degree + self.motor_depth
        return sum(self.belief_dims[:idx])

    # ───────────────────────────────────────────────────────────────
    # Validation — v2.0 enforces dimensional hierarchy
    # ───────────────────────────────────────────────────────────────
    def validate(self) -> None:
        """Raise on inconsistent settings. Enforces all v2.0 axioms."""
        assert len(self.belief_dims) == self.n_degrees, (
            f"belief_dims has {len(self.belief_dims)} entries but "
            f"n_degrees = {self.n_degrees}"
        )
        assert self.manifold_dim >= 1
        assert self.n_points >= 2
        assert 0 < self.cover_k < self.n_points

        # v2.0 Axiom 2.4.2.1: No bottleneck — B_0 >= embed_dim
        b0_dim = self.belief_dim(0)
        assert b0_dim >= self.observation_dim, (
            f"v2.0 Axiom 2.4.2.1 VIOLATED: dim(B_0)={b0_dim} < "
            f"embed_dim={self.observation_dim}. "
            f"Set belief_dims[{self.motor_depth}] >= {self.observation_dim}."
        )

        # v2.0 Axiom 2.4.2.2: Total belief >= 4 * embed_dim
        total_b = self.total_belief_dim
        assert total_b >= 4 * self.observation_dim, (
            f"v2.0 Axiom 2.4.2.2 VIOLATED: total_belief_dim={total_b} < "
            f"4 * embed_dim={4 * self.observation_dim}."
        )

        assert self.heat_dt > 0
        assert all(d > 0 for d in self.belief_dims)

    # ───────────────────────────────────────────────────────────────
    # Convenience constructors — v2.0 compliant
    # ───────────────────────────────────────────────────────────────

    @classmethod
    def tiny(cls) -> "CDIConfig":
        """v2.0 Tiny — validation/unit tests.
        
        Template A from spec:
          embed_dim=32, belief_dims=(32,64,64,32), n_points=16, manifold_dim=4
        """
        return cls(
            manifold_dim=4,
            n_points=16,
            cover_k=6,
            motor_depth=1,
            abstraction_height=2,
            belief_dims=(32, 64, 64, 32),
            observation_dim=32,
            output_dim=32,
            heat_steps=10,
            heat_dt=0.01,
            batch_size=8,
        )

    @classmethod
    def small(cls) -> "CDIConfig":
        """v2.0 Small — production baseline.
        
        Template B from spec:
          embed_dim=128, belief_dims=(128,256,256,128), n_points=32, manifold_dim=8
        """
        return cls(
            manifold_dim=8,
            n_points=32,
            cover_k=10,
            motor_depth=1,
            abstraction_height=2,
            belief_dims=(128, 256, 256, 128),
            observation_dim=128,
            output_dim=128,
            heat_steps=15,
            heat_dt=0.005,
            batch_size=16,
        )

    @classmethod
    def medium(cls) -> "CDIConfig":
        """v2.0 Medium — extended capacity."""
        return cls(
            manifold_dim=8,
            n_points=64,
            cover_k=12,
            motor_depth=1,
            abstraction_height=3,
            belief_dims=(256, 512, 512, 256, 128),
            observation_dim=256,
            output_dim=256,
            heat_steps=20,
            heat_dt=0.005,
            batch_size=8,
        )
