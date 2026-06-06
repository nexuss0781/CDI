"""
§10.2 Gauge Transformations and Noether Currents
==================================================

Definition 10.2.1: A gauge transformation U ∈ Γ(M, U(𝔹)) is unitary
and preserves δ: UδU⁻¹ = δ.

Theorem 10.2.2: 𝔸Ψ = 𝒥 is gauge-invariant under Ψ → UΨ, 𝔸 → U𝔸U⁻¹.

Theorem 10.3.1 (Noether): Every continuous symmetry of the Lagrangian
yields a conserved current J^μ with ∇_μ J^μ = 0.
"""

from __future__ import annotations

from typing import Optional

import torch

from cdi.config import CDIConfig


class GaugeTransformation:
    """Gauge symmetry verification and Noether current computation.

    Attributes
    ----------
    config : CDIConfig
    """

    def __init__(self, config: CDIConfig) -> None:
        self.config = config
        self.N = config.total_state_dim

    def random_gauge(self, epsilon: float = 0.1) -> torch.Tensor:
        """Generate a random unitary gauge transformation near identity.

        U = exp(εH) ≈ I + εH + ε²H²/2  where H is skew-symmetric.

        Parameters
        ----------
        epsilon : float  Perturbation magnitude.

        Returns
        -------
        torch.Tensor  Shape ``(N, N)`` — approximately unitary.
        """
        H = torch.randn(self.N, self.N, dtype=self.config.dtype)
        H = epsilon * (H - H.T) / 2.0  # skew-symmetric
        # First-order approximation of matrix exponential
        I = torch.eye(self.N, dtype=self.config.dtype)
        U = I + H + 0.5 * H @ H
        return U

    def apply_to_state(self, U: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Ψ → UΨ."""
        return U @ state

    def apply_to_operator(self, U: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
        """𝔸 → U𝔸U⁻¹."""
        U_inv = torch.linalg.inv(U)
        return U @ op @ U_inv

    def verify_invariance(
        self,
        U: torch.Tensor,
        state: torch.Tensor,
        J: torch.Tensor,
        superconn_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Check ‖U𝔸U⁻¹(UΨ) − U𝒥‖.

        Should be ≈ 0 for exact gauge invariance.

        Returns
        -------
        torch.Tensor  Scalar error.
        """
        U_psi = self.apply_to_state(U, state)
        U_A_Uinv = self.apply_to_operator(U, superconn_matrix)
        lhs = U_A_Uinv @ U_psi
        rhs = U @ J
        return torch.norm(lhs - rhs)

    def noether_current(
        self,
        state: torch.Tensor,
        J: torch.Tensor,
        generator: torch.Tensor,
    ) -> torch.Tensor:
        """Conserved Noether current from a symmetry generator.

        For an infinitesimal gauge transformation U = I + εH,
        the Noether current is J^μ = ⟨Ψ, H·Ψ⟩.

        Parameters
        ----------
        state : torch.Tensor      Shape ``(N,)``.
        J : torch.Tensor           Shape ``(N,)`` — source.
        generator : torch.Tensor   Shape ``(N, N)`` — skew-symmetric H.

        Returns
        -------
        torch.Tensor  Scalar — conserved charge.
        """
        return torch.dot(state, generator @ state)
