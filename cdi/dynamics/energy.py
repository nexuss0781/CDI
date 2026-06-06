"""
§10 Energy Functional and Conservation Laws
============================================

Definition 10.1.1:
    E[Ψ] = ½⟨Ψ, Δ_ℬ Ψ⟩ − ⟨Ψ, 𝒥⟩

Theorem 10.1.2 (Energy Dissipation):
    dE/dt = −‖∂_t Ψ‖² ≤ 0

Corollary 10.1.3:
    The heat flow is the gradient flow of E w.r.t. the L² metric.
"""

from __future__ import annotations

from typing import Optional

import torch

from cdi.config import CDIConfig


class EnergyFunctional:
    """Cognitive energy and dissipation tracking.

    Attributes
    ----------
    laplacian : BeliefLaplacian
    config : CDIConfig
    """

    def __init__(self, laplacian, config: CDIConfig) -> None:
        self.laplacian = laplacian
        self.config = config

    def cognitive_energy(self, psi: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        """E[Ψ] = ½ Ψᵀ Δ_ℬ Ψ − Ψᵀ 𝒥.

        Parameters
        ----------
        psi : torch.Tensor  Shape ``(N,)``.
        J   : torch.Tensor  Shape ``(N,)``.

        Returns
        -------
        torch.Tensor  Scalar.
        """
        lap_psi = self.laplacian.apply(psi)
        return 0.5 * torch.dot(psi, lap_psi) - torch.dot(psi, J)

    def energy_gradient(self, psi: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        """∇E = Δ_ℬ Ψ − 𝒥 = −∂_t Ψ.

        Parameters
        ----------
        psi, J : torch.Tensor  Shape ``(N,)``.

        Returns
        -------
        torch.Tensor  Shape ``(N,)``.
        """
        return self.laplacian.apply(psi) - J

    def dissipation_rate(self, psi: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        """dE/dt = −‖∂_t Ψ‖² = −‖Δ_ℬ Ψ − 𝒥‖².

        Returns
        -------
        torch.Tensor  Non-positive scalar.
        """
        grad = self.energy_gradient(psi, J)
        return -torch.dot(grad, grad)

    def verify_dissipation(
        self,
        psi_t: torch.Tensor,
        psi_t_next: torch.Tensor,
        J: torch.Tensor,
    ) -> bool:
        """Check E(t + dt) ≤ E(t)."""
        e_now = self.cognitive_energy(psi_t, J)
        e_next = self.cognitive_energy(psi_t_next, J)
        return bool(e_next <= e_now + 1e-10)

    def lagrangian(
        self,
        psi: torch.Tensor,
        J: torch.Tensor,
        superconn_apply=None,
    ) -> torch.Tensor:
        """Lagrangian ℒ[Ψ] = ½⟨𝔸Ψ, 𝔸Ψ⟩ − ⟨Ψ, 𝒥⟩.

        If superconn_apply is None, falls back to E[Ψ].

        Parameters
        ----------
        psi : torch.Tensor  Shape ``(N,)``.
        J   : torch.Tensor  Shape ``(N,)``.
        superconn_apply : callable, optional
            Function 𝔸·Ψ.

        Returns
        -------
        torch.Tensor  Scalar.
        """
        if superconn_apply is not None:
            A_psi = superconn_apply(psi)
            return 0.5 * torch.dot(A_psi, A_psi) - torch.dot(psi, J)
        return self.cognitive_energy(psi, J)
