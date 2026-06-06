"""
§6 Cohomodynamic Heat Equation
==============================

Definition 6.1.2:
    ∂Ψ/∂t = −Δ_ℬ Ψ + 𝒥

Theorem 6.1.3 (Well-Posedness, Duhamel):
    Ψ(t) = e^{−tΔ_ℬ} Ψ₀ + ∫₀ᵗ e^{−(t−s)Δ_ℬ} 𝒥 ds

Spectral solution (Theorem 6.2.1):
    Ψ(t) = Σⱼ [cⱼ e^{−λⱼt} + (𝒥ⱼ/λⱼ)(1 − e^{−λⱼt})] φⱼ

Convergence (Theorem 6.2.2):
    ‖Ψ(t) − Ψ_∞‖ ≤ C e^{−λ₁t}

Learning time (Corollary 6.2.3):
    τ_learn = 1/λ₁
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from cdi.config import CDIConfig


class HeatEquation:
    """Cohomodynamic heat equation solver.

    Provides both Euler time-stepping (O(n) per step) and exact
    spectral solutions for the belief-state evolution.

    Attributes
    ----------
    laplacian : BeliefLaplacian
    config : CDIConfig
    """

    def __init__(self, laplacian, config: CDIConfig) -> None:
        self.laplacian = laplacian
        self.config = config
        self._eigenvalues: Optional[torch.Tensor] = None
        self._eigenvectors: Optional[torch.Tensor] = None

    def _ensure_spectral(self) -> None:
        """Compute eigendecomposition if not cached."""
        if self._eigenvalues is None:
            self._eigenvalues, self._eigenvectors = self.laplacian.eigendecompose()

    # ------------------------------------------------------------------
    # Euler integration  (O(n) per step)
    # ------------------------------------------------------------------

    def evolve_euler(
        self, psi: torch.Tensor, J: torch.Tensor, dt: float, steps: int
    ) -> torch.Tensor:
        """Explicit Euler: Ψ_{t+1} = Ψ_t − dt·Δ_ℬ Ψ_t + dt·𝒥.

        Parameters
        ----------
        psi : torch.Tensor   Shape ``(N,)`` — initial state.
        J   : torch.Tensor   Shape ``(N,)`` — source term.
        dt  : float           Time step.
        steps : int           Number of integration steps.

        Returns
        -------
        torch.Tensor  Shape ``(N,)`` — state after ``steps`` steps.
        """
        for _ in range(steps):
            psi = psi - dt * self.laplacian.apply(psi) + dt * J
        return psi

    # ------------------------------------------------------------------
    # Exact spectral solution
    # ------------------------------------------------------------------

    def evolve_spectral(
        self, psi_0: torch.Tensor, J: torch.Tensor, t: float
    ) -> torch.Tensor:
        """Exact spectral solution at time t.

        Ψ(t) = Σⱼ [cⱼ e^{−λⱼt} + (𝒥ⱼ/λⱼ)(1 − e^{−λⱼt})] φⱼ

        For harmonic modes (λ ≈ 0): Ψ = c₀ + 𝒥₀·t.

        Parameters
        ----------
        psi_0 : torch.Tensor  Shape ``(N,)`` — initial state.
        J     : torch.Tensor  Shape ``(N,)`` — source.
        t     : float          Time.

        Returns
        -------
        torch.Tensor  Shape ``(N,)``.
        """
        self._ensure_spectral()
        evals = self._eigenvalues
        evecs = self._eigenvectors

        c = evecs.T @ psi_0        # (N,) — initial coefficients
        J_coeffs = evecs.T @ J     # (N,) — source coefficients

        decay = torch.exp(-evals * t)
        harmonic = evals.abs() < 1e-10
        nonharm = ~harmonic

        result_coeffs = torch.zeros_like(c)

        # Harmonic: c_j + J_j·t
        result_coeffs[harmonic] = c[harmonic] + J_coeffs[harmonic] * t

        # Non-harmonic: c_j e^{−λt} + (J_j/λ_j)(1 − e^{−λt})
        if nonharm.any():
            lam = evals[nonharm]
            result_coeffs[nonharm] = (
                c[nonharm] * decay[nonharm]
                + (J_coeffs[nonharm] / lam) * (1.0 - decay[nonharm])
            )

        return evecs @ result_coeffs

    # ------------------------------------------------------------------
    # Steady state
    # ------------------------------------------------------------------

    def steady_state(self, J: torch.Tensor) -> torch.Tensor:
        """Ψ_∞ = Δ_ℬ⁻¹ 𝒥 on (ker Δ_ℬ)⊥.

        Definition 6.3.1: Δ_ℬ Ψ_∞ = 𝒥.

        Parameters
        ----------
        J : torch.Tensor  Shape ``(N,)``.

        Returns
        -------
        torch.Tensor  Shape ``(N,)``.
        """
        self._ensure_spectral()
        evals = self._eigenvalues
        evecs = self._eigenvectors

        J_coeffs = evecs.T @ J
        result_coeffs = torch.zeros_like(J_coeffs)

        nonharm = evals.abs() > 1e-10
        result_coeffs[nonharm] = J_coeffs[nonharm] / evals[nonharm]

        return evecs @ result_coeffs

    # ------------------------------------------------------------------
    # Convergence analysis
    # ------------------------------------------------------------------

    def convergence_rate(self) -> torch.Tensor:
        """λ₁ — first positive eigenvalue."""
        self._ensure_spectral()
        positive = self._eigenvalues[self._eigenvalues > 1e-10]
        if len(positive) == 0:
            return torch.tensor(0.0, dtype=self.config.dtype)
        return positive.min()

    def learning_time(self) -> torch.Tensor:
        """τ = 1/λ₁ — characteristic learning time (Corollary 6.2.3)."""
        lam1 = self.convergence_rate()
        if lam1.abs() < 1e-12:
            return torch.tensor(float("inf"), dtype=self.config.dtype)
        return 1.0 / lam1

    def convergence_bound(
        self, psi: torch.Tensor, psi_inf: torch.Tensor, t: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Verify ‖Ψ(t) − Ψ_∞‖ ≤ C e^{−λ₁t}.

        Returns (actual_error, theoretical_bound).
        """
        actual = torch.norm(psi - psi_inf)
        C = torch.norm(psi - psi_inf)  # constant from Theorem 6.2.2
        lam1 = self.convergence_rate()
        bound = C * torch.exp(-lam1 * t)
        return actual, bound

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def invalidate_cache(self) -> None:
        """Call when Laplacian parameters change."""
        self._eigenvalues = None
        self._eigenvectors = None
