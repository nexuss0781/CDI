"""
§6 Cohomodynamic Heat Equation — v2.0
=======================================

v2.0 Spec §2.2 / Fix F2 — CRITICAL CHANGES:
  - Belief state Ψ persists across tokens (recurrent, not reset to zero)
  - Initial state theta_init is a LEARNABLE parameter (CDIEngine owns it)
  - evolve_euler() takes the current state and evolves it, no zero reset
  - Eigendecomposition cache ELIMINATED from training forward path
  - invalidate_cache() still provided for completeness

v2.0 Spec §2.1.3 / Fix F3:
  - Euler steps use live Laplacian.apply() — differentiable
  - No spectral caching in the training loop

Definition 6.1.2:
    ∂Ψ/∂t = −Δ_ℬ Ψ + 𝒥

Theorem 6.2.2 (Convergence):
    ‖Ψ(t) − Ψ_∞‖ ≤ C e^{−λ₁t}
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
from cdi.config import CDIConfig


class HeatEquation:
    """Cohomodynamic heat equation solver — v2.0 recurrent.

    In v2.0 the caller (CDIEngine) owns the recurrent state Ψ and
    calls evolve_euler() to advance it by K steps for each token.
    The state is NEVER reset between tokens within a sequence.

    Attributes
    ----------
    laplacian : BeliefLaplacian
    config : CDIConfig
    """

    def __init__(self, laplacian, config: CDIConfig) -> None:
        self.laplacian = laplacian
        self.config = config

    # ------------------------------------------------------------------
    # Euler integration — v2.0: differentiable, uses live Laplacian
    # ------------------------------------------------------------------

    def evolve_euler(
        self,
        psi: torch.Tensor,
        J: torch.Tensor,
        dt: float,
        steps: int,
    ) -> torch.Tensor:
        """Explicit Euler: Ψ_{k+1} = Ψ_k − dt·Δ_ℬ Ψ_k + dt·𝒥.

        v2.0 changes:
          - psi is the CURRENT state (not torch.zeros)
          - Laplacian.apply() uses the live matrix (no detach)
          - Each step creates a fresh computation node (no in-place)

        Parameters
        ----------
        psi   : (N,) — current belief state (persisted across tokens)
        J     : (N,) — observation source term for this token
        dt    : float
        steps : int  (= K from spec)

        Returns (N,) — evolved belief state.
        """
        current = psi
        for _ in range(steps):
            lap_term = self.laplacian.apply(current)  # live Δ_ℬ, in graph
            current = current - dt * lap_term + dt * J
        return current

    # ------------------------------------------------------------------
    # Steady state — for monitoring / PCG diagnostics
    # ------------------------------------------------------------------

    def steady_state(self, J: torch.Tensor) -> torch.Tensor:
        """Ψ_∞ = Δ_ℬ⁻¹ 𝒥 on (ker Δ_ℬ)⊥ via eigendecomposition.

        Used for monitoring convergence, not in training forward pass.
        """
        evals, evecs = self.laplacian.eigendecompose()
        J_coeffs = evecs.T @ J
        result_coeffs = torch.zeros_like(J_coeffs)
        nonharm = evals.abs() > 1e-10
        result_coeffs[nonharm] = J_coeffs[nonharm] / evals[nonharm]
        return evecs @ result_coeffs

    # ------------------------------------------------------------------
    # Spectral solution — diagnostics only
    # ------------------------------------------------------------------

    def evolve_spectral(self, psi_0: torch.Tensor, J: torch.Tensor, t: float) -> torch.Tensor:
        """Exact spectral solution at time t. DIAGNOSTICS ONLY."""
        evals, evecs = self.laplacian.eigendecompose()
        c = evecs.T @ psi_0
        J_coeffs = evecs.T @ J
        decay = torch.exp(-evals * t)
        harmonic = evals.abs() < 1e-10
        nonharm = ~harmonic
        result_coeffs = torch.zeros_like(c)
        result_coeffs[harmonic] = c[harmonic] + J_coeffs[harmonic] * t
        if nonharm.any():
            lam = evals[nonharm]
            result_coeffs[nonharm] = (
                c[nonharm] * decay[nonharm]
                + (J_coeffs[nonharm] / lam) * (1.0 - decay[nonharm])
            )
        return evecs @ result_coeffs

    # ------------------------------------------------------------------
    # Convergence diagnostics — use Laplacian.spectral_gap() directly
    # ------------------------------------------------------------------

    def convergence_rate(self) -> torch.Tensor:
        """λ₁ from Laplacian diagnostics."""
        return self.laplacian.spectral_gap()

    def learning_time(self) -> torch.Tensor:
        """τ = 1/λ₁."""
        lam1 = self.convergence_rate()
        if lam1.abs() < 1e-12:
            return torch.tensor(float("inf"), dtype=self.config.dtype)
        return 1.0 / lam1

    def convergence_bound(
        self, psi: torch.Tensor, psi_inf: torch.Tensor, t: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Verify ‖Ψ(t) − Ψ_∞‖ ≤ C e^{−λ₁t}."""
        actual = torch.norm(psi - psi_inf)
        C = actual.clone()
        lam1 = self.convergence_rate()
        bound = C * torch.exp(-lam1 * t)
        return actual, bound

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def invalidate_cache(self) -> None:
        """No-op in v2.0 (no cache in heat equation itself).
        Laplacian spectral cache is managed by BeliefLaplacian.invalidate().
        """
        pass
