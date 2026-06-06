"""
§6.2 Spectral Decomposition Utilities
======================================

Provides the heat semigroup e^{−tΔ_ℬ}, Duhamel integral,
spectral entropy, and the Cheeger-type bound from §11.2.
"""

from __future__ import annotations

import torch

from cdi.config import CDIConfig


class SpectralDecomposition:
    """Spectral tools built on top of the Laplacian eigendecomposition.

    Attributes
    ----------
    laplacian : BeliefLaplacian
    config : CDIConfig
    """

    def __init__(self, laplacian, config: CDIConfig) -> None:
        self.laplacian = laplacian
        self.config = config

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def decompose(self):
        """Full eigendecomposition (eigenvalues, eigenvectors)."""
        return self.laplacian.eigendecompose()

    # ------------------------------------------------------------------
    # Heat semigroup
    # ------------------------------------------------------------------

    def heat_semigroup(self, t: float) -> torch.Tensor:
        """e^{−tΔ_ℬ} = Σⱼ e^{−λⱼt} φⱼ φⱼᵀ.

        Returns
        -------
        torch.Tensor  Shape ``(N, N)``.
        """
        evals, evecs = self.decompose()
        decay = torch.exp(-evals * t)
        return evecs @ torch.diag(decay) @ evecs.T

    def duhamel_integral(self, J: torch.Tensor, t: float) -> torch.Tensor:
        """∫₀ᵗ e^{−(t−s)Δ_ℬ} 𝒥 ds.

        Spectral: Σⱼ (𝒥ⱼ/λⱼ)(1 − e^{−λⱼt}) φⱼ  for λⱼ > 0.

        Parameters
        ----------
        J : torch.Tensor  Shape ``(N,)``.
        t : float

        Returns
        -------
        torch.Tensor  Shape ``(N,)``.
        """
        evals, evecs = self.decompose()
        J_coeffs = evecs.T @ J

        result = torch.zeros_like(J_coeffs)
        harmonic = evals.abs() < 1e-10
        nonharm = ~harmonic

        result[harmonic] = J_coeffs[harmonic] * t
        if nonharm.any():
            lam = evals[nonharm]
            decay = torch.exp(-lam * t)
            result[nonharm] = (J_coeffs[nonharm] / lam) * (1.0 - decay)

        return evecs @ result

    # ------------------------------------------------------------------
    # Spectral statistics
    # ------------------------------------------------------------------

    def spectral_entropy(self) -> torch.Tensor:
        """−Σⱼ pⱼ log pⱼ  where pⱼ = λⱼ / Σλⱼ  (positive eigenvalues)."""
        evals, _ = self.decompose()
        positive = evals[evals > 1e-10]
        if len(positive) == 0:
            return torch.tensor(0.0, dtype=self.config.dtype)
        p = positive / positive.sum()
        return -(p * torch.log(p + 1e-20)).sum()

    def effective_dimension(self) -> torch.Tensor:
        """exp(spectral_entropy) — effective degrees of freedom."""
        return torch.exp(self.spectral_entropy())

    def condition_number(self) -> torch.Tensor:
        """λ_max / λ_min (over positive eigenvalues)."""
        evals, _ = self.decompose()
        positive = evals[evals > 1e-10]
        if len(positive) < 2:
            return torch.tensor(1.0, dtype=self.config.dtype)
        return positive.max() / positive.min()

    def cheeger_bound(self, kappa: float, d: int) -> torch.Tensor:
        """Theorem 11.2.2: λ₁ ≥ d·κ/(d−1) for Ricci ≥ κ > 0.

        Parameters
        ----------
        kappa : float  Lower bound on Ricci curvature.
        d : int        Manifold dimension.

        Returns
        -------
        torch.Tensor  Theoretical lower bound on λ₁.
        """
        if d <= 1 or kappa <= 0:
            return torch.tensor(0.0, dtype=self.config.dtype)
        return torch.tensor(d * kappa / (d - 1), dtype=self.config.dtype)
