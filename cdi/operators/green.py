"""
§5.3 Green's Operator
=====================

Implements G_ℬ — the pseudo-inverse of Δ_ℬ on (ker Δ_ℬ)⊥.

Theorem 5.3.2:
    ψ = G_ℬ D* ι(s) + h      (h ∈ ℋ(𝔹))

Spectral representation:
    G_ℬ = Σ_{λ_j > 0} (1/λ_j) φ_j φ_jᵀ
"""

from __future__ import annotations

import torch

from cdi.config import CDIConfig


class GreenOperator:
    """Pseudo-inverse of the Belief Laplacian on (ker Δ_ℬ)⊥.

    Attributes
    ----------
    laplacian : BeliefLaplacian
    threshold : float
        Eigenvalue threshold separating harmonic from non-harmonic modes.
    """

    def __init__(self, laplacian, threshold: float = 1e-8) -> None:
        self.laplacian = laplacian
        self.threshold = threshold
        self.config = laplacian.config

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def apply(self, source: torch.Tensor) -> torch.Tensor:
        """G_ℬ · f — apply Green's operator.

        Spectral: G_ℬ f = Σ_{λ_j>0} (fᵀφ_j / λ_j) φ_j

        Parameters
        ----------
        source : torch.Tensor
            Shape ``(N,)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N,)``.
        """
        evals, evecs = self.laplacian.eigendecompose()
        coeffs = evecs.T @ source  # (N,)

        result_coeffs = torch.zeros_like(coeffs)
        nonharmnic = evals.abs() > self.threshold
        result_coeffs[nonharmnic] = coeffs[nonharmnic] / evals[nonharmnic]

        return evecs @ result_coeffs

    def matrix(self) -> torch.Tensor:
        """Dense Green's matrix G_ℬ = Σ_{λ>0} (1/λ) φ φᵀ.

        Returns
        -------
        torch.Tensor
            Shape ``(N, N)``.
        """
        evals, evecs = self.laplacian.eigendecompose()

        inv_evals = torch.zeros_like(evals)
        nonharmnic = evals.abs() > self.threshold
        inv_evals[nonharmnic] = 1.0 / evals[nonharmnic]

        # G = V diag(1/λ) Vᵀ
        return evecs @ torch.diag(inv_evals) @ evecs.T

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self) -> torch.Tensor:
        """Check ‖G_ℬ Δ_ℬ + H − I‖_F ≈ 0.

        Returns
        -------
        torch.Tensor
            Frobenius-norm error (should be near zero).
        """
        G = self.matrix()
        Lap = self.laplacian.matrix()
        evals, evecs = self.laplacian.eigendecompose()

        harmonic_mask = evals.abs() < self.threshold
        H_vecs = evecs[:, harmonic_mask]
        H = H_vecs @ H_vecs.T if harmonic_mask.any() else torch.zeros_like(G)

        I = torch.eye(G.shape[0], dtype=self.config.dtype)
        return torch.norm(G @ Lap + H - I)
