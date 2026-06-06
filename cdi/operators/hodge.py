"""
§5.2 Hodge Decomposition
========================

Implements the Cognitive Hodge Theorem from CDI Specification §5.2.

Theorem 5.2.1:
    Γ(𝔹) = ℋ(𝔹) ⊕ im(Δ_ℬ)

where ℋ(𝔹) = ker(Δ_ℬ) is the space of *harmonic beliefs* and

    ker Δ_ℬ ≅ ℍ^•(M, ℬ^•)

(harmonic beliefs ↔ intelligence cohomology).
"""

from __future__ import annotations

from typing import Tuple

import torch

from cdi.config import CDIConfig


class HodgeDecomposition:
    """Hodge decomposition into harmonic and non-harmonic components.

    Uses the spectral decomposition of Δ_ℬ: eigenvalues near zero
    (|λ| < threshold) span the harmonic space ℋ(𝔹).

    Attributes
    ----------
    laplacian : BeliefLaplacian
    threshold : float
        Eigenvalue threshold for identifying harmonic modes.
    """

    def __init__(self, laplacian, threshold: float = 1e-8) -> None:
        self.laplacian = laplacian
        self.threshold = threshold
        self.config = laplacian.config

    # ------------------------------------------------------------------
    # Projectors
    # ------------------------------------------------------------------

    def harmonic_projector(self) -> torch.Tensor:
        """Harmonic projector H = Σ_{λ_j ≈ 0} φ_j φ_jᵀ.

        Returns
        -------
        torch.Tensor
            Shape ``(N, N)``.
        """
        evals, evecs = self.laplacian.eigendecompose()
        harmonic_mask = evals.abs() < self.threshold
        if not harmonic_mask.any():
            return torch.zeros(
                self.laplacian.N, self.laplacian.N, dtype=self.config.dtype
            )
        H_vecs = evecs[:, harmonic_mask]  # (N, n_harmonic)
        return H_vecs @ H_vecs.T

    def harmonic_basis(self) -> torch.Tensor:
        """Orthonormal basis of harmonic beliefs.

        Returns
        -------
        torch.Tensor
            Shape ``(N, n_harmonic)`` — columns are harmonic eigenvectors.
        """
        evals, evecs = self.laplacian.eigendecompose()
        harmonic_mask = evals.abs() < self.threshold
        return evecs[:, harmonic_mask]

    # ------------------------------------------------------------------
    # Decompose
    # ------------------------------------------------------------------

    def decompose(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hodge-decompose a state into harmonic and non-harmonic parts.

        Parameters
        ----------
        state : torch.Tensor
            Shape ``(N,)``.

        Returns
        -------
        (harmonic, non_harmonic) : tuple[Tensor, Tensor]
            Both shape ``(N,)``, with state = harmonic + non_harmonic.
        """
        H = self.harmonic_projector()
        harmonic = H @ state
        non_harmonic = state - harmonic
        return harmonic, non_harmonic

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    def harmonic_dimension(self) -> int:
        """Number of harmonic modes = dim ℋ(𝔹) = dim ℍ^•(M, ℬ^•)."""
        evals, _ = self.laplacian.eigendecompose()
        return int((evals.abs() < self.threshold).sum().item())

    def is_harmonic(self, state: torch.Tensor, tol: float = 1e-6) -> bool:
        """Check if Δ_ℬ ψ ≈ 0."""
        return bool(torch.norm(self.laplacian.apply(state)).item() < tol)
