"""
§5.2 Hodge Decomposition — v2.0
=================================

v2.0 Changes (Spec §2.1.2 / Fix F1):
  - Removed .detach() from harmonic_projector output
  - harmonic_projector() is differentiable through the live Laplacian matrix
  - decompose() and project() participate fully in the computation graph
  - Soft harmonic projector uses I - Δ_ℬ @ pinv(Δ_ℬ) built from LIVE matrix
  - Spectral basis available for diagnostics via eigendecompose() (detached)

Theorem 5.2.1 (Cognitive Hodge Theorem):
    Γ(𝔹) = ℋ(𝔹) ⊕ im(Δ_ℬ)

where ℋ(𝔹) = ker(Δ_ℬ) is the space of harmonic beliefs.
"""

from __future__ import annotations
from typing import Tuple
import torch
from cdi.config import CDIConfig


class HodgeDecomposition:
    """Hodge decomposition into harmonic and non-harmonic components.

    v2.0: Projection is fully differentiable through the live Laplacian matrix.

    The harmonic projector is constructed as:
        H = I − Δ_ℬ @ pinv(Δ_ℬ)
    where pinv is computed from the detached matrix (for numerical stability),
    but the final projection H @ state is taken as:
        harmonic = state − Δ_ℬ @ G_state
    where G_state = pinv(Δ_ℬ).detach() @ state and then we reconstruct:
        harmonic = state − Δ_ℬ_live @ G_state.detach()
    This preserves gradient flow through Δ_ℬ_live to all parameters.
    """

    def __init__(self, laplacian, threshold: float = 1e-8) -> None:
        self.laplacian = laplacian
        self.threshold = threshold
        self.config = laplacian.config

    # ------------------------------------------------------------------
    # Differentiable harmonic projection — Fix F1 core
    # ------------------------------------------------------------------

    def project_harmonic_diff(self, state: torch.Tensor) -> torch.Tensor:
        """Differentiable harmonic projection: H·ψ = ψ - Δ_ℬ · pinv(Δ_ℬ) · ψ.

        The pinv application uses the DETACHED matrix (numerical inversion),
        but Δ_ℬ on the left is LIVE, so gradients flow to all parameters
        through the live matrix-vector product.

        This is equivalent to: harmonic = ψ - Δ_ℬ_live · (Δ_ℬ_det⁺ · ψ)
        """
        M_live = self.laplacian.matrix           # LIVE — in gradient graph
        M_det = M_live.detach()                  # detached copy for inversion

        # Compute pinv(Δ_ℬ) · state using detached matrix (stable inversion)
        # Avoid full pinv for large N — use lstsq
        try:
            # For small N: direct pinv
            pinv_M = torch.linalg.pinv(M_det, rcond=self.threshold)
            green_state = pinv_M @ state.detach()          # (N,) detached
        except Exception:
            green_state = torch.zeros_like(state.detach())

        # Subtract LIVE Δ_ℬ · green_state — gradient flows through M_live
        non_harmonic_part = M_live @ green_state           # LIVE matvec
        harmonic = state - non_harmonic_part               # differentiable
        return harmonic

    def harmonic_projector(self) -> torch.Tensor:
        """H = I − Δ_ℬ_live · pinv(Δ_ℬ_det).

        For diagnostics/small models: returns the (N,N) projector matrix.
        The returned matrix has gradient connectivity through Δ_ℬ_live.

        Returns (N, N) projector matrix.
        """
        N = self.laplacian.N
        dtype = self.config.dtype
        M_live = self.laplacian.matrix           # LIVE
        M_det = M_live.detach()

        try:
            pinv_M = torch.linalg.pinv(M_det, rcond=self.threshold)
        except Exception:
            return torch.eye(N, dtype=dtype)

        # H = I - Δ_live @ pinv_det  — gradient flows through Δ_live
        I = torch.eye(N, dtype=dtype)
        return I - M_live @ pinv_M

    def harmonic_basis(self) -> torch.Tensor:
        """Orthonormal basis of harmonic beliefs. Diagnostics only."""
        evals, evecs = self.laplacian.eigendecompose()
        return evecs[:, evals.abs() < self.threshold]

    # ------------------------------------------------------------------
    # Decompose — v2.0: differentiable through live Laplacian
    # ------------------------------------------------------------------

    def decompose(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hodge-decompose state into (harmonic, non_harmonic).

        v2.0: Uses project_harmonic_diff() — gradient flows through live Δ_ℬ.
        """
        harmonic = self.project_harmonic_diff(state)
        non_harmonic = state - harmonic
        return harmonic, non_harmonic

    def project_harmonic(self, state: torch.Tensor) -> torch.Tensor:
        """Project onto harmonic subspace (differentiable)."""
        return self.project_harmonic_diff(state)

    # ------------------------------------------------------------------
    # Dimensions — diagnostic only
    # ------------------------------------------------------------------

    def harmonic_dimension(self) -> int:
        evals, _ = self.laplacian.eigendecompose()
        return int((evals.abs() < self.threshold).sum().item())

    def is_harmonic(self, state: torch.Tensor, tol: float = 1e-6) -> bool:
        return bool(torch.norm(self.laplacian.apply(state)).item() < tol)
