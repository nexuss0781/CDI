"""
§5.3 Green's Operator — v2.0
==============================

v2.0 Spec §2.1.4 (Fix F1):
  - Green's operator in the FORWARD PATH uses Preconditioned Conjugate Gradient (PCG)
    which is fully differentiable through the Laplacian matrix-vector products.
  - Dense matrix form retained for diagnostics and verification only.
  - No .detach() on Green's output in forward path.

Theorem 5.3.2:
    ψ = G_ℬ D* ι(s) + h      (h ∈ ℋ(𝔹))

Spectral representation (diagnostics only):
    G_ℬ = Σ_{λ_j > 0} (1/λ_j) φ_j φ_jᵀ
"""

from __future__ import annotations
import torch
from cdi.config import CDIConfig


class GreenOperator:
    """Pseudo-inverse of Δ_ℬ on (ker Δ_ℬ)⊥.

    v2.0: apply() uses PCG — differentiable through matvec calls.
    matrix() remains available for diagnostics.
    """

    def __init__(self, laplacian, threshold: float = 1e-8) -> None:
        self.laplacian = laplacian
        self.threshold = threshold
        self.config = laplacian.config

    # ------------------------------------------------------------------
    # PCG Apply — v2.0 differentiable forward path
    # ------------------------------------------------------------------

    def apply(self, source: torch.Tensor, max_iter: int = 30, tol: float = 1e-6) -> torch.Tensor:
        """G_ℬ · f via Preconditioned Conjugate Gradient.

        v2.0 Spec §2.1.4: PCG is fully differentiable through the
        Laplacian matvec calls (self.laplacian.apply). Each CG iteration
        is a composition of differentiable operations.

        Parameters
        ----------
        source : torch.Tensor   Shape (N,).
        max_iter : int           Maximum CG iterations.
        tol : float              Convergence tolerance.

        Returns
        -------
        torch.Tensor  Shape (N,) — G_ℬ source.
        """
        # Project source onto (ker Δ_ℬ)⊥ by removing harmonic component
        # Use spectral projector to identify and subtract harmonic part
        evals, evecs = self.laplacian.eigendecompose()
        harmonic_mask = evals.abs() < self.threshold
        if harmonic_mask.any():
            H_vecs = evecs[:, harmonic_mask]          # (N, h)
            harm_coeff = H_vecs.T @ source            # (h,)
            source_orth = source - H_vecs @ harm_coeff
        else:
            source_orth = source

        # CG solve: Δ_ℬ φ = source_orth
        phi = torch.zeros_like(source_orth)
        r = source_orth - self.laplacian.apply(phi)
        p = r.clone()
        rs_old = r.dot(r)

        for _ in range(max_iter):
            if rs_old.sqrt() < tol:
                break
            Ap = self.laplacian.apply(p)
            denom = p.dot(Ap).clamp(min=1e-20)
            alpha = rs_old / denom
            phi = phi + alpha * p
            r = r - alpha * Ap
            rs_new = r.dot(r)
            if rs_new.sqrt() < tol:
                break
            beta = rs_new / rs_old
            p = r + beta * p
            rs_old = rs_new

        return phi

    # ------------------------------------------------------------------
    # Dense matrix — DIAGNOSTICS ONLY
    # ------------------------------------------------------------------

    def matrix(self) -> torch.Tensor:
        """Dense G_ℬ = Σ_{λ>0} (1/λ) φ φᵀ. For diagnostics only."""
        evals, evecs = self.laplacian.eigendecompose()
        inv_evals = torch.zeros_like(evals)
        nonharm = evals.abs() > self.threshold
        inv_evals[nonharm] = 1.0 / evals[nonharm]
        return evecs @ torch.diag(inv_evals) @ evecs.T

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self) -> torch.Tensor:
        """‖G_ℬ Δ_ℬ + H − I‖_F ≈ 0."""
        G = self.matrix()
        Lap = self.laplacian.matrix
        evals, evecs = self.laplacian.eigendecompose()
        harmonic_mask = evals.abs() < self.threshold
        H_vecs = evecs[:, harmonic_mask]
        H = H_vecs @ H_vecs.T if harmonic_mask.any() else torch.zeros_like(G)
        I = torch.eye(G.shape[0], dtype=self.config.dtype)
        return torch.norm(G @ Lap.detach() + H - I)
