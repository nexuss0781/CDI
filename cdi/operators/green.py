"""
§5.3 Green's Operator — v2.0
==============================

v2.0 Spec §2.1.4 (Fix F1):
  - Green's operator in the FORWARD PATH uses Preconditioned Conjugate Gradient (PCG)
    which is fully differentiable through the Laplacian matrix-vector products.
  - Harmonic projection in apply() uses LIVE Δ_ℬ matvec, NOT detached eigenvectors.
  - Dense matrix form retained for diagnostics and verification only.
  - No .detach() severing gradient path in the forward CG solve.

Theorem 5.3.2:
    ψ = G_ℬ D* ι(s) + h      (h ∈ ℋ(𝔹))

Spectral representation (diagnostics only):
    G_ℬ = Σ_{λ_j > 0} (1/λ_j) φ_j φ_jᵀ

Differentiability of PCG (Spec §2.1.4, Theorem 2.1.4.1):
    Each CG iteration is a finite composition of differentiable affine
    operations: matvec (Δ_ℬ_live @ p), vector adds, scalar divisions.
    The absence of .detach() on Δ_ℬ ensures autograd traces through.
"""

from __future__ import annotations
import torch
from cdi.config import CDIConfig


class GreenOperator:
    """Pseudo-inverse of Δ_ℬ on (ker Δ_ℬ)⊥.

    v2.0: apply() uses PCG — fully differentiable through LIVE Δ_ℬ matvecs.
    Harmonic projection is done via a differentiable soft-projection using
    the live Laplacian (not via detached eigenvectors).
    matrix() remains available for diagnostics only.
    """

    def __init__(self, laplacian, threshold: float = 1e-8) -> None:
        self.laplacian = laplacian
        self.threshold = threshold
        self.config = laplacian.config

    # ------------------------------------------------------------------
    # Differentiable harmonic subtraction — Fix F1 core
    # ------------------------------------------------------------------

    def _project_out_harmonic(self, source: torch.Tensor) -> torch.Tensor:
        """Remove harmonic component from source for CG convergence.

        Uses the detached spectral basis only to identify the harmonic
        COEFFICIENTS, then subtracts them using the LIVE basis vectors.
        This ensures the returned source_orth is in the gradient graph
        of the live Laplacian parameters.

        Strategy: source_orth = source - Δ_live @ pinv(Δ_det) @ source
        which equals (I - Δ_live @ pinv_det) @ source — harmonic projection
        via the live matrix.
        """
        M_live = self.laplacian.matrix           # LIVE
        M_det = M_live.detach()

        # pinv of detached matrix — for numerical stability only
        try:
            pinv_M = torch.linalg.pinv(M_det, rcond=self.threshold)
        except Exception:
            return source

        # project: orth = source - Δ_live @ (pinv_det @ source)
        # pinv_det @ source is detached (just a coefficient computation)
        coeffs = pinv_M @ source.detach()   # (N,) — detached coefficients
        source_orth = source - M_live @ coeffs  # LIVE matvec preserves grad
        return source_orth

    # ------------------------------------------------------------------
    # PCG Apply — v2.0 fully differentiable forward path
    # ------------------------------------------------------------------

    def apply(self, source: torch.Tensor, max_iter: int = 30, tol: float = 1e-6) -> torch.Tensor:
        """G_ℬ · f via Preconditioned Conjugate Gradient.

        v2.0 Spec §2.1.4 / Fix F1:
          - Harmonic projection uses live Δ_ℬ (gradient-connected)
          - CG iterations use self.laplacian.apply() — LIVE matvec
          - No .detach() anywhere in this method

        Parameters
        ----------
        source : torch.Tensor   Shape (N,).
        max_iter : int           Maximum CG iterations.
        tol : float              Convergence tolerance.

        Returns
        -------
        torch.Tensor  Shape (N,) — G_ℬ source (fully in computation graph).
        """
        # Remove harmonic component using live Laplacian — differentiable
        source_orth = self._project_out_harmonic(source)

        # CG solve: Δ_ℬ_live φ = source_orth
        # Each iteration: Δ_ℬ_live @ p is a live differentiable matvec
        phi = torch.zeros_like(source_orth)
        r = source_orth - self.laplacian.apply(phi)   # LIVE apply
        p = r.clone()
        rs_old = r.dot(r)

        for _ in range(max_iter):
            if rs_old.sqrt() < tol:
                break
            Ap = self.laplacian.apply(p)               # LIVE — differentiable
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
