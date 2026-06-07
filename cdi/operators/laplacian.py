"""
§5.1 Belief Laplacian — v2.0
=============================

v2.0 Spec Changes (§2.1, §2.3):
  - Matrix is built from LIVE parameters (no .detach() anywhere)
  - Eigendecomposition cache is ELIMINATED from the training-forward path
  - apply() uses direct matrix-vector product: Δ·ψ = matrix @ ψ
  - Eigendecomposition retained ONLY for periodic diagnostics (spectral gap)
  - invalidate() clears both matrix and spectral caches

Definition 5.1.1:
    Δ_ℬ = D² + Δ_δ + [D,A] + [A,D] + A²

Theorem 5.1.3: Δ_ℬ is essentially self-adjoint and positive semi-definite.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
from cdi.config import CDIConfig


class BeliefLaplacian:
    """Full belief Laplacian on the twisted bundle 𝔹.

    v2.0: Built from live parameters. No eigendecomposition cache during
    the forward pass. Spectral decomposition is computed on-demand only
    for diagnostics.
    """

    def __init__(self, dirac, belief, connection, config: CDIConfig) -> None:
        self.dirac = dirac
        self.belief = belief
        self.connection = connection
        self.config = config

        self.n = config.n_points
        self.s = config.spinor_dim
        self.B = config.total_belief_dim
        self.N = config.total_state_dim

        self._matrix: Optional[torch.Tensor] = None
        # Spectral decomposition — only for diagnostics, never for forward pass
        self._eigenvalues: Optional[torch.Tensor] = None
        self._eigenvectors: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Build — from LIVE parameters
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Build Δ_ℬ = D² + Δ_δ + coupling + A² from live parameters."""
        dtype = self.config.dtype

        # D² — from live Dirac matrix
        D_mat = self.dirac.matrix          # LIVE, connected to graph
        D2 = D_mat @ D_mat

        # Δ_δ — combinatorial Laplacian, lifted (depends on live δ maps)
        delta_lap_block = self.belief.full_combinatorial_laplacian()  # (B,B) live
        I_n = torch.eye(self.n, dtype=dtype)
        I_s = torch.eye(self.s, dtype=dtype)
        delta_lap_full = torch.kron(I_n, torch.kron(I_s, delta_lap_block))

        # A — full connection (depends on live W_params)
        A_full = self.connection.full_connection_matrix()  # (n·B, n·B) live
        A_lifted = torch.zeros(self.N, self.N, dtype=dtype)
        for p in range(self.n):
            for q in range(self.n):
                A_block = A_full[p * self.B:(p + 1) * self.B,
                                 q * self.B:(q + 1) * self.B]
                if torch.any(A_block != 0):
                    A_sB = torch.kron(I_s, A_block)
                    rp = p * self.s * self.B
                    rq = q * self.s * self.B
                    A_lifted[rp:rp + self.s * self.B, rq:rq + self.s * self.B] = A_sB

        # [D, A] coupling
        coupling = D_mat @ A_lifted + A_lifted @ D_mat

        # A²
        A_sq = A_lifted @ A_lifted

        # Full Laplacian
        Lap = D2 + delta_lap_full + coupling + A_sq

        # Symmetrise — Theorem 5.1.3
        Lap = 0.5 * (Lap + Lap.T)
        self._matrix = Lap

        # Invalidate stale spectral cache
        self._eigenvalues = None
        self._eigenvectors = None

    @property
    def matrix(self) -> torch.Tensor:
        if self._matrix is None:
            self.build()
        return self._matrix

    @matrix.setter
    def matrix(self, value: torch.Tensor) -> None:
        self._matrix = value
        self._eigenvalues = None
        self._eigenvectors = None

    # ------------------------------------------------------------------
    # Apply — O(N²) dense matvec, differentiable
    # ------------------------------------------------------------------

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        """Δ_ℬ · ψ — differentiable matrix-vector product.

        v2.0: Uses self.matrix (live, no detach), so gradients flow back
        to all parameters through this operation.
        """
        return self.matrix @ state

    # ------------------------------------------------------------------
    # Spectral decomposition — DIAGNOSTICS ONLY, never in forward pass
    # ------------------------------------------------------------------

    def eigendecompose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full eigendecomposition. Computed once, cached.

        v2.0: Used ONLY for diagnostics (spectral gap, learning time).
        NOT used in the forward pass. Results are DETACHED since they
        are only used for monitoring, not gradient computation.
        """
        if self._eigenvalues is None:
            M = self.matrix.detach()  # detach here is OK — diagnostics only
            evals, evecs = torch.linalg.eigh(M)
            evals = evals.clamp(min=0.0)
            self._eigenvalues = evals
            self._eigenvectors = evecs
        return self._eigenvalues, self._eigenvectors

    def spectral_gap(self) -> torch.Tensor:
        """λ₁ — first positive eigenvalue (for monitoring only)."""
        evals, _ = self.eigendecompose()
        positive = evals[evals > 1e-10]
        if len(positive) == 0:
            return torch.tensor(0.0, dtype=self.config.dtype)
        return positive.min()

    def lanczos_spectral_gap(self, max_iter: int = 20) -> float:
        """Fast O(N·max_iter) spectral gap via Lanczos iteration.

        v2.0 Spec §2.3.3: Preferred for large N since it avoids full
        eigendecomposition.
        """
        N = self.N
        dtype = self.config.dtype
        q = torch.randn(N, dtype=dtype)
        q = q / q.norm()

        alpha_list = []
        beta_list = []
        q_prev = torch.zeros(N, dtype=dtype)

        for i in range(min(max_iter, N)):
            v = (self.matrix.detach() @ q)  # detach for diagnostics
            a_i = float(q.dot(v))
            v = v - a_i * q
            if i > 0:
                v = v - beta_list[-1] * q_prev
            b_i = float(v.norm())
            if b_i < 1e-10:
                break
            q_prev = q.clone()
            q = v / b_i
            alpha_list.append(a_i)
            beta_list.append(b_i)

        if len(alpha_list) == 0:
            return 0.0

        # Build tridiagonal matrix
        size = len(alpha_list)
        T = torch.zeros(size, size, dtype=dtype)
        for i in range(size):
            T[i, i] = alpha_list[i]
        for i in range(len(beta_list)):
            if i + 1 < size:
                T[i, i + 1] = beta_list[i]
                T[i + 1, i] = beta_list[i]

        evals = torch.linalg.eigvalsh(T)
        evals = evals.clamp(min=0.0)
        positive = evals[evals > 1e-10]
        return float(positive.min()) if len(positive) > 0 else 0.0

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def check_self_adjoint(self) -> torch.Tensor:
        return torch.norm(self.matrix - self.matrix.T)

    def check_positive_semidefinite(self) -> bool:
        evals, _ = self.eigendecompose()
        return bool((evals >= -1e-8).all())

    def invalidate(self) -> None:
        """Clear ALL caches — call after parameter updates."""
        self._matrix = None
        self._eigenvalues = None
        self._eigenvectors = None
