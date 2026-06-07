"""
§4.2 Cognitive Dirac Operator — v2.0
======================================

v2.0 Fix F3 (Spec §2.3.2 / §7.3):
  - REMOVED .detach() on manifold points and frames
  - Dirac matrix is built from LIVE parameters so autograd traces through it
  - invalidate() still provided for explicit cache clearing after step
  - build() called by rebuild_operators() which is called after every step

Definition 4.2.2:
    D = Σᵢ c(eⁱ) ∇^𝔹_{eᵢ}

Theorem 4.2.3: D is first-order elliptic; D is self-adjoint.
"""

from __future__ import annotations
from typing import Optional
import torch
from cdi.config import CDIConfig


class DiracOperator:
    """Discretised cognitive Dirac operator on 𝔹 = S ⊗ ⊕_k ℬ_k.

    v2.0: Built WITHOUT .detach() so gradients flow through the matrix
    back to manifold.points, manifold.metric_L, and connection.W_params.

    Attributes
    ----------
    _matrix : torch.Tensor | None
        Dense (N,N) Dirac matrix. None until build() is called.
    """

    def __init__(self, manifold, clifford, connection, belief, cover, config: CDIConfig) -> None:
        self.manifold = manifold
        self.clifford = clifford
        self.connection = connection
        self.belief = belief
        self.cover = cover
        self.config = config

        self.n = config.n_points
        self.s = config.spinor_dim
        self.B = config.total_belief_dim
        self.sB = self.s * self.B
        self.N = config.total_state_dim

        self._matrix: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Build — NO .detach() — v2.0 critical fix
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Build the Dirac matrix from live (un-detached) parameters.

        v2.0 Spec §7.3: Uses self.manifold.points (live) and
        self.manifold.orthonormal_frame() (live, computed from metric_L).
        This ensures gradients flow back to the manifold parameters.
        """
        dtype = self.config.dtype
        D = torch.zeros(self.N, self.N, dtype=dtype)

        # v2.0: NO .detach() — live tensors for gradient connectivity
        pts = self.manifold.points          # (n, d) — LIVE
        frames = self.manifold.orthonormal_frame()  # (n, d, d) — LIVE
        I_B = torch.eye(self.B, dtype=dtype)

        for p, q in self.cover.edges:
            # Direction from q to p using live points
            diff = pts[p] - pts[q]          # differentiable w.r.t. pts
            dist = torch.norm(diff).clamp(min=1e-10)
            v = diff / dist                 # normalised direction

            # Clifford part at p — curved gammas via live frame
            curved_gammas_p = self.clifford.gamma_at_point(frames[p])
            cliff_p = torch.zeros(self.s, self.s, dtype=dtype)
            for i in range(self.clifford.d):
                cliff_p = cliff_p + v[i] * curved_gammas_p[i]
            cliff_p = cliff_p / dist

            # Belief transport: I_B + A_{pq}  (A from live W_params)
            A_pq = self.connection.connection_on_edge(p, q)  # (B,B) live
            belief_block = I_B + A_pq

            full_block = torch.kron(cliff_p, belief_block)

            rp, rq = p * self.sB, q * self.sB
            D[rp:rp + self.sB, rq:rq + self.sB] = (
                D[rp:rp + self.sB, rq:rq + self.sB] + full_block
            )

            # Symmetric block q ← p
            diff_qp = pts[q] - pts[p]
            v_qp = diff_qp / dist
            curved_gammas_q = self.clifford.gamma_at_point(frames[q])
            cliff_q = torch.zeros(self.s, self.s, dtype=dtype)
            for i in range(self.clifford.d):
                cliff_q = cliff_q + v_qp[i] * curved_gammas_q[i]
            cliff_q = cliff_q / dist

            A_qp = self.connection.connection_on_edge(q, p)
            belief_block_qp = I_B + A_qp
            full_block_qp = torch.kron(cliff_q, belief_block_qp)
            D[rq:rq + self.sB, rp:rp + self.sB] = (
                D[rq:rq + self.sB, rp:rp + self.sB] + full_block_qp
            )

        # Symmetrise — Theorem 4.2.3: D = D*
        D = 0.5 * (D + D.T)
        self._matrix = D

    @property
    def matrix(self) -> torch.Tensor:
        if self._matrix is None:
            self.build()
        return self._matrix

    @matrix.setter
    def matrix(self, value: torch.Tensor) -> None:
        self._matrix = value

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        """D·ψ."""
        return self.matrix @ state

    def apply_adjoint(self, state: torch.Tensor) -> torch.Tensor:
        """D*·ψ = D·ψ (self-adjoint)."""
        return self.apply(state)

    def squared(self) -> torch.Tensor:
        """D²."""
        M = self.matrix
        return M @ M

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def check_self_adjoint(self) -> torch.Tensor:
        """‖D − Dᵀ‖_F — should be ≈ 0."""
        M = self.matrix
        return torch.norm(M - M.T)

    def invalidate(self) -> None:
        """Clear cached matrix. Call after parameter updates."""
        self._matrix = None
