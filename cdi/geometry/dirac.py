"""
§4.2 Cognitive Dirac Operator
=============================

Implements the cognitive Dirac operator D from CDI Specification §4.2.

Definition 4.2.2:
    D = Σᵢ c(eⁱ) ∇^𝔹_{eᵢ}

where {eᵢ} is a local orthonormal frame, c(eⁱ) is the Clifford action,
and ∇^𝔹 is the Clifford connection on the twisted bundle 𝔹 = S ⊗ ⊕_k ℬ_k.

Theorem 4.2.3: D is first-order elliptic with symbol σ(D)(ξ) = c(ξ).
"""

from __future__ import annotations

from typing import Optional

import torch

from cdi.config import CDIConfig


class DiracOperator:
    """Discretised cognitive Dirac operator on 𝔹 = S ⊗ ⊕_k ℬ_k.

    The state Ψ lives in ℝ^{n × s × B} (flattened to ℝ^{n·s·B}).
    D is represented as a dense (n·s·B, n·s·B) matrix.

    For each edge (p, q) in the cover, the Dirac contribution is:

        D_{p←q} = [Σᵢ vⁱ_{pq} γⁱ(p) / |pq|] ⊗ [I_B + A_{pq}]

    where v_{pq} is the normalised direction from q to p.

    Attributes
    ----------
    config : CDIConfig
    _matrix : torch.Tensor or None
        Dense Dirac matrix, shape ``(N, N)`` with N = total_state_dim.
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
        self.N = config.total_state_dim  # n * s * B

        self._matrix: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Construct the dense Dirac matrix."""
        dtype = self.config.dtype
        D = torch.zeros(self.N, self.N, dtype=dtype)

        pts = self.manifold.points.detach()  # (n, d)
        frames = self.manifold.orthonormal_frame().detach()  # (n, d, d)
        I_B = torch.eye(self.B, dtype=dtype)

        for p, q in self.cover.edges:
            # Direction and distance
            diff = pts[p] - pts[q]  # (d,)
            dist = torch.norm(diff).clamp(min=1e-10)
            v = diff / dist  # normalised direction p←q

            # Clifford part: Σᵢ vⁱ γⁱ(p) / |pq|
            # Use curved gammas at point p
            curved_gammas = self.clifford.gamma_at_point(frames[p])
            cliff_block = torch.zeros(self.s, self.s, dtype=dtype)
            for i in range(self.clifford.d):
                cliff_block = cliff_block + v[i] * curved_gammas[i]
            cliff_block = cliff_block / dist

            # Belief part: I_B + A_{pq}
            A_pq = self.connection.connection_on_edge(p, q)  # (B, B)
            belief_block = I_B + A_pq

            # Kronecker: (s, s) ⊗ (B, B) → (sB, sB)
            full_block = torch.kron(cliff_block, belief_block)

            # Place off-diagonal block: p ← q
            rp = p * self.sB
            rq = q * self.sB
            D[rp : rp + self.sB, rq : rq + self.sB] += full_block

            # Symmetric block: q ← p
            diff_qp = pts[q] - pts[p]
            v_qp = diff_qp / dist
            cliff_block_qp = torch.zeros(self.s, self.s, dtype=dtype)
            curved_gammas_q = self.clifford.gamma_at_point(frames[q])
            for i in range(self.clifford.d):
                cliff_block_qp = cliff_block_qp + v_qp[i] * curved_gammas_q[i]
            cliff_block_qp = cliff_block_qp / dist

            A_qp = self.connection.connection_on_edge(q, p)
            belief_block_qp = I_B + A_qp
            full_block_qp = torch.kron(cliff_block_qp, belief_block_qp)
            D[rq : rq + self.sB, rp : rp + self.sB] += full_block_qp

        # Symmetrise: D should be self-adjoint (Theorem 4.2.3 discussion)
        D = 0.5 * (D + D.T)
        self._matrix = D

    def matrix(self) -> torch.Tensor:
        """Return the dense Dirac matrix. Builds if not yet constructed.
        
        Call invalidate() to force a rebuild after parameter updates.
        """
        if self._matrix is None:
            self.build()
        return self._matrix

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        """D·ψ.

        Parameters
        ----------
        state : torch.Tensor
            Shape ``(N,)`` — flattened state vector.

        Returns
        -------
        torch.Tensor
            Shape ``(N,)``.
        """
        return self.matrix() @ state

    def apply_adjoint(self, state: torch.Tensor) -> torch.Tensor:
        """D*·ψ.  Since D is self-adjoint, D* = D."""
        return self.apply(state)

    def squared(self) -> torch.Tensor:
        """D² — Lichnerowicz: D² = ∇*∇ + R/4.

        Returns
        -------
        torch.Tensor
            Shape ``(N, N)``.
        """
        M = self.matrix()
        return M @ M

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def check_self_adjoint(self) -> torch.Tensor:
        """‖D − Dᵀ‖_F — should be ≈ 0."""
        M = self.matrix()
        return torch.norm(M - M.T)

    def invalidate(self) -> None:
        """Clear cached matrix (call after parameter updates)."""
        self._matrix = None
