"""
§7.1 Quillen Superconnection
=============================

Definition 7.1.1:
    𝔸 = D + δ + A

where D is the Dirac operator, δ is the belief differential,
and A is the belief connection.

Theorem 7.1.2: 𝔸 is a superconnection satisfying the Leibniz rule.

The Chern character ch(𝔸) = Tr_s(e^{−𝔸²}) is a closed form whose
cohomology class is a topological invariant (Theorem 7.3.2).
"""

from __future__ import annotations

from typing import Optional

import torch

from cdi.config import CDIConfig


class Superconnection:
    """Quillen superconnection 𝔸 = D + δ + A on the twisted bundle 𝔹.

    Attributes
    ----------
    dirac : DiracOperator
    belief : BeliefComplex
    connection : BeliefConnection
    config : CDIConfig
    """

    def __init__(self, dirac, belief, connection, config: CDIConfig) -> None:
        self.dirac = dirac
        self.belief = belief
        self.connection = connection
        self.config = config

        self.N = config.total_state_dim
        self.n = config.n_points
        self.s = config.spinor_dim
        self.B = config.total_belief_dim

    # ------------------------------------------------------------------
    # δ in the full 𝔹 space
    # ------------------------------------------------------------------

    def _delta_full_matrix(self) -> torch.Tensor:
        """δ lifted to the full state space: I_n ⊗ I_s ⊗ δ_block.

        Returns shape ``(N, N)``.
        """
        delta_block = self.belief.full_coboundary_matrix()  # (B, B)
        I_n = torch.eye(self.n, dtype=self.config.dtype)
        I_s = torch.eye(self.s, dtype=self.config.dtype)
        return torch.kron(I_n, torch.kron(I_s, delta_block))

    def _delta_star_full_matrix(self) -> torch.Tensor:
        """δ* in full space."""
        return self._delta_full_matrix().T

    # ------------------------------------------------------------------
    # A in the full space
    # ------------------------------------------------------------------

    def _connection_full_matrix(self) -> torch.Tensor:
        """Belief connection lifted to full space with spinor identity."""
        A_nB = self.connection.full_connection_matrix()  # (n·B, n·B)
        I_s = torch.eye(self.s, dtype=self.config.dtype)

        A_full = torch.zeros(self.N, self.N, dtype=self.config.dtype)
        for p in range(self.n):
            for q in range(self.n):
                block = A_nB[p * self.B : (p + 1) * self.B, q * self.B : (q + 1) * self.B]
                if torch.any(block != 0):
                    lifted = torch.kron(I_s, block)
                    rp = p * self.s * self.B
                    rq = q * self.s * self.B
                    A_full[rp : rp + self.s * self.B, rq : rq + self.s * self.B] = lifted
        return A_full

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def matrix(self) -> torch.Tensor:
        """Full superconnection matrix 𝔸 = D + δ + δ* + A.

        Note: we include both δ and δ* for the full decomposition
        (Theorem 7.2.2).

        Returns shape ``(N, N)``.
        """
        D = self.dirac.matrix          # property access — LIVE, not a call
        delta = self._delta_full_matrix()
        delta_star = self._delta_star_full_matrix()
        A = self._connection_full_matrix()
        return D + delta + delta_star + A

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        """𝔸·Ψ.

        Parameters
        ----------
        state : torch.Tensor  Shape ``(N,)``.

        Returns
        -------
        torch.Tensor  Shape ``(N,)``.
        """
        return self.matrix() @ state

    def squared(self) -> torch.Tensor:
        """𝔸² — curvature of the superconnection.

        Returns shape ``(N, N)``.
        """
        M = self.matrix()
        return M @ M

    # ------------------------------------------------------------------
    # Supertrace and Chern character
    # ------------------------------------------------------------------

    def supertrace(self, operator: torch.Tensor) -> torch.Tensor:
        """Tr_s(op) = Σ_k (−1)^k Tr(op|_{degree k}).

        For the full (N, N) operator acting on (n, s, B), the supertrace
        sums block traces with alternating signs by belief degree.

        Parameters
        ----------
        operator : torch.Tensor  Shape ``(N, N)``.

        Returns
        -------
        torch.Tensor  Scalar.
        """
        result = torch.tensor(0.0, dtype=self.config.dtype)
        for p in range(self.n):
            for k_idx, k in enumerate(self.belief.degrees):
                sign = (-1.0) ** k
                dim_k = self.belief.dims[k_idx]
                offset_k = self.config.belief_offset(k)
                for si in range(self.s):
                    # Index into the flattened state
                    row = p * self.s * self.B + si * self.B + offset_k
                    for di in range(dim_k):
                        result = result + sign * operator[row + di, row + di]
        return result

    def chern_character(self) -> torch.Tensor:
        """ch(𝔸) = Tr_s(e^{−𝔸²}).

        Returns
        -------
        torch.Tensor  Scalar.
        """
        A2 = self.squared()
        # Use matrix exponential (expensive but exact)
        exp_A2 = torch.matrix_exp(-A2)
        return self.supertrace(exp_A2)
