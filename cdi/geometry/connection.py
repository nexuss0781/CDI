"""
§4.3 Belief Connection — Gauge Field A and Curvature F_A
=========================================================

Implements the belief connection from CDI Specification §4.3.

Definition 4.3.1: The belief connection is
    A ∈ Ω¹(M, End(ℬ^•))
satisfying compatibility with the differential: A∘δ = δ∘A.

Definition 4.3.2: The curvature is
    F_A = dA + A ∧ A

Theorem 4.3.3 (Bianchi): d_A F_A = 0.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from cdi.config import CDIConfig


class BeliefConnection:
    """Discrete belief connection on the nerve graph.

    On each edge (i, j) of the cover's nerve, the connection is
    a skew-symmetric matrix A_{ij} ∈ End(ℬ_total) with A_{ij} = −A_{ji}.

    Parameterised as A = W − Wᵀ for learnable W, ensuring skew-symmetry.

    Attributes
    ----------
    config : CDIConfig
    B : int
        Total belief dimension Σ_k dim(ℬ_k).
    edges : list[tuple[int, int]]
        Edges of the nerve.
    W_params : dict[tuple, torch.Tensor]
        Raw learnable matrices; A_{ij} = W − Wᵀ.
    """

    def __init__(
        self, config: CDIConfig, edges: List[Tuple[int, int]]
    ) -> None:
        self.config = config
        self.B = config.total_belief_dim
        self.n = config.n_points
        self.edges = list(edges)
        dtype = config.dtype

        # Learnable raw matrices — one per edge
        self.W_params: dict = {}
        scale = 0.01  # small initial connection
        for i, j in self.edges:
            key = (min(i, j), max(i, j))
            if key not in self.W_params:
                W = torch.randn(self.B, self.B, dtype=dtype) * scale
                W.requires_grad_(True)
                self.W_params[key] = W

    # ------------------------------------------------------------------
    # Connection on edges
    # ------------------------------------------------------------------

    def connection_on_edge(self, i: int, j: int) -> torch.Tensor:
        """A_{ij} = W − Wᵀ (skew-symmetric).

        Returns shape ``(B, B)``; satisfies A_{ij} = −A_{ji}.
        """
        key = (min(i, j), max(i, j))
        W = self.W_params.get(key)
        if W is None:
            return torch.zeros(self.B, self.B, dtype=self.config.dtype)
        A = W - W.T
        if i > j:
            A = -A  # A_{ji} = −A_{ij}
        return A

    # ------------------------------------------------------------------
    # Curvature on triangles
    # ------------------------------------------------------------------

    def curvature_on_triangle(self, i: int, j: int, k: int) -> torch.Tensor:
        """Discrete curvature F_A on triangle (i, j, k).

        F_{ijk} = A_{ij} + A_{jk} + A_{ki}  (holonomy around triangle).

        Returns shape ``(B, B)``.
        """
        return (
            self.connection_on_edge(i, j)
            + self.connection_on_edge(j, k)
            + self.connection_on_edge(k, i)
        )

    # ------------------------------------------------------------------
    # Parallel transport
    # ------------------------------------------------------------------

    def parallel_transport(self, section: torch.Tensor, i: int, j: int) -> torch.Tensor:
        """Transport a section from point j to point i.

        Uses first-order approximation: P_{j→i} ≈ I + A_{ij}.

        Parameters
        ----------
        section : torch.Tensor
            Shape ``(B,)`` or ``(..., B)``.

        Returns
        -------
        torch.Tensor
            Transported section — same shape.
        """
        A = self.connection_on_edge(i, j)  # (B, B)
        transport = torch.eye(self.B, dtype=self.config.dtype) + A
        return section @ transport.T

    # ------------------------------------------------------------------
    # Full connection matrix
    # ------------------------------------------------------------------

    def full_connection_matrix(self) -> torch.Tensor:
        """Assemble A into an (n·B, n·B) block matrix.

        Block (i, j) of size (B, B) is A_{ij} if (i,j) is an edge, else 0.
        """
        N = self.n * self.B
        A_full = torch.zeros(N, N, dtype=self.config.dtype)
        for i, j in self.edges:
            A_ij = self.connection_on_edge(i, j)
            A_full[i*self.B:(i+1)*self.B, j*self.B:(j+1)*self.B] = A_ij
            A_full[j*self.B:(j+1)*self.B, i*self.B:(i+1)*self.B] = -A_ij
        return A_full

    # ------------------------------------------------------------------
    # Penalties
    # ------------------------------------------------------------------

    def bianchi_penalty(self, triangles: List[Tuple[int, int, int]]) -> torch.Tensor:
        """‖d_A F_A‖² approximation on triangles.

        The Bianchi identity says d_A F_A = 0.  We measure the
        violation on each triangle as the commutator [A, F].
        """
        penalty = torch.tensor(0.0, dtype=self.config.dtype)
        for i, j, k in triangles:
            F = self.curvature_on_triangle(i, j, k)
            A_ij = self.connection_on_edge(i, j)
            comm = A_ij @ F - F @ A_ij
            penalty = penalty + torch.sum(comm ** 2)
        return penalty / max(len(triangles), 1)

    def compatibility_penalty(self, belief_delta_full: torch.Tensor) -> torch.Tensor:
        """‖[A, δ]‖² — compatibility of connection with the differential.

        Parameters
        ----------
        belief_delta_full : torch.Tensor
            Full δ matrix, shape ``(B, B)``.

        Returns
        -------
        torch.Tensor
            Non-negative scalar.
        """
        penalty = torch.tensor(0.0, dtype=self.config.dtype)
        count = 0
        for i, j in self.edges:
            A_ij = self.connection_on_edge(i, j)
            comm = A_ij @ belief_delta_full - belief_delta_full @ A_ij
            penalty = penalty + torch.sum(comm ** 2)
            count += 1
        return penalty / max(count, 1)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def get_parameters(self) -> List[torch.Tensor]:
        """Learnable parameters."""
        return list(self.W_params.values())
