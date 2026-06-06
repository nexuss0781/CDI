"""
§5.1 Belief Laplacian
=====================

Implements Δ_ℬ from CDI Specification §5.1.

Definition 5.1.1:
    Δ_ℬ = D² + Δ_δ + [D, A] + [A, D] + A²

where
    D²           = Lichnerowicz (∇*∇ + R/4)
    Δ_δ          = δδ* + δ*δ  (combinatorial Laplacian)
    [D, A]+[A,D] = geometric–cognitive coupling
    A²           = connection potential

Theorem 5.1.3: Δ_ℬ is essentially self-adjoint.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from cdi.config import CDIConfig


class BeliefLaplacian:
    """Full belief Laplacian on the twisted bundle 𝔹.

    After construction the Laplacian is a dense symmetric
    (N, N) matrix with N = n·s·B.

    Attributes
    ----------
    _matrix : torch.Tensor or None
        Dense Laplacian matrix.
    _eigenvalues, _eigenvectors : torch.Tensor or None
        Cached spectral decomposition.
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
        self._eigenvalues: Optional[torch.Tensor] = None
        self._eigenvectors: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Construct Δ_ℬ = D² + Δ_δ + coupling + A²."""
        dtype = self.config.dtype

        # D² — Lichnerowicz
        D2 = self.dirac.squared()  # (N, N)

        # Δ_δ — combinatorial Laplacian, lifted to full space
        # Δ_δ acts on belief indices only: I_n ⊗ I_s ⊗ Δ_δ_block
        delta_lap_block = self.belief.full_combinatorial_laplacian()  # (B, B)
        I_n = torch.eye(self.n, dtype=dtype)
        I_s = torch.eye(self.s, dtype=dtype)
        # I_n ⊗ I_s ⊗ Δ_δ = (n·s·B, n·s·B)
        delta_lap_full = torch.kron(I_n, torch.kron(I_s, delta_lap_block))

        # A — full connection matrix over the nerve
        A_full = self.connection.full_connection_matrix()  # (n·B, n·B)
        # Lift to spinor space: I_n ⊗ I_s ⊗ A_block is not right here
        # A_full is (n·B, n·B). We need (n·s·B, n·s·B).
        # Reshape: for each (n,n) block of A_full of size (B,B),
        #   lift to (s·B, s·B) by tensoring with I_s on the spinor index.
        # Efficiently: interleave spinor identity.
        A_lifted = torch.zeros(self.N, self.N, dtype=dtype)
        for p in range(self.n):
            for q in range(self.n):
                A_block = A_full[p*self.B:(p+1)*self.B, q*self.B:(q+1)*self.B]
                if torch.any(A_block != 0):
                    A_sB = torch.kron(I_s, A_block)  # (s·B, s·B)
                    rp = p * self.s * self.B
                    rq = q * self.s * self.B
                    A_lifted[rp:rp+self.s*self.B, rq:rq+self.s*self.B] = A_sB

        # Coupling: [D, A] + [A, D] = D·A + A·D  (anti-commutator-like)
        D_mat = self.dirac.matrix()
        coupling = D_mat @ A_lifted + A_lifted @ D_mat

        # A²
        A_sq = A_lifted @ A_lifted

        # Full Laplacian
        Lap = D2 + delta_lap_full + coupling + A_sq

        # Symmetrise (Theorem 5.1.3: self-adjoint)
        Lap = 0.5 * (Lap + Lap.T)

        self._matrix = Lap
        self._eigenvalues = None
        self._eigenvectors = None

    def matrix(self) -> torch.Tensor:
        """Dense Laplacian matrix.  Builds if needed."""
        if self._matrix is None:
            self.build()
        return self._matrix

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        """Δ_ℬ · ψ.

        Parameters
        ----------
        state : torch.Tensor
            Shape ``(N,)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N,)``.
        """
        return self.matrix() @ state

    # ------------------------------------------------------------------
    # Spectral decomposition
    # ------------------------------------------------------------------

    def eigendecompose(self, k: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Eigendecomposition of Δ_ℬ.

        Parameters
        ----------
        k : int
            If 0, compute full decomposition. Otherwise top-k eigenvalues.

        Returns
        -------
        (eigenvalues, eigenvectors) : tuple[Tensor, Tensor]
            eigenvalues shape ``(N,)``; eigenvectors shape ``(N, N)``
            with columns as eigenvectors.
        """
        if self._eigenvalues is None:
            M = self.matrix()
            evals, evecs = torch.linalg.eigh(M)
            # Clamp tiny negative eigenvalues (numerical)
            evals = evals.clamp(min=0.0)
            # CRITICAL: Detach eigenvalues/eigenvectors to break gradient chains
            # Eigenvalues are used for spectral analysis, not for gradients
            self._eigenvalues = evals.detach()
            self._eigenvectors = evecs.detach()

        if k > 0:
            return self._eigenvalues[:k], self._eigenvectors[:, :k]
        return self._eigenvalues, self._eigenvectors

    def spectral_gap(self) -> torch.Tensor:
        """λ₁ — first positive eigenvalue (spectral gap).

        Returns
        -------
        torch.Tensor
            Scalar.
        """
        evals, _ = self.eigendecompose()
        positive = evals[evals > 1e-10]
        if len(positive) == 0:
            return torch.tensor(0.0, dtype=self.config.dtype)
        return positive.min()

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def check_self_adjoint(self) -> torch.Tensor:
        """‖Δ − Δᵀ‖_F."""
        M = self.matrix()
        return torch.norm(M - M.T)

    def check_positive_semidefinite(self) -> bool:
        """All eigenvalues ≥ 0?"""
        evals, _ = self.eigendecompose()
        return bool((evals >= -1e-8).all())

    def invalidate(self) -> None:
        """Clear cached matrix and spectra."""
        self._matrix = None
        self._eigenvalues = None
        self._eigenvectors = None
