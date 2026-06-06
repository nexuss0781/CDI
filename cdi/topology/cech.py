"""
§2.2 Čech Cohomology
====================

Definition 2.2.1: The Čech cochain complex w.r.t. a good cover 𝔘:

    Č^k(𝔘, 𝒪) = ∏_{i₀ < ··· < iₖ} 𝒪(U_{i₀} ∩ ··· ∩ U_{iₖ})

Coboundary:
    (δ̌σ)_{i₀···i_{k+1}} = Σⱼ (−1)^j σ_{i₀···î_j···i_{k+1}}|_{intersection}

Theorem 2.2.2 (Leray): For a good cover of a soft sheaf,
    Ȟ^k(𝔘, 𝒪) ≅ H^k(M, 𝒪).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from cdi.config import CDIConfig


class CechCohomology:
    """Čech cohomology on the nerve of a good cover.

    Attributes
    ----------
    cover : GoodCover
    belief : BeliefComplex
    config : CDIConfig
    """

    def __init__(self, cover, belief, config: CDIConfig) -> None:
        self.cover = cover
        self.belief = belief
        self.config = config

    # ------------------------------------------------------------------
    # Simplex enumeration
    # ------------------------------------------------------------------

    def _simplices(self, dim: int) -> list:
        """Return simplices of the nerve at given dimension."""
        return self.cover.nerve_simplices(dim)

    # ------------------------------------------------------------------
    # Čech coboundary
    # ------------------------------------------------------------------

    def coboundary_matrix(self, degree: int, belief_degree: int = 0) -> torch.Tensor:
        """Čech coboundary δ̌: Č^k → Č^{k+1} for belief sheaf at given degree.

        The matrix maps k-cochain values to (k+1)-cochain values using
        the alternating-sum formula.

        Parameters
        ----------
        degree : int       Čech degree k.
        belief_degree : int  Degree q in the belief complex.

        Returns
        -------
        torch.Tensor
            Shape ``(n_{k+1} · d_q, n_k · d_q)`` where n_k is the
            number of k-simplices and d_q = dim ℬ_q.
        """
        simplices_k = self._simplices(degree)
        simplices_k1 = self._simplices(degree + 1)
        d_q = self.belief.dims[self.belief.degree_to_index(belief_degree)]

        n_k = len(simplices_k)
        n_k1 = len(simplices_k1)

        if n_k == 0 or n_k1 == 0:
            return torch.zeros(n_k1 * d_q, n_k * d_q, dtype=self.config.dtype)

        # Build map: simplex → index
        if degree == 0:
            simplex_to_idx = {s: i for i, s in enumerate(simplices_k)}
        else:
            simplex_to_idx = {tuple(s): i for i, s in enumerate(simplices_k)}

        mat = torch.zeros(n_k1 * d_q, n_k * d_q, dtype=self.config.dtype)
        I_dq = torch.eye(d_q, dtype=self.config.dtype)

        for row_idx, sigma in enumerate(simplices_k1):
            if degree == 0:
                # sigma is a pair (i, j); faces are i and j
                i, j = sigma
                for face_pos, face in enumerate([j, i]):
                    sign = (-1.0) ** face_pos
                    if face in simplex_to_idx:
                        col_idx = simplex_to_idx[face]
                        mat[
                            row_idx * d_q : (row_idx + 1) * d_q,
                            col_idx * d_q : (col_idx + 1) * d_q,
                        ] += sign * I_dq
            elif degree == 1:
                # sigma is a triple (i, j, k); faces are (j,k), (i,k), (i,j)
                i, j, k = sigma
                faces = [(j, k), (i, k), (i, j)]
                for face_pos, face in enumerate(faces):
                    sign = (-1.0) ** face_pos
                    key = tuple(sorted(face))
                    if key in simplex_to_idx:
                        col_idx = simplex_to_idx[key]
                        mat[
                            row_idx * d_q : (row_idx + 1) * d_q,
                            col_idx * d_q : (col_idx + 1) * d_q,
                        ] += sign * I_dq

        return mat

    # ------------------------------------------------------------------
    # Cohomology computation
    # ------------------------------------------------------------------

    def cohomology(self, degree: int, belief_degree: int = 0) -> Tuple[int, torch.Tensor]:
        """Compute Ȟ^k(𝔘, ℬ_q) = ker δ̌^k / im δ̌^{k−1}.

        Returns
        -------
        (dimension, basis) : tuple[int, Tensor]
        """
        d_q = self.belief.dims[self.belief.degree_to_index(belief_degree)]
        n_k = len(self._simplices(degree))

        if n_k == 0:
            return 0, torch.zeros(0, 0, dtype=self.config.dtype)

        # Kernel of δ̌^k
        delta_k = self.coboundary_matrix(degree, belief_degree)
        if delta_k.numel() == 0 or delta_k.shape[1] == 0:
            ker_dim = n_k * d_q
            ker_basis = torch.eye(ker_dim, dtype=self.config.dtype)
        else:
            U, S, Vh = torch.linalg.svd(delta_k, full_matrices=True)
            rank_k = int((S > 1e-10).sum().item())
            ker_basis = Vh[rank_k:].T  # (n_k·d_q, ker_dim)
            ker_dim = ker_basis.shape[1]

        # Image of δ̌^{k−1}
        if degree == 0:
            im_dim = 0
        else:
            delta_km1 = self.coboundary_matrix(degree - 1, belief_degree)
            if delta_km1.numel() == 0:
                im_dim = 0
            else:
                _, S_km1, _ = torch.linalg.svd(delta_km1, full_matrices=False)
                im_dim = int((S_km1 > 1e-10).sum().item())

        h_dim = max(ker_dim - im_dim, 0)
        return h_dim, ker_basis[:, :h_dim] if h_dim > 0 else torch.zeros(n_k * d_q, 0, dtype=self.config.dtype)

    # ------------------------------------------------------------------
    # Double complex
    # ------------------------------------------------------------------

    def total_cohomology(self, total_degree: int) -> int:
        """dim of total cohomology H^k of the Čech–belief double complex.

        H^k = ⊕_{p+q=k} Ȟ^p(𝔘, ℬ_q).
        """
        total = 0
        for q_idx, q in enumerate(self.belief.degrees):
            p = total_degree - q
            if p >= 0 and p <= 2:  # nerve dimension ≤ 2 tracked
                dim, _ = self.cohomology(p, q)
                total += dim
        return total
