"""
§3 Belief Complex — Graded Cochain Complex B^• with δ²=0
=========================================================

Implements the belief complex ℬ^• from CDI Specification §3.

The belief complex is a bounded cochain complex:
    ··· → ℬ_{-m} →^{δ⁻ᵐ} ℬ_{-m+1} → ··· → ℬ₀ →^{δ⁰} ℬ₁ → ··· → ℬₙ → 0

with δ^{k+1} ∘ δ^k = 0 (logical consistency axiom §3.1.2).

Degree interpretation (§3.2):
    k = -m,...,-1  Motor sheaves (action potentials)
    k = 0          Perceptual sheaf (raw sensorium)
    k = 1          Causal sheaf (cause-effect)
    k = 2          Abstract sheaf (analogies / metaphors)
    k ≥ 3          Meta-sheaves (higher-order self-reference)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from cdi.config import CDIConfig


class BeliefComplex:
    """Graded cochain complex ℬ^• with coboundary maps δ.

    Axiom 3.1.2: δ² = 0 is the logical consistency axiom, enforced
    via a differentiable penalty in the loss function.

    Attributes
    ----------
    config : CDIConfig
    dims : list[int]
        Dimension of each belief sheaf [dim ℬ_{-m}, ..., dim ℬ_n].
    degrees : list[int]
        List of integer degrees [-m, ..., n].
    deltas : list[torch.Tensor]
        Coboundary maps δ^k: ℬ_k → ℬ_{k+1}.
        ``deltas[i]`` has shape ``(dims[i+1], dims[i])``.
    """

    def __init__(self, config: CDIConfig) -> None:
        self.config = config
        self.dims: List[int] = list(config.belief_dims)
        self.n_degrees: int = config.n_degrees
        self.motor_depth: int = config.motor_depth
        self.degrees: List[int] = list(config.degree_range)
        dtype = config.dtype

        assert len(self.dims) == self.n_degrees, (
            f"belief_dims length {len(self.dims)} ≠ n_degrees {self.n_degrees}"
        )

        # Coboundary maps δ^k: ℬ_k → ℬ_{k+1} — Xavier init
        self.deltas: List[torch.Tensor] = []
        for i in range(self.n_degrees - 1):
            dim_from = self.dims[i]
            dim_to = self.dims[i + 1]
            scale = (2.0 / (dim_from + dim_to)) ** 0.5
            delta = torch.randn(dim_to, dim_from, dtype=dtype) * scale
            delta.requires_grad_(True)
            self.deltas.append(delta)

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def degree_to_index(self, k: int) -> int:
        """Convert degree k ∈ [-m, n] to a 0-based list index."""
        idx = k + self.motor_depth
        assert 0 <= idx < self.n_degrees, f"degree {k} out of range"
        return idx

    # ------------------------------------------------------------------
    # Coboundary operators
    # ------------------------------------------------------------------

    def coboundary(self, k: int, section: torch.Tensor) -> torch.Tensor:
        """Apply δ^k: ℬ_k → ℬ_{k+1}.

        Parameters
        ----------
        k : int
            Source degree.
        section : torch.Tensor
            Shape ``(..., dim_k)``.

        Returns
        -------
        torch.Tensor
            Shape ``(..., dim_{k+1})``.
        """
        idx = self.degree_to_index(k)
        assert idx < len(self.deltas), f"No coboundary from top degree {k}"
        return section @ self.deltas[idx].T

    def adjoint_coboundary(self, k: int, section: torch.Tensor) -> torch.Tensor:
        """Apply (δ^{k-1})* = (δ^{k-1})ᵀ : ℬ_k → ℬ_{k-1}.

        Parameters
        ----------
        k : int
            Source degree (must be > -m).
        section : torch.Tensor
            Shape ``(..., dim_k)``.

        Returns
        -------
        torch.Tensor
            Shape ``(..., dim_{k-1})``.
        """
        idx = self.degree_to_index(k) - 1
        assert idx >= 0, f"No adjoint coboundary from bottom degree {k}"
        return section @ self.deltas[idx]

    # ------------------------------------------------------------------
    # Combinatorial Laplacian  Δ_δ = δδ* + δ*δ
    # ------------------------------------------------------------------

    def combinatorial_laplacian_at_degree(self, k: int) -> torch.Tensor:
        """Combinatorial Laplacian at degree k:

        Δ_δ|_k = (δ^k)* δ^k + δ^{k-1} (δ^{k-1})*

        Returns
        -------
        torch.Tensor
            Shape ``(dim_k, dim_k)``.
        """
        idx = self.degree_to_index(k)
        dim_k = self.dims[idx]
        result = torch.zeros(dim_k, dim_k, dtype=self.config.dtype)

        # (δ^k)* δ^k  term — "upward" Laplacian
        if idx < len(self.deltas):
            delta_k = self.deltas[idx]  # (dim_{k+1}, dim_k)
            result = result + delta_k.T @ delta_k

        # δ^{k-1} (δ^{k-1})*  term — "downward" Laplacian
        if idx > 0:
            delta_km1 = self.deltas[idx - 1]  # (dim_k, dim_{k-1})
            result = result + delta_km1 @ delta_km1.T

        return result

    def full_combinatorial_laplacian(self) -> torch.Tensor:
        """Block-diagonal combinatorial Laplacian Δ_δ over all degrees.

        Returns
        -------
        torch.Tensor
            Shape ``(B_total, B_total)`` where B_total = Σ_k dim_k.
        """
        B = self.config.total_belief_dim
        result = torch.zeros(B, B, dtype=self.config.dtype)
        offset = 0
        for k in self.degrees:
            dim_k = self.dims[self.degree_to_index(k)]
            block = self.combinatorial_laplacian_at_degree(k)
            result[offset : offset + dim_k, offset : offset + dim_k] = block
            offset += dim_k
        return result

    def full_coboundary_matrix(self) -> torch.Tensor:
        """Full δ matrix in the concatenated belief space.

        Returns the (B_total, B_total) matrix that applies δ^k
        on the appropriate block for each degree k.
        """
        B = self.config.total_belief_dim
        delta_full = torch.zeros(B, B, dtype=self.config.dtype)
        offset = 0
        for i, k in enumerate(self.degrees[:-1]):
            dim_from = self.dims[i]
            dim_to = self.dims[i + 1]
            offset_to = offset + dim_from
            delta_full[offset_to : offset_to + dim_to, offset : offset + dim_from] = (
                self.deltas[i]
            )
            offset += dim_from
        return delta_full

    def full_adjoint_coboundary_matrix(self) -> torch.Tensor:
        """Full δ* matrix — transpose of full_coboundary_matrix."""
        return self.full_coboundary_matrix().T

    # ------------------------------------------------------------------
    # Cohomology
    # ------------------------------------------------------------------

    def cohomology_dim(self, k: int) -> int:
        """dim H^k = dim(ker δ^k) − dim(im δ^{k-1}).

        Returns
        -------
        int
            Non-negative integer.
        """
        idx = self.degree_to_index(k)
        dim_k = self.dims[idx]

        # dim ker(δ^k) = dim_k − rank(δ^k)
        if idx < len(self.deltas):
            rank_dk = int(torch.linalg.matrix_rank(self.deltas[idx]).item())
            ker_dim = dim_k - rank_dk
        else:
            ker_dim = dim_k  # top degree — everything is closed

        # dim im(δ^{k-1}) = rank(δ^{k-1})
        if idx > 0:
            im_dim = int(torch.linalg.matrix_rank(self.deltas[idx - 1]).item())
        else:
            im_dim = 0  # bottom degree — no image from below

        return max(ker_dim - im_dim, 0)

    # ------------------------------------------------------------------
    # State assembly / splitting
    # ------------------------------------------------------------------

    def assemble_state(self, degree_sections: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Concatenate per-degree sections into a single belief vector.

        Parameters
        ----------
        degree_sections : dict[int, Tensor]
            ``{k: tensor of shape (..., dim_k)}`` for each k in self.degrees.

        Returns
        -------
        torch.Tensor
            Shape ``(..., total_belief_dim)``.
        """
        parts = [degree_sections[k] for k in self.degrees]
        return torch.cat(parts, dim=-1)

    def split_state(self, state: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Split a concatenated belief vector into per-degree tensors.

        Parameters
        ----------
        state : torch.Tensor
            Shape ``(..., total_belief_dim)``.

        Returns
        -------
        dict[int, Tensor]
            ``{k: tensor of shape (..., dim_k)}``.
        """
        result = {}
        offset = 0
        for i, k in enumerate(self.degrees):
            dim_k = self.dims[i]
            result[k] = state[..., offset : offset + dim_k]
            offset += dim_k
        return result

    # ------------------------------------------------------------------
    # Consistency penalty   ‖δ²‖²
    # ------------------------------------------------------------------

    def consistency_penalty(self) -> torch.Tensor:
        """Penalty for δ²=0 violation: Σ_k ‖δ^{k+1} δ^k‖²_F.

        Returns
        -------
        torch.Tensor
            Non-negative scalar.
        """
        penalty = torch.tensor(0.0, dtype=self.config.dtype)
        for i in range(len(self.deltas) - 1):
            delta_sq = self.deltas[i + 1] @ self.deltas[i]  # should be 0
            penalty = penalty + torch.sum(delta_sq ** 2)
        return penalty

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def get_parameters(self) -> List[torch.Tensor]:
        """Learnable parameters: all coboundary matrices δ^k."""
        return list(self.deltas)
