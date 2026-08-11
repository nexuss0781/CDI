"""Structural sparse cochain maps for the DCSS-CDI Stage B substrate."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .topology import SparseTopology


@dataclass(frozen=True)
class SparseCochainMap:
    """Coboundary maps derived from immutable topology boundaries.

    The degree-zero map is ``δ₀ = ∂₁ᵀ`` and the degree-one map is
    ``δ₁ = ∂₂ᵀ``.  Thus ``δ₁ δ₀ = (∂₁ ∂₂)ᵀ = 0`` structurally rather than by
    penalty fitting.  Production application uses sparse matrix multiplication;
    dense materialization is reserved for test-only references.
    """

    topology: SparseTopology

    def apply_degree(self, degree: int, x: torch.Tensor) -> torch.Tensor:
        if degree == 0:
            return self._right_apply_sparse(self.topology.boundary_one.transpose(0, 1).coalesce(), x, self.topology.n_vertices)
        if degree == 1:
            return self._right_apply_sparse(self.topology.boundary_two.transpose(0, 1).coalesce(), x, self.topology.n_edges)
        raise ValueError(f"No Stage B cochain map is declared for degree {degree}.")

    def compose(self, degree: int, x: torch.Tensor) -> torch.Tensor:
        if degree != 0:
            raise ValueError("Stage B only declares the adjacent composition δ₁δ₀.")
        return self.apply_degree(1, self.apply_degree(0, x))

    def residual(self, degree: int, x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        composed = self.compose(degree, x)
        numerator = torch.linalg.vector_norm(composed.reshape(-1))
        denominator = torch.linalg.vector_norm(x.reshape(-1)).clamp_min(eps)
        return numerator / denominator

    def dense(self, degree: int, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        if degree == 0:
            return self.topology.boundary_one.to_dense().transpose(0, 1).to(dtype=dtype, device=device)
        if degree == 1:
            return self.topology.boundary_two.to_dense().transpose(0, 1).to(dtype=dtype, device=device)
        raise ValueError(f"No Stage B cochain map is declared for degree {degree}.")

    @staticmethod
    def _right_apply_sparse(matrix: torch.Tensor, x: torch.Tensor, expected_cells: int) -> torch.Tensor:
        if x.ndim < 2 or x.shape[-2] != expected_cells:
            raise ValueError(f"Expected (..., {expected_cells}, channels), received {tuple(x.shape)}.")
        leading = x.shape[:-2]
        channels = x.shape[-1]
        flat = x.reshape(-1, expected_cells, channels)
        outputs = []
        operator = matrix.to(dtype=x.dtype, device=x.device)
        for sample in flat:
            outputs.append(torch.sparse.mm(operator, sample))
        return torch.stack(outputs, dim=0).reshape(*leading, matrix.shape[0], channels)
