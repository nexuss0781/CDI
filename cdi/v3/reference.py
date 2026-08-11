"""Test-only dense oracle for validating DCSS-CDI matrix-free operators."""
from __future__ import annotations

import torch

from .laplacian import MatrixFreeLaplacian


class DenseReferenceOperators:
    """Dense small-system reference built from exactly the sparse topology and weights.

    This object is intentionally not an ``nn.Module`` and rejects systems whose
    factorized full-state width exceeds its safety limit.  It must not be used in
    a standard DCSS-CDI forward path.
    """

    def __init__(self, laplacian: MatrixFreeLaplacian, max_state_dim: int | None = None) -> None:
        self.laplacian = laplacian
        self.max_state_dim = max_state_dim or laplacian.config.dense_reference_limit
        if laplacian.config.total_state_dim > self.max_state_dim:
            raise ValueError(
                f"Dense reference refused: state dim {laplacian.config.total_state_dim} exceeds "
                f"the hard safety limit {self.max_state_dim}."
            )

    @classmethod
    def build_small(cls, laplacian: MatrixFreeLaplacian, max_state_dim: int | None = None) -> "DenseReferenceOperators":
        return cls(laplacian, max_state_dim=max_state_dim)

    def matrix(self, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> torch.Tensor:
        if self.laplacian.config.geometry_ablation:
            vertices = self.laplacian.topology.n_vertices
            target_dtype = dtype or self.laplacian.edge_log_weights.dtype
            target_device = device or self.laplacian.edge_log_weights.device
            return torch.zeros((vertices, vertices), dtype=target_dtype, device=target_device)
        target_dtype = dtype or self.laplacian.edge_log_weights.dtype
        target_device = device or self.laplacian.edge_log_weights.device
        incidence = self.laplacian.incidence.dense(dtype=target_dtype, device=target_device)
        weights = self.laplacian.edge_weights.to(dtype=target_dtype, device=target_device)
        return incidence.transpose(0, 1) @ torch.diag(weights) @ incidence

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2 or x.shape[-2] != self.laplacian.topology.n_vertices:
            raise ValueError("Dense reference expects (..., n_vertices, channels) input.")
        matrix = self.matrix(dtype=x.dtype, device=x.device)
        return torch.einsum("ij,...jc->...ic", matrix, x)

    def full_state_matrix(self, channels: int | None = None) -> torch.Tensor:
        """Create a test-only block diagonal global matrix; never call in production."""
        width = channels or self.laplacian.config.state_width
        state_dim = self.laplacian.topology.n_vertices * width
        if state_dim > self.max_state_dim:
            raise ValueError(f"Dense global reference refused at state dim {state_dim}.")
        return torch.kron(self.matrix(), torch.eye(width, dtype=self.laplacian.edge_log_weights.dtype))
