"""Factorized matrix-free graph Laplacian for DCSS-CDI Stage B."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .config import DCSSConfig
from .incidence import SparseIncidence
from .topology import SparseTopology


class MatrixFreeLaplacian(nn.Module):
    """Apply ``Lx = Sᵀ W Sx`` without a full-state dense operator.

    ``edge_log_weights`` are the only learnable geometry parameters in Stage B.
    Positive weights are produced by ``softplus`` so the declared Laplacian is
    symmetric positive semidefinite.  The ablation path returns an exact zero
    contribution and never falls back to a dense surrogate.
    """

    def __init__(self, topology: SparseTopology, config: DCSSConfig) -> None:
        super().__init__()
        config.validate()
        self.topology = topology
        self.config = config
        self.incidence = SparseIncidence(topology)
        initial = torch.linspace(-0.2, 0.2, topology.n_edges, dtype=config.dtype, device=config.device)
        self.edge_log_weights = nn.Parameter(initial)

    @property
    def state_shape(self) -> tuple[int, int]:
        return (self.topology.n_vertices, self.config.state_width)

    @property
    def full_state_square(self) -> int:
        return self.config.total_state_dim ** 2

    @property
    def edge_weights(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.edge_log_weights) + 1.0e-6

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the factored geometry to ``(..., vertices, channels)`` state."""
        if x.ndim < 2 or tuple(x.shape[-2:]) != self.state_shape:
            raise ValueError(f"Expected (..., {self.state_shape[0]}, {self.state_shape[1]}), got {tuple(x.shape)}.")
        if self.config.geometry_ablation:
            return torch.zeros_like(x)
        edge_values = self.incidence.apply(x)
        weights = self.edge_weights.to(dtype=x.dtype, device=x.device)
        edge_values = edge_values * weights.view((1,) * (edge_values.ndim - 2) + (-1, 1))
        return self.incidence.transpose_apply(edge_values)

    def quadratic_form(self, x: torch.Tensor) -> torch.Tensor:
        laplacian_x = self.apply(x)
        return (x * laplacian_x).sum(dim=(-2, -1))

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Return non-negative geometry energy per leading batch item."""
        if self.config.geometry_ablation:
            return torch.zeros(x.shape[:-2], dtype=x.dtype, device=x.device)
        edge_values = self.incidence.apply(x)
        weights = self.edge_weights.to(dtype=x.dtype, device=x.device)
        return 0.5 * (edge_values.square() * weights.view((1,) * (edge_values.ndim - 2) + (-1, 1))).sum(dim=(-2, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply(x)

    def production_metadata(self) -> Dict[str, object]:
        return {
            "operator": self.__class__.__name__,
            "factorization": "S_transpose @ diag(softplus(edge_log_weights)) @ S",
            "geometry_ablation": self.config.geometry_ablation,
            "full_state_dim": self.config.total_state_dim,
            "full_state_square": self.full_state_square,
            "sparse_nnz": self.incidence.nnz,
            "learnable_parameters": ["edge_log_weights"],
            "forbidden_operations": ["torch.kron", "dense_full_state_operator"],
        }
