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
        self.register_buffer(
            "_dense_incidence",
            self.incidence.dense(dtype=config.dtype, device=config.device),
            persistent=False,
        )
        # Preserve the historical softplus effective initialization while using
        # a bounded logit parameterization thereafter.  A hard post-update
        # rejection makes a valid optimizer trajectory terminate at the cap;
        # this mapping instead keeps all learned weights in the same declared
        # stability envelope with a differentiable gradient.
        historical_raw = torch.linspace(-0.2, 0.2, topology.n_edges, dtype=config.dtype, device=config.device)
        historical_weights = torch.nn.functional.softplus(historical_raw) + 1.0e-6
        maximum = torch.as_tensor(config.max_geometry_edge_weight, dtype=config.dtype, device=config.device)
        ratio = (historical_weights / maximum).clamp(min=1.0e-6, max=1.0 - 1.0e-6)
        initial = torch.logit(ratio)
        self.edge_log_weights = nn.Parameter(initial)
        self.register_buffer(
            "_maximum_edge_weight",
            torch.as_tensor(config.max_geometry_edge_weight, dtype=config.dtype, device=config.device),
            persistent=False,
        )

    @property
    def state_shape(self) -> tuple[int, int]:
        return (self.topology.n_vertices, self.config.state_width)

    @property
    def full_state_square(self) -> int:
        return self.config.total_state_dim ** 2

    @property
    def edge_weights(self) -> torch.Tensor:
        return self._maximum_edge_weight.to(
            dtype=self.edge_log_weights.dtype,
            device=self.edge_log_weights.device,
        ) * torch.sigmoid(self.edge_log_weights)

    def operator(self, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> torch.Tensor:
        """Build the differentiable vertex Laplacian once for a forward chunk.

        The returned operator is intentionally not cached across optimizer steps:
        it remains connected to ``edge_log_weights`` so gradients are preserved.
        Callers that process multiple recurrent tokens may reuse it for the whole
        chunk instead of rebuilding the same tiny matrix at every token.
        """
        target_dtype = dtype or self.edge_log_weights.dtype
        target_device = device or self.edge_log_weights.device
        if self.config.geometry_ablation:
            return torch.zeros(
                self.topology.n_vertices,
                self.topology.n_vertices,
                dtype=target_dtype,
                device=target_device,
            )
        weights = self.edge_weights.to(dtype=target_dtype, device=target_device)
        incidence = self._dense_incidence.to(dtype=target_dtype, device=target_device)
        return incidence.transpose(-2, -1) @ (incidence * weights.unsqueeze(-1))

    def apply(self, x: torch.Tensor, operator: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the factored geometry to ``(..., vertices, channels)`` state.

        ``operator`` may be supplied by a chunk-level caller to reuse the
        differentiable Laplacian across recurrent tokens.
        """
        if x.ndim < 2 or tuple(x.shape[-2:]) != self.state_shape:
            raise ValueError(f"Expected (..., {self.state_shape[0]}, {self.state_shape[1]}), got {tuple(x.shape)}.")
        if self.config.geometry_ablation:
            return torch.zeros_like(x)
        laplacian = operator if operator is not None else self.operator(dtype=x.dtype, device=x.device)
        if laplacian.ndim != 2 or tuple(laplacian.shape) != (self.topology.n_vertices, self.topology.n_vertices):
            raise ValueError("Geometry operator must have shape (vertices, vertices).")
        return torch.matmul(laplacian, x)


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
            "factorization": "S_transpose @ diag(max_edge_weight * sigmoid(edge_log_weights)) @ S",
            "geometry_ablation": self.config.geometry_ablation,
            "full_state_dim": self.config.total_state_dim,
            "full_state_square": self.full_state_square,
            "sparse_nnz": self.incidence.nnz,
            "learnable_parameters": ["edge_log_weights"],
            "max_geometry_edge_weight": self.config.max_geometry_edge_weight,
            "weight_parameterization": "bounded_sigmoid_preserving_historical_softplus_initialization",
            "forbidden_operations": ["torch.kron", "dense_full_state_operator"],
        }
