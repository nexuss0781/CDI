"""Matrix-free oriented incidence operators for DCSS-CDI Stage B."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .topology import SparseTopology


@dataclass(frozen=True)
class SparseIncidence:
    """Apply oriented vertex-edge incidence without materializing a dense matrix.

    Input layout is ``(..., n_vertices, channels)`` for :meth:`apply` and
    ``(..., n_edges, channels)`` for :meth:`transpose_apply`.  The canonical
    edge orientation is the immutable topology ordering ``source < target``.
    """

    topology: SparseTopology

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.topology.n_edges, self.topology.n_vertices)

    @property
    def nnz(self) -> int:
        return 2 * self.topology.n_edges

    @property
    def edge_index(self) -> torch.Tensor:
        return self.topology.edge_index

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``Sx`` as target values minus source values by indexed gather."""
        self._validate_vertex_state(x)
        edges = self.edge_index.to(device=x.device)
        return x[..., edges[1], :] - x[..., edges[0], :]

    def transpose_apply(self, y: torch.Tensor) -> torch.Tensor:
        """Compute ``Sᵀy`` by scatter-add, preserving leading batch dimensions."""
        self._validate_edge_state(y)
        output = torch.zeros(
            *y.shape[:-2], self.topology.n_vertices, y.shape[-1], dtype=y.dtype, device=y.device
        )
        edges = self.edge_index.to(device=y.device)
        view = (1,) * (y.ndim - 2) + (self.topology.n_edges, 1)
        source_index = edges[0].view(view).expand_as(y)
        target_index = edges[1].view(view).expand_as(y)
        output.scatter_add_(-2, source_index, -y)
        output.scatter_add_(-2, target_index, y)
        return output

    def dense(self, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> torch.Tensor:
        """Create a dense incidence matrix for diagnostics or the reference oracle only."""
        target_device = device if device is not None else self.edge_index.device
        target_dtype = dtype if dtype is not None else torch.float32
        matrix = torch.zeros(self.shape, dtype=target_dtype, device=target_device)
        edges = self.edge_index.to(device=target_device)
        matrix[torch.arange(self.topology.n_edges, device=target_device), edges[0]] = -1
        matrix[torch.arange(self.topology.n_edges, device=target_device), edges[1]] = 1
        return matrix

    def _validate_vertex_state(self, x: torch.Tensor) -> None:
        if x.ndim < 2 or x.shape[-2] != self.topology.n_vertices:
            raise ValueError(
                f"Expected (..., {self.topology.n_vertices}, channels) vertex state, received {tuple(x.shape)}."
            )

    def _validate_edge_state(self, y: torch.Tensor) -> None:
        if y.ndim < 2 or y.shape[-2] != self.topology.n_edges:
            raise ValueError(
                f"Expected (..., {self.topology.n_edges}, channels) edge state, received {tuple(y.shape)}."
            )
