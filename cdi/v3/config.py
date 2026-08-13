"""Configuration for the DCSS-CDI v3 sparse operator substrate.

Stage B deliberately keeps this configuration independent from CDI v2 execution.
The legacy ``CDIConfig`` receives the compatible ``geometry_ablation`` flag, while
new factorized-state options live here to avoid changing the frozen baseline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch


@dataclass(frozen=True)
class DCSSConfig:
    """Immutable configuration for CPU-first sparse cochain operators.

    ``state_width`` is the factorized channel dimension.  It is intentionally
    not the legacy Kronecker-expanded global-state size.
    """

    name: str = "nano"
    n_vertices: int = 4
    state_width: int = 8
    cover_k: int = 2
    seed: int = 42
    dtype_str: str = "float32"
    device: str = "cpu"
    geometry_ablation: bool = False
    dense_reference_limit: int = 2048
    allocation_fraction_limit: float = 0.50
    spectral_target: float = 0.10
    energy_limit: float = 1.0e8
    max_geometry_edge_weight: float = 2.0

    @property
    def dtype(self) -> torch.dtype:
        try:
            return getattr(torch, self.dtype_str)
        except AttributeError as exc:
            raise ValueError(f"Unsupported torch dtype: {self.dtype_str}") from exc

    @property
    def total_state_dim(self) -> int:
        """The factorized state width used by Stage B diagnostics."""
        return self.n_vertices * self.state_width

    def validate(self) -> None:
        if self.name != "nano" and not self.name.startswith("scale-"):
            raise ValueError(f"Unknown Stage B configuration tier: {self.name}")
        if self.n_vertices < 3:
            raise ValueError("A non-vacuous Stage B topology needs at least three vertices.")
        if self.state_width <= 0:
            raise ValueError("state_width must be positive.")
        if not 1 <= self.cover_k < self.n_vertices:
            raise ValueError("cover_k must be in [1, n_vertices).")
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("Stage B supports only float32 and float64.")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("Stage B accepts CPU or CUDA device strings only.")
        if self.allocation_fraction_limit <= 0.0:
            raise ValueError("allocation_fraction_limit must be positive.")
        if not 0.0 < self.max_geometry_edge_weight <= 2.0:
            raise ValueError("max_geometry_edge_weight must lie in (0, 2] for the nano stability envelope.")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def nano(cls, seed: int = 42, geometry_ablation: bool = False) -> "DCSSConfig":
        """Rapid-iteration configuration with a factorized state dimension of 32."""
        config = cls(seed=seed, geometry_ablation=geometry_ablation)
        config.validate()
        if config.total_state_dim >= 64:
            raise AssertionError("The nano tier must remain below total_state_dim 64.")
        return config

    @classmethod
    def scaled(cls, n_vertices: int, seed: int = 42) -> "DCSSConfig":
        """Create a valid CPU scaling configuration without increasing channel width."""
        config = cls(
            name=f"scale-{n_vertices}",
            n_vertices=n_vertices,
            state_width=8,
            cover_k=min(2, n_vertices - 1),
            seed=seed,
        )
        config.validate()
        return config


def load_config(name: str, seed: int = 42, geometry_ablation: bool = False) -> DCSSConfig:
    """Resolve an explicitly supported Stage B configuration tier."""
    if name != "nano":
        raise ValueError(f"Stage B currently exposes only the customized 'nano' tier, not {name!r}.")
    return DCSSConfig.nano(seed=seed, geometry_ablation=geometry_ablation)
