"""
§2 Observation Sheaf — Mapping Data to Local Sections
=====================================================

Implements the observation sheaf 𝒪 from CDI Specification §2.1.

The observation sheaf assigns to each open set U a module 𝒪(U) of
observations restricted to U.  The observation current 𝒥: 𝒪 → ℬ₀
embeds observations into the degree-0 belief sheaf.
"""

from __future__ import annotations

from typing import List

import torch

from cdi.config import CDIConfig


class ObservationSheaf:
    """Observation sheaf with embedding into the belief complex.

    §2.1.1: An observation sheaf 𝒪 assigns to each open set a
    finite-dimensional R-module satisfying local finiteness,
    softness, and separability.

    §2.1.2: The observation current ι: 𝒪 → ℬ₀ maps observations
    into the degree-0 term of the belief complex.

    Attributes
    ----------
    config : CDIConfig
    obs_dim : int
        Dimension of raw observation vectors.
    belief_0_dim : int
        Dimension of the degree-0 belief sheaf ℬ₀.
    embedding_matrix : torch.Tensor
        Learnable matrix for ι: 𝒪 → ℬ₀, shape ``(belief_0_dim, obs_dim)``.
    output_matrix : torch.Tensor
        Learnable projection ℬ₀ → output space, shape ``(output_dim, belief_0_dim)``.
    """

    def __init__(self, config: CDIConfig) -> None:
        self.config = config
        self.obs_dim = config.observation_dim
        self.belief_0_dim = config.belief_dim(0)
        dtype = config.dtype

        # ι: 𝒪 → ℬ₀ — observation embedding (Xavier init)
        scale = (2.0 / (self.obs_dim + self.belief_0_dim)) ** 0.5
        self.embedding_matrix = (
            torch.randn(self.belief_0_dim, self.obs_dim, dtype=dtype) * scale
        )
        self.embedding_matrix.requires_grad_(True)

        # Readout: ℬ₀ → output
        scale_out = (2.0 / (self.belief_0_dim + config.output_dim)) ** 0.5
        self.output_matrix = (
            torch.randn(config.output_dim, self.belief_0_dim, dtype=dtype) * scale_out
        )
        self.output_matrix.requires_grad_(True)

    # ------------------------------------------------------------------
    # Embedding and projection
    # ------------------------------------------------------------------

    def embed(self, data: torch.Tensor) -> torch.Tensor:
        """Observation current ι: 𝒪 → ℬ₀.

        Parameters
        ----------
        data : torch.Tensor
            Shape ``(..., obs_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(..., belief_0_dim)``.
        """
        return data @ self.embedding_matrix.T

    def project_output(self, belief_0: torch.Tensor) -> torch.Tensor:
        """Project degree-0 belief back to output space.

        Parameters
        ----------
        belief_0 : torch.Tensor
            Shape ``(..., belief_0_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(..., output_dim)``.
        """
        return belief_0 @ self.output_matrix.T

    # ------------------------------------------------------------------
    # Sheaf restriction
    # ------------------------------------------------------------------

    def section(self, data: torch.Tensor, patch_indices: torch.Tensor) -> torch.Tensor:
        """Local section: restrict data to the points of a patch.

        Parameters
        ----------
        data : torch.Tensor
            Shape ``(n, *)``.
        patch_indices : torch.Tensor
            1-D LongTensor of point indices in the patch.

        Returns
        -------
        torch.Tensor
            Shape ``(|patch|, *)``.
        """
        return data[patch_indices]

    def restrict(
        self,
        section: torch.Tensor,
        from_indices: torch.Tensor,
        to_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Restriction map 𝒪(U) → 𝒪(V) for V ⊂ U.

        Parameters
        ----------
        section : torch.Tensor
            Shape ``(|from_indices|, *)``.
        from_indices, to_indices : torch.Tensor
            1-D LongTensors; ``to_indices`` must be a subset of ``from_indices``.

        Returns
        -------
        torch.Tensor
            Shape ``(|to_indices|, *)``.
        """
        mask = torch.isin(from_indices, to_indices)
        return section[mask]

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def get_parameters(self) -> List[torch.Tensor]:
        """Learnable parameters."""
        return [self.embedding_matrix, self.output_matrix]
