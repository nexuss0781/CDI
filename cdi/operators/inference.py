"""
§5.3 Inference Operator — v2.0
================================

v2.0 Spec §2.1 / Fix F1 — CRITICAL CHANGES:
  - REMOVED all .detach() calls from the inference forward path
  - harmonic_part and green_d_star are fully connected to the graph
  - Gradients now flow to: manifold, connection, Dirac, Laplacian, belief δ maps
  - embed_observation uses live sheaf.embedding_matrix (unchanged, already live)

Definition 5.3.3:
    ℱ(s) = H(ι(s)) + δ* G_ℬ D* ι(s)

Theorem 5.3.4: ℱ is a Fredholm operator of index zero.
"""

from __future__ import annotations
from typing import Optional
import torch
from cdi.config import CDIConfig


class InferenceOperator:
    """Hodge-theoretic inference: replaces attention/softmax.

    v2.0: Fully differentiable forward path. No .detach() anywhere.
    The spectral operators (Hodge projection, Green's PCG) propagate
    gradients to all geometric and algebraic parameters.
    """

    def __init__(self, hodge, green, dirac, belief, sheaf, config: CDIConfig) -> None:
        self.hodge = hodge
        self.green = green
        self.dirac = dirac
        self.belief = belief
        self.sheaf = sheaf
        self.config = config

    # ------------------------------------------------------------------
    # Observation embedding into 𝔹
    # ------------------------------------------------------------------

    def embed_observation(self, data: torch.Tensor) -> torch.Tensor:
        """Map raw observation(s) into the full 𝔹 state space.

        Steps:
          1. Embed into ℬ₀ via ι: O → ℬ₀ (live embedding_matrix)
          2. Pad other degrees with zeros
          3. Tensor with spinor weight
          4. Flatten to (N,)

        Parameters
        ----------
        data : torch.Tensor  Shape (n, obs_dim) or (obs_dim,).

        Returns torch.Tensor  Shape (N,).
        """
        n = self.config.n_points
        s = self.config.spinor_dim
        B = self.config.total_belief_dim
        dtype = self.config.dtype

        if data.dim() == 1:
            data = data.unsqueeze(0).expand(n, -1)

        # 1. Embed into B_0 via live sheaf.embedding_matrix
        belief_0 = self.sheaf.embed(data)  # (n, dim_B0)

        # 2. Assemble full belief: other degrees zero
        sections = {}
        for k in self.belief.degrees:
            idx = self.belief.degree_to_index(k)
            dim_k = self.belief.dims[idx]
            if k == 0:
                sections[k] = belief_0
            else:
                sections[k] = torch.zeros(n, dim_k, dtype=dtype)
        full_belief = self.belief.assemble_state(sections)  # (n, B)

        # 3. Tensor with uniform spinor weight
        spinor_weight = torch.ones(s, dtype=dtype) / (s ** 0.5)
        full_twisted = full_belief.unsqueeze(1) * spinor_weight.unsqueeze(0).unsqueeze(-1)

        return full_twisted.reshape(-1)  # (N,)

    # ------------------------------------------------------------------
    # Inference — v2.0: FULLY DIFFERENTIABLE, no .detach()
    # ------------------------------------------------------------------

    def infer(self, observation: torch.Tensor) -> torch.Tensor:
        """Full inference ℱ(s) = H(ι(s)) + δ* G_ℬ D* ι(s).

        v2.0 Fix F1: Both harmonic_part and green_d_star are kept in
        the computation graph. Gradients flow through:
          - harmonic_part → Hodge projector → Laplacian matrix → all params
          - green_d_star  → Green PCG → Laplacian apply → all params
          - d_star_embedded → Dirac adjoint → Dirac matrix → all params

        Parameters
        ----------
        observation : torch.Tensor  Shape (n, obs_dim).

        Returns torch.Tensor  Shape (n, output_dim).
        """
        # Embed observation into 𝔹
        embedded = self.embed_observation(observation)  # (N,)

        # H(ι(s)) — harmonic projection via Hodge decomposition
        # v2.0: NO .detach() — fully in graph
        harmonic_part, _ = self.hodge.decompose(embedded)

        # D* ι(s)
        d_star_embedded = self.dirac.apply_adjoint(embedded)

        # G_ℬ D* ι(s) — PCG solve, fully differentiable
        # v2.0: NO .detach() — PCG propagates gradients through Laplacian
        green_d_star = self.green.apply(d_star_embedded)

        # δ* G_ℬ D* ι(s)
        delta_star_green = self._apply_delta_star_full(green_d_star)

        # Full inference result
        result = harmonic_part + delta_star_green  # (N,)

        return self.extract_prediction(result)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_prediction(self, full_state: torch.Tensor) -> torch.Tensor:
        """Extract ℬ₀ from full 𝔹 state, project to output_dim.

        Parameters
        ----------
        full_state : torch.Tensor  Shape (N,).

        Returns torch.Tensor  Shape (n, output_dim).
        """
        n = self.config.n_points
        s = self.config.spinor_dim
        B = self.config.total_belief_dim

        state_3d = full_state.reshape(n, s, B)
        state_avg = state_3d.mean(dim=1)  # (n, B)

        offset_0 = self.config.belief_offset(0)
        dim_0 = self.config.belief_dim(0)
        belief_0 = state_avg[:, offset_0:offset_0 + dim_0]  # (n, dim_B0)

        return self.sheaf.project_output(belief_0)  # (n, output_dim)

    # ------------------------------------------------------------------
    # δ* in the full space
    # ------------------------------------------------------------------

    def _apply_delta_star_full(self, state: torch.Tensor) -> torch.Tensor:
        """Apply δ* in the full 𝔹 = (n,s,B) space. Differentiable."""
        n = self.config.n_points
        s = self.config.spinor_dim
        B = self.config.total_belief_dim

        delta_star = self.belief.full_adjoint_coboundary_matrix()  # (B,B) live
        state_3d = state.reshape(n, s, B)
        result_3d = torch.einsum("ij,...j->...i", delta_star, state_3d)
        return result_3d.reshape(-1)

    def fredholm_index(self) -> int:
        return 0  # Theorem 5.3.4
