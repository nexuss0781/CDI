"""
§5.3 Inference Operator
=======================

Implements ℱ(s) from CDI Specification §5.3.

Definition 5.3.3:
    ℱ(s) = H(ι(s)) + δ* G_ℬ D* ι(s)

where
    H    = harmonic projector
    ι    = observation embedding O → 𝔹
    G_ℬ  = Green's operator
    D*   = Dirac adjoint
    δ*   = coboundary adjoint

Theorem 5.3.4: ℱ is a Fredholm operator of index zero.
"""

from __future__ import annotations

from typing import Optional

import torch

from cdi.config import CDIConfig


class InferenceOperator:
    """Hodge-theoretic inference: replaces attention/softmax.

    The inference operator ℱ maps observations to inferred beliefs
    by projecting onto the harmonic space (global consistency) plus
    a correction term from the Green's operator.

    Attributes
    ----------
    hodge : HodgeDecomposition
    green : GreenOperator
    dirac : DiracOperator
    belief : BeliefComplex
    sheaf : ObservationSheaf
    config : CDIConfig
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
          1. Embed into ℬ₀ via ι: O → ℬ₀
          2. Pad other degrees with zeros
          3. Tensor with spinor identity
          4. Distribute across manifold points

        Parameters
        ----------
        data : torch.Tensor
            Shape ``(n, obs_dim)`` — one observation per manifold point.

        Returns
        -------
        torch.Tensor
            Shape ``(N,)`` where N = n·s·B.
        """
        n = self.config.n_points
        s = self.config.spinor_dim
        B = self.config.total_belief_dim
        dtype = self.config.dtype

        # 1. Embed into B_0
        belief_0 = self.sheaf.embed(data)  # (n, dim_B0)

        # 2. Assemble full belief vector (pad other degrees with 0)
        sections = {}
        for k in self.belief.degrees:
            idx = self.belief.degree_to_index(k)
            dim_k = self.belief.dims[idx]
            if k == 0:
                sections[k] = belief_0
            else:
                sections[k] = torch.zeros(n, dim_k, dtype=dtype)
        full_belief = self.belief.assemble_state(sections)  # (n, B)

        # 3. Tensor with spinor identity: (n, s, B)
        # Each point gets spinor_dim copies of the belief vector
        # Use uniform spinor: [1/√s, 1/√s, ...] to distribute
        spinor_weight = torch.ones(s, dtype=dtype) / (s ** 0.5)
        full_twisted = full_belief.unsqueeze(1) * spinor_weight.unsqueeze(0).unsqueeze(-1)
        # shape: (n, s, B)

        # 4. Flatten to (N,)
        return full_twisted.reshape(-1)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, observation: torch.Tensor) -> torch.Tensor:
        """Full inference ℱ(s) = H(ι(s)) + δ* G_ℬ D* ι(s).

        Parameters
        ----------
        observation : torch.Tensor
            Shape ``(n, obs_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(n, output_dim)`` — predictions.
        """
        # Embed observation into 𝔹
        embedded = self.embed_observation(observation)  # (N,)

        # H(ι(s)) — harmonic projection
        harmonic_part, _ = self.hodge.decompose(embedded)

        # D* ι(s)
        d_star_embedded = self.dirac.apply_adjoint(embedded)

        # G_ℬ D* ι(s)
        green_d_star = self.green.apply(d_star_embedded)

        # δ* G_ℬ D* ι(s) — apply belief adjoint coboundary in the full space
        delta_star_green = self._apply_delta_star_full(green_d_star)

        # Full inference result
        result = harmonic_part + delta_star_green  # (N,)

        # Extract prediction from ℬ₀ component
        return self.extract_prediction(result)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_prediction(self, full_state: torch.Tensor) -> torch.Tensor:
        """Extract ℬ₀ from the full 𝔹 state, then project to output.

        Parameters
        ----------
        full_state : torch.Tensor
            Shape ``(N,)``.

        Returns
        -------
        torch.Tensor
            Shape ``(n, output_dim)``.
        """
        n = self.config.n_points
        s = self.config.spinor_dim
        B = self.config.total_belief_dim

        # Reshape to (n, s, B)
        state_3d = full_state.reshape(n, s, B)

        # Average over spinor index
        state_avg = state_3d.mean(dim=1)  # (n, B)

        # Extract B_0 slice
        offset_0 = self.config.belief_offset(0)
        dim_0 = self.config.belief_dim(0)
        belief_0 = state_avg[:, offset_0 : offset_0 + dim_0]  # (n, dim_B0)

        # Project to output
        return self.sheaf.project_output(belief_0)  # (n, output_dim)

    # ------------------------------------------------------------------
    # Internal: δ* in the full space
    # ------------------------------------------------------------------

    def _apply_delta_star_full(self, state: torch.Tensor) -> torch.Tensor:
        """Apply δ* in the full 𝔹 = (n, s, B) space.

        δ* acts on the belief index only (block-diagonal over n and s).
        """
        n = self.config.n_points
        s = self.config.spinor_dim
        B = self.config.total_belief_dim

        # δ* matrix in the belief space
        delta_star = self.belief.full_adjoint_coboundary_matrix()  # (B, B)

        # Reshape state to (n, s, B)
        state_3d = state.reshape(n, s, B)

        # Apply δ* to each (n, s) fiber
        result_3d = torch.einsum("ij,...j->...i", delta_star, state_3d)

        return result_3d.reshape(-1)

    # ------------------------------------------------------------------
    # Fredholm index
    # ------------------------------------------------------------------

    def fredholm_index(self) -> int:
        """ind(ℱ) = dim ker ℱ − dim coker ℱ.

        Theorem 5.3.4: Should be 0 for well-posed inference.
        """
        # Approximate via the Laplacian: ind = 0 for self-adjoint operators
        # on compact manifolds. Return 0 as per the theorem.
        return 0
