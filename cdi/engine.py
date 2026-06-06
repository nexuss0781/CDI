"""
CDI Engine — Main Integration Layer
=====================================

Wires together all mathematical components into a single engine:

    Manifold → Cover → Sheaf → Belief → Clifford → Connection → Dirac
    → Laplacian → Hodge → Green → Inference → HeatEquation
    → Superconnection → FieldEquations → Invariants

Usage::

    config = CDIConfig.small()
    engine = CDIEngine(config)
    engine.build()

    prediction = engine.forward(input_data, target_data)
    loss = engine.compute_loss(prediction, target_data)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from cdi.config import CDIConfig
from cdi.core.manifold import CognitiveManifold
from cdi.core.cover import GoodCover
from cdi.core.sheaf import ObservationSheaf
from cdi.core.belief import BeliefComplex
from cdi.geometry.clifford import CliffordAlgebra
from cdi.geometry.connection import BeliefConnection
from cdi.geometry.dirac import DiracOperator
from cdi.operators.laplacian import BeliefLaplacian
from cdi.operators.hodge import HodgeDecomposition
from cdi.operators.green import GreenOperator
from cdi.operators.inference import InferenceOperator
from cdi.dynamics.heat_equation import HeatEquation
from cdi.dynamics.spectral import SpectralDecomposition
from cdi.dynamics.energy import EnergyFunctional
from cdi.topology.invariants import SystemInvariants
from cdi.field.superconnection import Superconnection
from cdi.field.field_equations import FieldEquations
from cdi.field.gauge import GaugeTransformation


class CDIEngine:
    """Cohomodynamic Intelligence engine.

    Orchestrates the full CDI pipeline: observation → inference → learning.

    No neural network layers.  All computation is explicit linear
    algebra, spectral methods, and heat-equation dynamics.

    Attributes
    ----------
    config : CDIConfig
    manifold : CognitiveManifold
    cover : GoodCover
    sheaf : ObservationSheaf
    belief : BeliefComplex
    clifford : CliffordAlgebra
    connection : BeliefConnection
    dirac : DiracOperator
    laplacian : BeliefLaplacian
    hodge : HodgeDecomposition
    green : GreenOperator
    inference_op : InferenceOperator
    heat : HeatEquation
    energy : EnergyFunctional
    superconn : Superconnection
    invariants : SystemInvariants
    """

    def __init__(self, config: CDIConfig) -> None:
        config.validate()
        self.config = config
        self._built = False

        # ── Core (§1-3) ──────────────────────────────────────────
        self.manifold = CognitiveManifold(config)
        self.cover = GoodCover(self.manifold, config)
        self.sheaf = ObservationSheaf(config)
        self.belief = BeliefComplex(config)

        # ── Geometry (§4) ────────────────────────────────────────
        self.clifford = CliffordAlgebra(config)
        self.connection = BeliefConnection(config, self.cover.edges)

        # ── Operators (§4-5) — require build() ───────────────────
        self.dirac: Optional[DiracOperator] = None
        self.laplacian: Optional[BeliefLaplacian] = None
        self.hodge: Optional[HodgeDecomposition] = None
        self.green: Optional[GreenOperator] = None
        self.inference_op: Optional[InferenceOperator] = None

        # ── Dynamics (§6, §10) ───────────────────────────────────
        self.heat: Optional[HeatEquation] = None
        self.spectral: Optional[SpectralDecomposition] = None
        self.energy: Optional[EnergyFunctional] = None

        # ── Field (§7) ──────────────────────────────────────────
        self.superconn: Optional[Superconnection] = None
        self.field_eqs: Optional[FieldEquations] = None
        self.gauge: Optional[GaugeTransformation] = None

        # ── Invariants (§11) ────────────────────────────────────
        self.invariants: Optional[SystemInvariants] = None

        # ── State ───────────────────────────────────────────────
        self.psi: Optional[torch.Tensor] = None  # current belief state

    # ==================================================================
    # Build
    # ==================================================================

    def build(self) -> "CDIEngine":
        """Construct all operator matrices.

        Must be called before forward / inference / learning.
        Call again after parameter updates if operators need refresh.
        """
        # Dirac
        self.dirac = DiracOperator(
            self.manifold, self.clifford, self.connection,
            self.belief, self.cover, self.config,
        )
        self.dirac.build()

        # Laplacian
        self.laplacian = BeliefLaplacian(
            self.dirac, self.belief, self.connection, self.config
        )
        self.laplacian.build()

        # Hodge & Green
        self.hodge = HodgeDecomposition(self.laplacian)
        self.green = GreenOperator(self.laplacian)

        # Inference
        self.inference_op = InferenceOperator(
            self.hodge, self.green, self.dirac,
            self.belief, self.sheaf, self.config,
        )

        # Dynamics
        self.heat = HeatEquation(self.laplacian, self.config)
        self.spectral = SpectralDecomposition(self.laplacian, self.config)
        self.energy = EnergyFunctional(self.laplacian, self.config)

        # Field
        self.superconn = Superconnection(
            self.dirac, self.belief, self.connection, self.config
        )
        self.field_eqs = FieldEquations(self.superconn, self.config)
        self.gauge = GaugeTransformation(self.config)

        # Invariants
        self.invariants = SystemInvariants(
            self.belief, self.laplacian, self.config
        )

        # Initial belief state
        N = self.config.total_state_dim
        self.psi = torch.zeros(N, dtype=self.config.dtype)

        self._built = True
        return self

    # ==================================================================
    # Forward pass: observe → infer → predict
    # ==================================================================

    def forward(
        self, input_data: torch.Tensor, target_data: torch.Tensor = None
    ) -> torch.Tensor:
        """Full forward pass.

        1. Distribute input across manifold points.
        2. Embed observations into 𝔹.
        3. Evolve belief state via heat equation.
        4. Infer using ℱ(s).
        5. Extract predictions.

        Parameters
        ----------
        input_data : torch.Tensor
            Shape ``(batch, obs_dim)``.
        target_data : torch.Tensor, optional
            Shape ``(batch, output_dim)`` — used for source term.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, output_dim)`` — predictions.
        """
        assert self._built, "Call engine.build() first."
        batch_size = input_data.shape[0]
        n = self.config.n_points
        obs_dim = self.config.observation_dim
        out_dim = self.config.output_dim
        dtype = self.config.dtype

        predictions = []
        for b in range(batch_size):
            x = input_data[b]  # (obs_dim,)

            # Distribute observation across manifold points
            # Each point sees the same observation (broadcast)
            obs_on_manifold = x.unsqueeze(0).expand(n, -1)  # (n, obs_dim)

            # Embed into full state
            J = self.inference_op.embed_observation(obs_on_manifold)

            # Heat equation: evolve from current state
            psi_evolved = self.heat.evolve_euler(
                self.psi, J,
                dt=self.config.heat_dt,
                steps=self.config.heat_steps,
            )

            # Infer
            pred_full = self.inference_op.infer(obs_on_manifold)

            # Combine heat evolution and inference
            combined = 0.5 * self.inference_op.extract_prediction(psi_evolved) + \
                       0.5 * pred_full

            # Average over manifold points → single prediction
            pred = combined.mean(dim=0)  # (output_dim,)
            predictions.append(pred)

            # Update belief state
            self.psi = psi_evolved.detach()

        return torch.stack(predictions, dim=0)

    # ==================================================================
    # Sequence forward: language modeling mode
    # ==================================================================

    def forward_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """Forward pass for language modeling.

        Each manifold point processes ONE token position.
        n_points = context_length.

        The Riemannian metric replaces positional encoding.
        The belief connection replaces attention (cross-position flow).
        The Dirac operator replaces feedforward layers.
        The heat equation provides convergent learning dynamics.

        Parameters
        ----------
        sequence : torch.Tensor
            Shape ``(n_points, embed_dim)`` — token embeddings at each position.

        Returns
        -------
        torch.Tensor
            Shape ``(n_points, output_dim)`` — output at each position.

        Complexity: O(n) heat steps + O(n) inference.
        
        Notes
        -----
        Each call creates a fresh computation graph. The psi state is reset
        to zero for each sequence to prevent graph accumulation.
        """
        assert self._built, "Call engine.build() first."
        n = self.config.n_points

        obs_on_manifold = sequence[:n]
        J = self.inference_op.embed_observation(obs_on_manifold)
        psi_init = torch.zeros(self.config.total_state_dim, dtype=self.config.dtype)
        psi_evolved = self.heat.evolve_euler(psi_init, J, dt=self.config.heat_dt, steps=self.config.heat_steps)
        pred_full = self.inference_op.infer(obs_on_manifold)
        state_pred = self.inference_op.extract_prediction(psi_evolved)
        combined = 0.5 * state_pred + 0.5 * pred_full
        self.psi = psi_evolved.detach()
        return combined

    def forward_sequence_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Batch of sequences for language modeling.

        Each batch item processes independently without sharing internal state (psi).
        This ensures each forward pass creates a fresh computation graph.

        Parameters
        ----------
        batch : torch.Tensor
            Shape ``(batch_size, n_points, embed_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size, n_points, output_dim)``.
        """
        outputs = []
        for b in range(batch.shape[0]):
            out = self.forward_sequence(batch[b])
            outputs.append(out)
        return torch.stack(outputs, dim=0)

    # ==================================================================
    # Language model loss (cross-entropy)
    # ==================================================================

    def compute_lm_loss(
        self,
        output: torch.Tensor,
        target_ids: torch.Tensor,
        embedding_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Cross-entropy loss for next-token prediction.

        output          : (batch, n_points, output_dim) — CDI output.
        target_ids      : (batch, n_points) — target token IDs.
        embedding_matrix: (vocab_size, embed_dim) — for logit projection.

        Loss = CE(output @ E^T, targets)
             + λ_c · ‖δ²‖²
             + λ_b · ‖Bianchi‖²
        
        CRITICAL: Regulariser terms depend on belief/connection parameters.
        They are computed with gradient tracking, but are detached for logging
        to avoid complications with rebuild_operators().
        """
        cfg = self.config

        # Logits via weight tying: (batch, n_points, vocab_size)
        logits = output @ embedding_matrix.T

        # Reshape for cross-entropy: (batch*n_points, vocab_size) vs (batch*n_points,)
        B, S, V = logits.shape
        logits_flat = logits.reshape(B * S, V)
        targets_flat = target_ids.reshape(B * S)

        # Cross-entropy — manual (no nn.CrossEntropyLoss)
        # log_softmax then negative log likelihood
        log_probs = logits_flat - logits_flat.logsumexp(dim=-1, keepdim=True)
        ce_loss = -log_probs[torch.arange(B * S), targets_flat].mean()

        # Mathematical regularisers
        # These are lightweight penalties on the belief structure itself,
        # not on the forward pass.  Keep them attached for gradients.
        consistency = self.belief.consistency_penalty()
        bianchi = self.connection.bianchi_penalty(self.cover.triangles)
        delta_full = self.belief.full_coboundary_matrix()
        compat = self.connection.compatibility_penalty(delta_full)

        total = (
            ce_loss
            + cfg.consistency_weight * consistency
            + cfg.bianchi_weight * (bianchi + compat)
        )

        # Perplexity (detach to avoid graph issues)
        perplexity = torch.exp(ce_loss.detach()).item()

        loss_dict = {
            "ce": ce_loss.detach().item(),
            "perplexity": perplexity,
            "consistency": consistency.detach().item(),
            "bianchi": bianchi.detach().item(),
            "total": total.detach().item(),
        }
        return total, loss_dict

    # ==================================================================
    # Loss computation
    # ==================================================================

    def compute_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss with mathematical regularisers.

        Loss = MSE(pred, target)
             + λ_c · ‖δ²‖²           (consistency)
             + λ_e · E[Ψ]            (energy)
             + λ_b · ‖d_A F_A‖²      (Bianchi)

        Returns
        -------
        (total_loss, loss_dict) : tuple
        """
        cfg = self.config

        # Prediction error
        mse = torch.mean((prediction - target) ** 2)

        # Consistency penalty: δ²=0  (Axiom 3.1.2)
        consistency = self.belief.consistency_penalty()

        # Bianchi identity penalty
        bianchi = self.connection.bianchi_penalty(self.cover.triangles)

        # Connection-delta compatibility
        delta_full = self.belief.full_coboundary_matrix()
        compat = self.connection.compatibility_penalty(delta_full)

        total = (
            mse
            + cfg.consistency_weight * consistency
            + cfg.bianchi_weight * (bianchi + compat)
        )

        loss_dict = {
            "mse": mse.item(),
            "consistency": consistency.item(),
            "bianchi": bianchi.item(),
            "compatibility": compat.item(),
            "total": total.item(),
        }

        return total, loss_dict

    # ==================================================================
    # Parameters
    # ==================================================================

    def get_parameters(self) -> List[torch.Tensor]:
        """All learnable parameters across the engine."""
        params = []
        params.extend(self.manifold.get_parameters())
        params.extend(self.sheaf.get_parameters())
        params.extend(self.belief.get_parameters())
        params.extend(self.connection.get_parameters())
        return params

    # ==================================================================
    # Rebuild operators (after parameter updates)
    # ==================================================================

    def rebuild_operators(self) -> None:
        """Rebuild Dirac, Laplacian, etc. after parameter changes.

        This is necessary because the operators depend on the
        learnable metric, connection, and coboundary maps.
        
        CRITICAL: Invalidate all caches to prevent old computation graphs
        from persisting when parameters change.
        """
        if not self._built:
            self.build()
            return

        # Rebuild cover (topology may change with point positions)
        self.cover = GoodCover(self.manifold, self.config)
        self.connection = BeliefConnection(self.config, self.cover.edges)

        # Rebuild operators — this creates NEW matrix instances
        self.dirac.invalidate()
        self.dirac = DiracOperator(
            self.manifold, self.clifford, self.connection,
            self.belief, self.cover, self.config
        )
        self.dirac.build()

        self.laplacian.invalidate()
        self.laplacian = BeliefLaplacian(
            self.dirac, self.belief, self.connection, self.config
        )
        self.laplacian.build()

        # Invalidate spectral decomposition cache — CRITICAL
        self.heat.invalidate_cache()

        self.hodge = HodgeDecomposition(self.laplacian)
        self.green = GreenOperator(self.laplacian)

        self.inference_op = InferenceOperator(
            self.hodge, self.green, self.dirac,
            self.belief, self.sheaf, self.config
        )

        # Rebuild heat equation to use new laplacian
        self.heat = HeatEquation(self.laplacian, self.config)

        self.superconn = Superconnection(
            self.dirac, self.belief, self.connection, self.config
        )

    # ==================================================================
    # Diagnostics
    # ==================================================================

    def diagnostics(self) -> Dict[str, object]:
        """Collect mathematical diagnostics and invariants."""
        assert self._built
        diag = {}

        # Spectral
        diag["spectral_gap"] = self.laplacian.spectral_gap().item()
        diag["learning_time"] = self.heat.learning_time().item()
        diag["harmonic_dim"] = self.hodge.harmonic_dimension()

        # Self-adjointness
        diag["dirac_symmetry_error"] = self.dirac.check_self_adjoint().item()
        diag["laplacian_symmetry_error"] = self.laplacian.check_self_adjoint().item()
        diag["laplacian_psd"] = self.laplacian.check_positive_semidefinite()

        # Consistency
        diag["delta_sq_norm"] = self.belief.consistency_penalty().item()

        # Green verification
        diag["green_error"] = self.green.verify().item()

        # Intelligence
        diag.update(self.invariants.summary())

        return diag
