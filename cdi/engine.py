"""
CDI Engine — v2.0
==================

v2.0 Spec Corrections (CDI_LM_v2_Technical_Specification.md §3, §4, §7):

  Fix F1: No .detach() in inference forward path — fully differentiable
  Fix F2: Recurrent belief state Ψ; learnable theta_init instead of zeros
  Fix F3: rebuild_operators() after every optimizer.step(); no stale caches
  Fix F4: Dimensional hierarchy enforced via CDIConfig.validate()

Architecture (v2.0 §3.1 / §7.1):
    token_ids → embed → J_t (observation current into B_0 slice)
    Ψ_0 = theta_init  (learnable, not zeros)
    for t = 1..L:
        J_t[b0_slice] = W_iota @ e_t        ← per-token current
        for k = 1..K:
            Ψ_t = Ψ_t - dt * Δ_ℬ Ψ_t + dt * J_t   ← Euler, live Δ_ℬ
        h_t = W_out @ Proj_B0(Ψ_t)          ← readout from B_0 slice
        logit_t = h_t @ E^T                  ← weight tying
    return [L, V]

No bypass path. v1.0's 0.5*state_pred + 0.5*pred_full is REMOVED.
Engine output is the sole prediction path (Spec §3.1).
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
    """Cohomodynamic Intelligence engine — v2.0.

    Key v2.0 additions vs v1.0
    ---------------------------
    theta_init  Learnable initial belief state (N,); replaces zeros (Fix F2)
    W_iota      Learnable observation injection map (dim_b0, embed_dim) (Fix F4)
    W_out       Learnable readout B_0 → embed_dim (Fix F4)
    b0_indices  Precomputed index tensor for fast B_0 extraction from flat Ψ

    Gradient connectivity (Fix F1/F3)
    ----------------------------------
    All operator matrices (Dirac, Laplacian) are built from live parameters.
    No .detach() in the forward path. rebuild_operators() must be called
    after every optimizer.step().
    """

    def __init__(self, config: CDIConfig) -> None:
        config.validate()
        self.config = config
        self._built = False
        self.global_step: int = 0

        # ── Core (§1-3) ──────────────────────────────────────────
        torch.manual_seed(config.seed)
        self.manifold = CognitiveManifold(config)
        self.cover = GoodCover(self.manifold, config)
        self.sheaf = ObservationSheaf(config)
        self.belief = BeliefComplex(config)

        # ── Geometry (§4) ────────────────────────────────────────
        self.clifford = CliffordAlgebra(config)
        self.connection = BeliefConnection(config, self.cover.edges)

        # ── v2.0 Learnable parameters ────────────────────────────
        dtype = config.dtype
        N = config.total_state_dim
        dim_b0 = config.belief_dim(0)
        embed_dim = config.observation_dim
        s = config.spinor_dim
        B = config.total_belief_dim
        sB = s * B

        # Fix F2 (Spec §2.2.3): learnable initial belief state
        self.theta_init: torch.Tensor = torch.randn(N, dtype=dtype) * 1e-4
        self.theta_init.requires_grad_(True)

        # Fix F4 (Spec §3.1 Step 2): dedicated observation injection
        scale_iota = (2.0 / (embed_dim + dim_b0)) ** 0.5
        self.W_iota: torch.Tensor = (
            torch.randn(dim_b0, embed_dim, dtype=dtype) * scale_iota
        )
        self.W_iota.requires_grad_(True)

        # Fix F4 (Spec §3.1 Step 4): dedicated readout
        scale_out = (2.0 / (dim_b0 + embed_dim)) ** 0.5
        self.W_out: torch.Tensor = (
            torch.randn(embed_dim, dim_b0, dtype=dtype) * scale_out
        )
        self.W_out.requires_grad_(True)

        # Precompute flat B_0 index tensor for extraction
        b0_off = config.belief_offset(0)
        b0_idx = []
        for p in range(config.n_points):
            for si in range(s):
                st = p * sB + si * B + b0_off
                b0_idx.extend(range(st, st + dim_b0))
        self.b0_indices: torch.Tensor = torch.tensor(b0_idx, dtype=torch.long)
        self.dim_b0: int = dim_b0
        self._sB: int = sB

        # ── Operators — built via build() ────────────────────────
        self.dirac: Optional[DiracOperator] = None
        self.laplacian: Optional[BeliefLaplacian] = None
        self.hodge: Optional[HodgeDecomposition] = None
        self.green: Optional[GreenOperator] = None
        self.inference_op: Optional[InferenceOperator] = None
        self.heat: Optional[HeatEquation] = None
        self.spectral: Optional[SpectralDecomposition] = None
        self.energy: Optional[EnergyFunctional] = None
        self.superconn: Optional[Superconnection] = None
        self.field_eqs: Optional[FieldEquations] = None
        self.gauge: Optional[GaugeTransformation] = None
        self.invariants: Optional[SystemInvariants] = None

        # Persisted recurrent state (detached, not for grad)
        self.psi: Optional[torch.Tensor] = None

    # ==================================================================
    # Build / Rebuild
    # ==================================================================

    def build(self) -> "CDIEngine":
        """Initial operator construction. Call once at startup."""
        self._build_operators()
        self._built = True
        return self

    def _build_operators(self) -> None:
        """Construct all operators from LIVE parameters. No .detach()."""
        cfg = self.config

        # Dirac — v2.0: live manifold points and frames
        self.dirac = DiracOperator(
            self.manifold, self.clifford, self.connection,
            self.belief, self.cover, cfg,
        )
        self.dirac.build()

        # Laplacian — v2.0: live Dirac, belief, connection
        self.laplacian = BeliefLaplacian(
            self.dirac, self.belief, self.connection, cfg
        )
        self.laplacian.build()

        # Hodge & Green — v2.0: no .detach() on outputs
        self.hodge = HodgeDecomposition(self.laplacian)
        self.green = GreenOperator(self.laplacian)

        # Inference — v2.0: fully differentiable
        self.inference_op = InferenceOperator(
            self.hodge, self.green, self.dirac,
            self.belief, self.sheaf, cfg,
        )

        # Dynamics
        self.heat = HeatEquation(self.laplacian, cfg)
        self.spectral = SpectralDecomposition(self.laplacian, cfg)
        self.energy = EnergyFunctional(self.laplacian, cfg)

        # Field theory
        self.superconn = Superconnection(
            self.dirac, self.belief, self.connection, cfg
        )
        self.field_eqs = FieldEquations(self.superconn, cfg)
        self.gauge = GaugeTransformation(cfg)

        # Topological invariants
        self.invariants = SystemInvariants(self.belief, self.laplacian, cfg)

    def rebuild_operators(self) -> None:
        """Rebuild all operators after optimizer.step().

        v2.0 Spec §2.3.2 Axiom 2.3.2.1 — MANDATORY after every step.
        Rebuilds cover topology then all dependent operators.
        Clears all spectral caches.

        Implements Algorithm 2.3.2.2 from the specification.
        """
        if not self._built:
            self.build()
            return

        # Rebuild cover (point positions may have shifted)
        self.cover = GoodCover(self.manifold, self.config)
        self.connection = BeliefConnection(self.config, self.cover.edges)

        # Rebuild all operators from live (updated) parameters
        self._build_operators()

    # ==================================================================
    # v2.0 Forward Pass — Recurrent Language Model (Spec §3.1 / §7.1)
    # ==================================================================

    def forward_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """Recurrent CDI forward over a token embedding sequence.

        v2.0 Changes (Spec §3.1):
          - Ψ starts from theta_init (learnable), NOT zeros            [Fix F2]
          - Ψ is carried across ALL L tokens without reset             [Fix F2]
          - K Euler steps per token using live Laplacian.apply()       [Fix F3]
          - Prediction via W_out @ B_0_mean(Ψ) — no bypass path       [Fix F4]
          - No .detach() on intermediate belief states                 [Fix F1]

        Parameters
        ----------
        sequence : (L, embed_dim) — one embedding per token position.

        Returns (L, embed_dim) — output representation per position.
        """
        assert self._built, "Call engine.build() first."
        cfg = self.config
        n = cfg.n_points
        s = cfg.spinor_dim
        B = cfg.total_belief_dim
        sB = self._sB
        N = cfg.total_state_dim
        dt = cfg.heat_dt
        K = cfg.heat_steps
        dtype = cfg.dtype
        b0_off = cfg.belief_offset(0)
        L = sequence.shape[0]

        # Fix F2: Start from learnable theta_init, not zeros
        psi = self.theta_init  # (N,) — in computation graph

        outputs = []

        for t in range(L):
            e_t = sequence[t]  # (embed_dim,)

            # Spec §3.1 Step 2: Observation current injected into B_0 slice
            # J_t[b0_indices] = W_iota @ e_t / (n*s), rest = 0
            b0_vals = self.W_iota @ e_t        # (dim_b0,) — differentiable

            # Build sparse-style J_t via index scatter into a zeros tensor
            # This keeps the grad path: b0_vals → W_iota, e_t → embedding
            J_t = torch.zeros(N, dtype=dtype)
            norm = float(n * s)
            for p in range(n):
                for si in range(s):
                    st = p * sB + si * B + b0_off
                    # Use index_put_ on a clone to stay in graph
                    J_t = J_t.clone()
                    J_t[st:st + self.dim_b0] = b0_vals / norm

            # Spec §3.1 Step 3: K Euler steps (recurrent, live Δ_ℬ)
            psi = self.heat.evolve_euler(psi, J_t, dt=dt, steps=K)

            # Spec §3.1 Step 4: Extract B_0 from Ψ, average over n*s slots
            b0_all = psi[self.b0_indices].reshape(n * s, self.dim_b0)
            b0_mean = b0_all.mean(dim=0)        # (dim_b0,)

            # Spec §3.1 Step 4: Readout W_out @ b0_mean → (embed_dim,)
            h_t = self.W_out @ b0_mean          # differentiable

            outputs.append(h_t)

        # Store last state (detached) to avoid cross-sequence graph growth
        self.psi = psi.detach()

        return torch.stack(outputs, dim=0)  # (L, embed_dim)

    def forward_sequence_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Batch wrapper around forward_sequence.

        Each sequence in the batch gets an independent unrolling of the
        recurrent graph starting from theta_init.

        Parameters
        ----------
        batch : (batch_size, L, embed_dim)

        Returns (batch_size, L, embed_dim)
        """
        outputs = [self.forward_sequence(batch[b]) for b in range(batch.shape[0])]
        return torch.stack(outputs, dim=0)

    def forward(self, input_data: torch.Tensor, target_data: torch.Tensor = None) -> torch.Tensor:
        """Non-LM forward pass for regression tasks (batch, obs_dim)."""
        assert self._built
        n = self.config.n_points
        predictions = []
        for b in range(input_data.shape[0]):
            obs = input_data[b].unsqueeze(0).expand(n, -1)
            J = self.inference_op.embed_observation(obs)
            psi = self.heat.evolve_euler(
                self.theta_init, J,
                dt=self.config.heat_dt, steps=self.config.heat_steps,
            )
            pred = self.inference_op.extract_prediction(psi)
            predictions.append(pred.mean(dim=0))
        return torch.stack(predictions, dim=0)

    # ==================================================================
    # Loss — v2.0 (Spec §4.1)
    # ==================================================================

    def compute_lm_loss(
        self,
        output: torch.Tensor,
        target_ids: torch.Tensor,
        embedding_matrix: torch.Tensor,
        global_step: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Composite LM loss (Spec §4.1):
            L = L_CE + λ_B·L_Bianchi + λ_C·L_consist + λ_S·L_spectral

        Parameters
        ----------
        output           : (B, L, embed_dim)
        target_ids       : (B, L) int64
        embedding_matrix : (V, embed_dim)
        global_step      : for consistency warm-up schedule
        """
        cfg = self.config

        # Vocab projection via weight tying
        logits = output @ embedding_matrix.T   # (B, L, V)
        Bs, S, V = logits.shape
        logits_flat = logits.reshape(Bs * S, V)
        targets_flat = target_ids.reshape(Bs * S)

        # Cross-entropy (manual log-softmax, no nn.functional)
        log_probs = logits_flat - logits_flat.logsumexp(dim=-1, keepdim=True)
        ce_loss = -log_probs[torch.arange(Bs * S), targets_flat].mean()

        # Consistency warm-up (Spec §4.1 Term 3)
        consistency = self.belief.consistency_penalty()
        lam_c = 1.0 if global_step < cfg.consistency_warmup_steps else cfg.consistency_weight

        # Bianchi penalty (Spec §4.1 Term 2)
        bianchi = self.connection.bianchi_penalty(self.cover.triangles)

        # Spectral gap penalty (Spec §4.1 Term 4)
        lam1_val = self.laplacian.lanczos_spectral_gap(max_iter=15)
        lam1_t = torch.tensor(lam1_val, dtype=cfg.dtype)
        lam_target = torch.tensor(cfg.spectral_target, dtype=cfg.dtype)
        spectral_pen = torch.clamp(lam_target - lam1_t, min=0.0) ** 2

        total = (
            ce_loss
            + lam_c * consistency
            + cfg.bianchi_weight * bianchi
            + cfg.spectral_weight * spectral_pen
        )

        perplexity = float(torch.exp(ce_loss.detach()).clamp(max=1e6).item())

        loss_dict = {
            "ce": ce_loss.detach().item(),
            "perplexity": perplexity,
            "consistency": consistency.detach().item(),
            "bianchi": bianchi.detach().item(),
            "spectral_pen": spectral_pen.detach().item(),
            "lambda_1": lam1_val,
            "total": total.detach().item(),
        }
        return total, loss_dict

    def compute_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """MSE + regularisers (non-LM mode)."""
        cfg = self.config
        mse = torch.mean((prediction - target) ** 2)
        consistency = self.belief.consistency_penalty()
        bianchi = self.connection.bianchi_penalty(self.cover.triangles)
        delta_full = self.belief.full_coboundary_matrix()
        compat = self.connection.compatibility_penalty(delta_full)
        total = (
            mse
            + cfg.consistency_weight * consistency
            + cfg.bianchi_weight * (bianchi + compat)
        )
        return total, {
            "mse": mse.item(), "consistency": consistency.item(),
            "bianchi": bianchi.item(), "compatibility": compat.item(),
            "total": total.item(),
        }

    # ==================================================================
    # Gradient flow verification (Spec §4.3)
    # ==================================================================

    def verify_gradient_flow(self) -> Dict[str, bool]:
        """Check all parameter groups received non-zero gradients.

        Spec §4.3 Verification Test 1. Returns {name: bool}.
        If any critical param returns False after step 1, halt training.
        """
        tol = 1e-8

        def _has_grad(p: torch.Tensor) -> bool:
            return p.grad is not None and p.grad.abs().max().item() > tol

        checks: Dict[str, bool] = {
            "manifold.points":  _has_grad(self.manifold.points),
            "manifold.metric_L": _has_grad(self.manifold.metric_L),
            "theta_init":       _has_grad(self.theta_init),
            "W_iota":           _has_grad(self.W_iota),
            "W_out":            _has_grad(self.W_out),
            "sheaf.embedding":  _has_grad(self.sheaf.embedding_matrix),
            "sheaf.output":     _has_grad(self.sheaf.output_matrix),
            "connection":       any(_has_grad(p) for p in self.connection.get_parameters()),
            "belief.deltas":    any(_has_grad(p) for p in self.belief.get_parameters()),
        }
        return checks

    # ==================================================================
    # Parameters
    # ==================================================================

    def get_parameters(self) -> List[torch.Tensor]:
        """All learnable parameters — v2.0 includes theta_init, W_iota, W_out."""
        params: List[torch.Tensor] = []
        params.extend(self.manifold.get_parameters())   # points, metric_L
        params.extend(self.sheaf.get_parameters())       # embedding_matrix, output_matrix
        params.extend(self.belief.get_parameters())      # delta coboundary maps
        params.extend(self.connection.get_parameters())  # W_params per edge
        params.append(self.theta_init)
        params.append(self.W_iota)
        params.append(self.W_out)
        return params

    # ==================================================================
    # Diagnostics
    # ==================================================================

    def diagnostics(self) -> Dict[str, object]:
        """Mathematical diagnostics — spectral, topological, geometric."""
        assert self._built
        diag: Dict[str, object] = {}

        diag["spectral_gap"] = self.laplacian.spectral_gap().item()
        diag["spectral_gap_lanczos"] = self.laplacian.lanczos_spectral_gap()
        diag["learning_time"] = self.heat.learning_time().item()
        diag["harmonic_dim"] = self.hodge.harmonic_dimension()
        diag["dirac_symmetry_error"] = self.dirac.check_self_adjoint().item()
        diag["laplacian_symmetry_error"] = self.laplacian.check_self_adjoint().item()
        diag["laplacian_psd"] = self.laplacian.check_positive_semidefinite()
        diag["delta_sq_norm"] = self.belief.consistency_penalty().item()
        diag["green_error"] = self.green.verify().item()
        diag["gradient_flow"] = self.verify_gradient_flow()
        diag.update(self.invariants.summary())
        return diag

    def recompute_spectral_gap(self) -> float:
        """λ₁ via Lanczos. Call every spectral_diag_every steps."""
        return self.laplacian.lanczos_spectral_gap(max_iter=20)
