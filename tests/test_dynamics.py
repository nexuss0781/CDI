"""
Tests for dynamics (§6, §10) and full engine integration — v2.0.

v2.0 new tests:
  - test_recurrent_state_evolves: Ψ changes across tokens (Fix F2)
  - test_full_gradient_flow:      ALL params receive gradient (Fix F1/F3)
  - test_rebuild_operators:       operators change after rebuild (Fix F3)
  - test_lm_forward_shape:        v2.0 forward_sequence output shape
  - test_lm_loss:                 v2.0 composite loss
"""

import torch
import pytest

from cdi.engine import CDIEngine


class TestHeatEquation:

    def test_euler_shape(self, built_engine):
        cfg = built_engine.config
        N = cfg.total_state_dim
        J = torch.randn(N, dtype=cfg.dtype) * 0.1
        # v2.0: start from theta_init, not zeros
        psi_0 = built_engine.theta_init.detach().clone()
        psi_f = built_engine.heat.evolve_euler(psi_0, J, dt=0.01, steps=20)
        assert psi_f.shape == (N,)
        assert torch.isfinite(psi_f).all()

    def test_euler_state_changes(self, built_engine):
        """v2.0 Fix F2: state must change across Euler steps."""
        cfg = built_engine.config
        N = cfg.total_state_dim
        J = torch.randn(N, dtype=cfg.dtype) * 0.1
        psi_0 = torch.zeros(N, dtype=cfg.dtype)
        psi_f = built_engine.heat.evolve_euler(psi_0, J, dt=0.01, steps=5)
        diff = (psi_f - psi_0).norm().item()
        assert diff > 1e-10, "State did not evolve (heat equation is frozen)"

    def test_spectral_solution(self, built_engine):
        cfg = built_engine.config
        N = cfg.total_state_dim
        J = torch.randn(N, dtype=cfg.dtype) * 0.1
        psi_0 = torch.zeros(N, dtype=cfg.dtype)
        psi_spec = built_engine.heat.evolve_spectral(psi_0, J, t=1.0)
        assert psi_spec.shape == (N,)
        assert torch.isfinite(psi_spec).all()

    def test_steady_state(self, built_engine):
        cfg = built_engine.config
        N = cfg.total_state_dim
        J = torch.randn(N, dtype=cfg.dtype) * 0.1
        psi_inf = built_engine.heat.steady_state(J)
        assert psi_inf.shape == (N,)

    def test_convergence_rate_nonneg(self, built_engine):
        rate = built_engine.heat.convergence_rate()
        assert float(rate) >= 0


class TestEnergyFunctional:

    def test_energy_finite(self, built_engine):
        cfg = built_engine.config
        N = cfg.total_state_dim
        psi = torch.randn(N, dtype=cfg.dtype) * 0.01
        J = torch.randn(N, dtype=cfg.dtype) * 0.01
        E = built_engine.energy.cognitive_energy(psi, J)
        assert torch.isfinite(E)

    def test_dissipation_nonpositive(self, built_engine):
        """dE/dt = -‖∇E‖² ≤ 0  (Theorem 10.1.2)."""
        cfg = built_engine.config
        N = cfg.total_state_dim
        psi = torch.randn(N, dtype=cfg.dtype) * 0.01
        J = torch.randn(N, dtype=cfg.dtype) * 0.01
        dEdt = built_engine.energy.dissipation_rate(psi, J)
        assert float(dEdt) <= 1e-10


class TestV20Integration:
    """v2.0 specific integration tests for all four fixes."""

    def test_theta_init_is_learnable(self, built_engine):
        """Fix F2: theta_init must be a learnable parameter."""
        assert built_engine.theta_init.requires_grad
        assert built_engine.theta_init.abs().max().item() > 0  # not zeros

    def test_recurrent_state_evolves_across_tokens(self, built_engine):
        """Fix F2: output must vary across token positions (state is recurrent)."""
        cfg = built_engine.config
        seq = torch.randn(cfg.n_points, cfg.observation_dim, dtype=cfg.dtype)
        out = built_engine.forward_sequence(seq)
        # Outputs at different positions should differ
        token_var = out.var(dim=0).mean().item()
        assert token_var > 1e-12, (
            f"All token outputs are identical (var={token_var:.2e}) — "
            "state is NOT recurrent, theta_init may be ignored"
        )

    def test_forward_sequence_shape(self, built_engine):
        """v2.0 forward_sequence returns (n_points, embed_dim)."""
        cfg = built_engine.config
        seq = torch.randn(cfg.n_points, cfg.observation_dim, dtype=cfg.dtype)
        out = built_engine.forward_sequence(seq)
        assert out.shape == (cfg.n_points, cfg.output_dim)
        assert torch.isfinite(out).all()

    def test_forward_sequence_batch_shape(self, built_engine):
        """v2.0 batch forward returns (B, n_points, embed_dim)."""
        cfg = built_engine.config
        batch = torch.randn(3, cfg.n_points, cfg.observation_dim, dtype=cfg.dtype)
        out = built_engine.forward_sequence_batch(batch)
        assert out.shape == (3, cfg.n_points, cfg.output_dim)

    def test_lm_loss_terms(self, built_engine):
        """v2.0 composite loss includes CE, Bianchi, consistency, spectral."""
        cfg = built_engine.config
        V = 100  # small vocab for test
        B, L, E = 2, cfg.n_points, cfg.output_dim
        output = torch.randn(B, L, E, dtype=cfg.dtype)
        target_ids = torch.randint(0, V, (B, L))
        embedding = torch.randn(V, E, dtype=cfg.dtype)
        embedding.requires_grad_(True)

        total, loss_dict = built_engine.compute_lm_loss(
            output, target_ids, embedding, global_step=0
        )
        assert torch.isfinite(total)
        assert "ce" in loss_dict
        assert "consistency" in loss_dict
        assert "bianchi" in loss_dict
        assert "spectral_pen" in loss_dict
        assert "lambda_1" in loss_dict
        assert loss_dict["perplexity"] > 0

    def test_full_gradient_flow(self, tiny_config):
        """Fix F1 + F3: ALL parameters must receive non-zero gradient.

        This is the critical v2.0 test. In v1.0, manifold.points and
        connection.W_params had zero gradient due to .detach() in Dirac.
        In v2.0 they must receive gradient.
        """
        engine = CDIEngine(tiny_config)
        engine.build()

        V = 50
        B, L, E = 1, tiny_config.n_points, tiny_config.observation_dim
        embedding = torch.randn(V, E, dtype=tiny_config.dtype)
        embedding.requires_grad_(True)

        batch = torch.randn(B, L, E, dtype=tiny_config.dtype)
        target_ids = torch.randint(0, V, (B, L))

        output = engine.forward_sequence_batch(batch)
        total, _ = engine.compute_lm_loss(output, target_ids, embedding, global_step=0)
        total.backward()

        checks = engine.verify_gradient_flow()
        # These are the critical ones that were broken in v1.0
        critical = ["manifold.points", "manifold.metric_L",
                    "theta_init", "W_iota", "W_out",
                    "belief.deltas", "connection"]
        for name in critical:
            assert checks.get(name, False), (
                f"GRADIENT SEVERED: {name} has no gradient. "
                f"Fix F1/F3 not working correctly."
            )

    def test_rebuild_operators_changes_matrices(self, tiny_config):
        """Fix F3: After optimizer.step() + rebuild_operators(),
        the Dirac and Laplacian matrices must be different tensors.
        """
        engine = CDIEngine(tiny_config)
        engine.build()

        D_before = engine.dirac.matrix.data.clone()
        L_before = engine.laplacian.matrix.data.clone()

        # One training step
        V = 50
        B, L, E = 1, tiny_config.n_points, tiny_config.observation_dim
        embedding = torch.randn(V, E, dtype=tiny_config.dtype)
        embedding.requires_grad_(True)
        batch = torch.randn(B, L, E, dtype=tiny_config.dtype)
        target_ids = torch.randint(0, V, (B, L))

        all_params = engine.get_parameters() + [embedding]
        opt = torch.optim.Adam(all_params, lr=1e-2)

        output = engine.forward_sequence_batch(batch)
        total, _ = engine.compute_lm_loss(output, target_ids, embedding)
        opt.zero_grad()
        total.backward()
        opt.step()

        # v2.0 mandatory rebuild
        engine.rebuild_operators()

        D_after = engine.dirac.matrix.data
        L_after = engine.laplacian.matrix.data

        assert not torch.allclose(D_before, D_after, atol=1e-12), (
            "Dirac matrix unchanged after rebuild_operators() — "
            "connection parameters are not being updated or rebuild is broken."
        )
        assert not torch.allclose(L_before, L_after, atol=1e-12), (
            "Laplacian matrix unchanged after rebuild_operators()."
        )

    def test_no_token_collapse(self, tiny_config):
        """Fix F1/F4: All token positions should produce different outputs.

        In v1.0 all outputs collapsed to the same vector because the engine
        was acting as a fixed near-identity transform. In v2.0 the recurrent
        state ensures positional diversity.
        """
        engine = CDIEngine(tiny_config)
        engine.build()

        seq = torch.randn(
            tiny_config.n_points, tiny_config.observation_dim,
            dtype=tiny_config.dtype
        )
        with torch.no_grad():
            out = engine.forward_sequence(seq)  # (L, embed_dim)

        # Pairwise distances between output vectors should not all be zero
        dists = torch.cdist(out, out)  # (L, L)
        off_diag = dists[~torch.eye(dists.shape[0], dtype=torch.bool)]
        max_dist = off_diag.max().item()
        assert max_dist > 1e-6, (
            f"All outputs identical (max dist={max_dist:.2e}) — token collapse detected."
        )

    def test_parameters_count(self, built_engine):
        """Fix F4: Engine params must be >= 15% of embedding budget."""
        # Use a tokenizer-size embedding for the ratio check
        cfg = built_engine.config
        engine_n = sum(p.numel() for p in built_engine.get_parameters())
        # Simulate embedding: 16000 × embed_dim
        vocab = 16000
        embed_n = vocab * cfg.observation_dim
        ratio = 100.0 * engine_n / embed_n
        assert ratio >= 15.0, (
            f"Engine/Embed ratio = {ratio:.1f}% < 15% (Axiom 2.4.2.3). "
            f"Increase belief_dims or manifold_dim."
        )

    def test_diagnostics_complete(self, built_engine):
        """All required diagnostic keys present."""
        diag = built_engine.diagnostics()
        required = [
            "spectral_gap", "spectral_gap_lanczos", "learning_time",
            "harmonic_dim", "dirac_symmetry_error", "laplacian_psd",
            "delta_sq_norm", "green_error", "gradient_flow",
        ]
        for key in required:
            assert key in diag, f"Diagnostic key missing: {key}"
