from __future__ import annotations

import torch

from benchmarks.ethiobbpe_synaxarium_pilot import build_model, parameter_count
from cdi.v3.language_model import DCSSLanguageModel
from cdi.v3.ssm import DynamicsDiagnostics, StageCConfig
from cdi.v3.tokenizer import EthioBBPETokenizer, TokenizerConfig
from cdi.v3.training import StageDConfig, train_steps


def _models_and_ids():
    tokenizer = EthioBBPETokenizer.from_pretrained(TokenizerConfig(max_chunk_length=16, embedding_dim=4))
    full = DCSSLanguageModel(tokenizer, StageCConfig.nano(seed=11))
    harmonic_disabled = DCSSLanguageModel(tokenizer, StageCConfig.nano(seed=11, harmonic_ablation=True))
    geometry_free = DCSSLanguageModel(tokenizer, StageCConfig.nano(seed=11, geometry_ablation=True))
    harmonic_disabled.load_state_dict(full.state_dict(), strict=True)
    geometry_free.load_state_dict(full.state_dict(), strict=True)
    ids = torch.tensor([[tokenizer.bos_id, 5, 6, 7, tokenizer.eos_id]], dtype=torch.long)
    return tokenizer, full, harmonic_disabled, geometry_free, ids


def _harmonic_parameter_names(model: DCSSLanguageModel) -> frozenset[str]:
    return frozenset(
        name
        for name, parameter in model.named_parameters()
        if name.startswith("ssm.cell.bands.harmonic.") and parameter.requires_grad
    )


def test_harmonic_control_is_first_class_and_zeros_harmonic_state_at_every_step() -> None:
    _, full, harmonic_disabled, _, ids = _models_and_ids()
    assert harmonic_disabled.config.harmonic_ablation is True
    assert harmonic_disabled.config.as_dict()["harmonic_ablation"] is True
    state = harmonic_disabled.ssm.initial_state(batch_shape=(1,), mode="learned")
    assert torch.equal(state.harmonic, torch.zeros_like(state.harmonic))
    for token_index in range(ids.shape[1] - 1):
        _, state = harmonic_disabled.ssm.step(harmonic_disabled.embedding(ids[:, token_index]), state)
        assert torch.equal(state.harmonic, torch.zeros_like(state.harmonic))
        assert state.fast.shape == state.middle.shape == state.harmonic.shape == (1, 4, 4)
    full_state = full.ssm.initial_state(batch_shape=(1,))
    _, full_state = full.ssm.step(full.embedding(ids[:, 0]), full_state)
    assert float(full_state.harmonic.abs().max().detach()) > 1e-8


def test_harmonic_control_is_causally_distinguishable_and_meets_gradient_contract() -> None:
    _, full, harmonic_disabled, _, ids = _models_and_ids()
    full_report = full.causal_loss(ids)
    control_report = harmonic_disabled.causal_loss(ids)
    assert full_report.logits.shape == control_report.logits.shape
    assert full_report.targets.shape == control_report.targets.shape
    assert torch.equal(full_report.loss_mask, control_report.loss_mask)
    assert torch.isfinite(control_report.loss)
    assert float((full_report.logits - control_report.logits).abs().max().detach()) > 1e-8

    full_report.loss.backward()
    full_harmonic_gradients = [
        parameter.grad
        for name, parameter in full.named_parameters()
        if name.startswith("ssm.cell.bands.harmonic.")
    ]
    assert any(
        gradient is not None and float(torch.linalg.vector_norm(gradient).detach()) > 0.0
        for gradient in full_harmonic_gradients
    )

    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids, dtype=torch.bool)}
    config = StageDConfig(seed=11, chunk_length=ids.shape[1], batch_size=1, learning_rate=0.01)
    losses, _, _ = train_steps(harmonic_disabled, [batch], config, steps=1)
    assert len(losses) == 1
    assert harmonic_disabled.expected_inactive_trainable_parameters() == _harmonic_parameter_names(harmonic_disabled)
    assert "ssm.cell.geometry.edge_log_weights" not in harmonic_disabled.expected_inactive_trainable_parameters()
    for name in harmonic_disabled.expected_inactive_trainable_parameters():
        gradient = dict(harmonic_disabled.named_parameters())[name].grad
        assert gradient is None or torch.equal(gradient, torch.zeros_like(gradient))
    assert harmonic_disabled.ssm.cell.geometry.edge_log_weights.grad is not None
    assert bool(torch.isfinite(harmonic_disabled.ssm.cell.geometry.edge_log_weights.grad).all().item())


def test_harmonic_control_preserves_state_safety_and_cdi_parameter_count() -> None:
    tokenizer, full, harmonic_disabled, geometry_free, ids = _models_and_ids()
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids, dtype=torch.bool)}
    config = StageDConfig(seed=11, chunk_length=ids.shape[1], batch_size=1, learning_rate=0.01)
    losses, _, _ = train_steps(harmonic_disabled, [batch], config, steps=1)
    assert torch.isfinite(torch.tensor(losses)).all()
    _, state = harmonic_disabled.forward_chunk(ids[:, :-1], return_state=True)
    norms = DynamicsDiagnostics.norms(state)
    energies = DynamicsDiagnostics.energy(state)
    assert norms["harmonic"] == 0.0
    assert energies["harmonic"] == 0.0
    assert all(torch.isfinite(torch.tensor(value)) for value in (*norms.values(), *energies.values()))
    assert parameter_count(full) == parameter_count(harmonic_disabled) == parameter_count(geometry_free)
    assert parameter_count(build_model("dcss_harmonic_disabled", tokenizer, seed=11)) == parameter_count(full)
