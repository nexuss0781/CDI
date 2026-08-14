from __future__ import annotations

import torch

from benchmarks.ethiobbpe_synaxarium_pilot import build_model, parameter_count
from cdi.v3.language_model import DCSSLanguageModel
from cdi.v3.ssm import StageCConfig
from cdi.v3.tokenizer import EthioBBPETokenizer, TokenizerConfig
from cdi.v3.training import StageDConfig, train_steps


def _models_and_ids():
    tokenizer = EthioBBPETokenizer.from_pretrained(TokenizerConfig(max_chunk_length=16, embedding_dim=4))
    full = DCSSLanguageModel(tokenizer, StageCConfig.nano(seed=11))
    mean_control = DCSSLanguageModel(tokenizer, StageCConfig.nano(seed=11, contrast_readout_ablation=True))
    geometry_free = DCSSLanguageModel(tokenizer, StageCConfig.nano(seed=11, geometry_ablation=True))
    mean_control.load_state_dict(full.state_dict(), strict=True)
    geometry_free.load_state_dict(full.state_dict(), strict=True)
    ids = torch.tensor([[tokenizer.bos_id, 5, 6, 7, tokenizer.eos_id]], dtype=torch.long)
    return tokenizer, full, mean_control, geometry_free, ids


def test_mean_readout_control_zeros_only_contrast_feature_slots() -> None:
    _, full, mean_control, _, ids = _models_and_ids()
    state = full.ssm.initial_state(batch_shape=(1,))
    embedding = full.embedding(ids[:, 0])
    _, updated = full.ssm.step(embedding, state)
    full_features = full.ssm.cell._readout_features(updated)
    control_features = mean_control.ssm.cell._readout_features(updated)
    assert full_features.shape == control_features.shape == (1, 48)
    for start in (0, 16, 32):
        assert torch.equal(full_features[:, start : start + 4], control_features[:, start : start + 4])
        assert torch.equal(control_features[:, start + 4 : start + 16], torch.zeros_like(control_features[:, start + 4 : start + 16]))


def test_readout_control_is_causal_distinguishable_and_has_narrow_inactive_gradient() -> None:
    _, full, mean_control, _, ids = _models_and_ids()
    full_report = full.causal_loss(ids)
    control_report = mean_control.causal_loss(ids)
    assert full_report.logits.shape == control_report.logits.shape
    assert torch.isfinite(control_report.loss)
    assert float((full_report.logits - control_report.logits).abs().max().detach()) > 1e-8
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids, dtype=torch.bool)}
    config = StageDConfig(seed=11, chunk_length=ids.shape[1], batch_size=1, learning_rate=0.01)
    losses, _, _ = train_steps(mean_control, [batch], config, steps=1)
    assert len(losses) == 1
    assert mean_control.expected_inactive_trainable_parameters() == frozenset({"ssm.cell.geometry.edge_log_weights"})
    gradient = mean_control.ssm.cell.geometry.edge_log_weights.grad
    assert gradient is not None
    assert torch.equal(gradient, torch.zeros_like(gradient))


def test_g3_readout_control_preserves_cdi_parameter_count() -> None:
    tokenizer, full, mean_control, geometry_free, _ = _models_and_ids()
    assert parameter_count(full) == parameter_count(mean_control) == parameter_count(geometry_free)
    assert parameter_count(build_model("dcss_mean_readout_control", tokenizer, seed=11)) == parameter_count(full)
