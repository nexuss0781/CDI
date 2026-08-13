from __future__ import annotations

import torch

from cdi.v3.language_model import DCSSLanguageModel
from cdi.v3.ssm import StageCConfig
from cdi.v3.tokenizer import EthioBBPETokenizer, TokenizerConfig


LOGIT_EFFECT_EPS = 1e-8
LOSS_EFFECT_EPS = 1e-10
GRADIENT_EPS = 1e-12


def _models_and_ids():
    tokenizer = EthioBBPETokenizer.from_pretrained(TokenizerConfig(max_chunk_length=16, embedding_dim=4))
    full = DCSSLanguageModel(tokenizer, StageCConfig.nano(seed=11, geometry_ablation=False))
    disabled = DCSSLanguageModel(tokenizer, StageCConfig.nano(seed=11, geometry_ablation=True))
    disabled.load_state_dict(full.state_dict(), strict=True)
    ids = torch.tensor([[tokenizer.bos_id, 5, 6, 7, tokenizer.eos_id]], dtype=torch.long)
    return tokenizer, full, disabled, ids


def test_vertex_contrast_basis_is_zero_sum_and_orthonormal() -> None:
    _, full, _, _ = _models_and_ids()
    basis = full.ssm.cell.vertex_contrast_basis
    assert torch.allclose(basis.sum(dim=0), torch.zeros_like(basis.sum(dim=0)), atol=1e-7)
    assert torch.allclose(basis.transpose(0, 1) @ basis, torch.eye(basis.shape[1]), atol=1e-6, rtol=1e-6)
    assert full.ssm.cell.readout.in_features == 48


def test_geometry_reaches_causal_logits_loss_and_gradient() -> None:
    _, full, disabled, ids = _models_and_ids()
    full_report = full.causal_loss(ids)
    disabled_report = disabled.causal_loss(ids)
    logit_effect = (full_report.logits - disabled_report.logits).abs().max()
    loss_effect = (full_report.loss - disabled_report.loss).abs()
    assert float(logit_effect.detach()) > LOGIT_EFFECT_EPS
    assert float(loss_effect.detach()) > LOSS_EFFECT_EPS
    full.zero_grad(set_to_none=True)
    full_report.loss.backward()
    gradient = full.ssm.cell.geometry.edge_log_weights.grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    assert float(torch.linalg.vector_norm(gradient)) > GRADIENT_EPS


def test_geometry_ablation_is_exact_and_capacity_matched() -> None:
    _, full, disabled, ids = _models_and_ids()
    assert sum(parameter.numel() for parameter in full.parameters()) == sum(parameter.numel() for parameter in disabled.parameters())
    state = full.ssm.initial_state(batch_shape=(1,))
    assert torch.equal(disabled.ssm.cell.geometry.apply(state.fast), torch.zeros_like(state.fast))
    assert full.ssm.cell.geometry.apply(state.fast).shape == state.fast.shape
    assert full.causal_loss(ids).token_count == disabled.causal_loss(ids).token_count
