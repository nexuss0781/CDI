from __future__ import annotations

import torch

from benchmarks.ethiobbpe_synaxarium_pilot import build_model, parameter_count
from cdi.v3.language_model import DCSSLanguageModel
from cdi.v3.ssm import StageCConfig
from cdi.v3.tokenizer import EthioBBPETokenizer, TokenizerConfig
from cdi.v3.training import StageDConfig, train_steps


def _models_and_ids():
    tokenizer = EthioBBPETokenizer.from_pretrained(TokenizerConfig(max_chunk_length=16, embedding_dim=4))
    candidate = DCSSLanguageModel(
        tokenizer,
        StageCConfig.nano(seed=11, token_residual_enabled=True, residual_fusion_enabled=True),
    )
    control = DCSSLanguageModel(
        tokenizer,
        StageCConfig.nano(
            seed=11,
            token_residual_enabled=True,
            residual_fusion_enabled=True,
            residual_fusion_ablation=True,
        ),
    )
    predecessor = DCSSLanguageModel(tokenizer, StageCConfig.nano(seed=11, token_residual_enabled=True))
    control.load_state_dict(candidate.state_dict(), strict=True)
    predecessor.load_state_dict(
        {name: value for name, value in candidate.state_dict().items() if not name.startswith("residual_fusion.")},
        strict=True,
    )
    ids = torch.tensor([[tokenizer.bos_id, 5, 6, 7, tokenizer.eos_id]], dtype=torch.long)
    return tokenizer, candidate, control, predecessor, ids


def _fusion_parameter_names(model: DCSSLanguageModel) -> frozenset[str]:
    return frozenset(
        name
        for name, parameter in model.named_parameters()
        if name.startswith("residual_fusion.") and parameter.requires_grad
    )


def test_fusion_control_exactly_reproduces_predecessor_and_retains_paths() -> None:
    _, candidate, control, predecessor, ids = _models_and_ids()
    candidate_logits = candidate.forward_chunk(ids, return_state=False)
    control_logits = control.forward_chunk(ids, return_state=False)
    predecessor_logits = predecessor.forward_chunk(ids, return_state=False)
    assert torch.equal(control_logits, predecessor_logits)
    assert float((candidate_logits - control_logits).abs().max().detach()) > 1e-8

    candidate_state = candidate.ssm.initial_state(batch_shape=(1,))
    control_state = control.ssm.initial_state(batch_shape=(1,))
    predecessor_state = predecessor.ssm.initial_state(batch_shape=(1,))
    for token_index in range(ids.shape[1] - 1):
        candidate_embedding = candidate.embedding(ids[:, token_index])
        control_embedding = control.embedding(ids[:, token_index])
        predecessor_embedding = predecessor.embedding(ids[:, token_index])
        assert torch.equal(candidate.token_residual(candidate_embedding, ablated=False), control.token_residual(control_embedding, ablated=False))
        assert torch.equal(candidate.token_residual(candidate_embedding, ablated=False), predecessor.token_residual(predecessor_embedding, ablated=False))
        _, candidate_state = candidate.ssm.step(candidate_embedding, candidate_state)
        _, control_state = control.ssm.step(control_embedding, control_state)
        _, predecessor_state = predecessor.ssm.step(predecessor_embedding, predecessor_state)
        assert all(torch.equal(left, right) for left, right in zip(candidate_state.tensors(), control_state.tensors()))
        assert all(torch.equal(left, right) for left, right in zip(candidate_state.tensors(), predecessor_state.tensors()))


def test_fusion_is_causal_and_meets_exact_inactive_gradient_contract() -> None:
    _, candidate, control, _, ids = _models_and_ids()
    future_changed = ids.clone()
    future_changed[:, -1] = 9
    candidate_logits = candidate.forward_chunk(ids, return_state=False)
    changed_logits = candidate.forward_chunk(future_changed, return_state=False)
    assert torch.equal(candidate_logits[:, :-1], changed_logits[:, :-1])

    candidate_report = candidate.causal_loss(ids)
    control_report = control.causal_loss(ids)
    assert float((candidate_report.logits - control_report.logits).abs().max().detach()) > 1e-8
    candidate_report.loss.backward()
    fusion_gradients = [
        parameter.grad
        for name, parameter in candidate.named_parameters()
        if name.startswith("residual_fusion.")
    ]
    assert all(gradient is not None and bool(torch.isfinite(gradient).all().item()) for gradient in fusion_gradients)
    assert any(float(torch.linalg.vector_norm(gradient).detach()) > 0.0 for gradient in fusion_gradients)

    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids, dtype=torch.bool)}
    config = StageDConfig(seed=11, chunk_length=ids.shape[1], batch_size=1, learning_rate=0.01)
    losses, _, _ = train_steps(control, [batch], config, steps=1)
    assert len(losses) == 1
    assert control.expected_inactive_trainable_parameters() == _fusion_parameter_names(control)
    for name in control.expected_inactive_trainable_parameters():
        gradient = dict(control.named_parameters())[name].grad
        assert gradient is None or torch.equal(gradient, torch.zeros_like(gradient))


def test_fusion_parameter_fairness_and_stability() -> None:
    tokenizer, candidate, control, predecessor, ids = _models_and_ids()
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids, dtype=torch.bool)}
    config = StageDConfig(seed=11, chunk_length=ids.shape[1], batch_size=1, learning_rate=0.01)
    losses, _, _ = train_steps(candidate, [batch], config, steps=1)
    assert torch.isfinite(torch.tensor(losses)).all()
    assert parameter_count(candidate) == parameter_count(control) == 80_586
    assert parameter_count(predecessor) == 80_550
    assert parameter_count(build_model("dcss_fused_residual_cdi", tokenizer, seed=11)) == 80_586
    assert parameter_count(build_model("dcss_fusion_control", tokenizer, seed=11)) == 80_586
