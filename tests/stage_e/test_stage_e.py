import torch

from benchmarks.stage_e import (
    configuration_audit_gate,
    long_context_gate,
    reproducibility_gate,
    scaling_gate,
    streaming_allocation_gate,
    train_matrix_gate,
)
from cdi.v3 import (
    DCSSLanguageModel,
    ExplicitEulerIntegrator,
    LocalSyntheticCorpus,
    StageDConfig,
    UngatedSelectiveGate,
    build_matrix_model,
)


def resources(seed: int = 1):
    config = StageDConfig.nano(seed=seed)
    corpus = LocalSyntheticCorpus.default()
    return config, corpus.tokenizer(config)


def test_stage_e_matrix_builds_and_isolates_each_named_difference():
    config, tokenizer = resources()
    models = {identifier: build_matrix_model(identifier, tokenizer, config) for identifier in ("T", "V2", "U", "G", "H", "E", "C", "F")}
    assert all(sum(parameter.numel() for parameter in model.parameters()) > 0 for model in models.values())
    assert isinstance(models["F"], DCSSLanguageModel)
    assert models["G"].ssm.config.geometry_ablation
    assert models["H"].ssm.cell.disable_harmonic
    assert models["C"].ssm.cell.unconstrained_cochain is not None
    assert all(isinstance(band.gate, UngatedSelectiveGate) for band in models["U"].ssm.cell.bands.values())
    assert all(isinstance(band.integrator, ExplicitEulerIntegrator) for band in models["E"].ssm.cell.bands.values())


def test_harmonic_ablation_zeroes_only_the_harmonic_state_band():
    config, tokenizer = resources()
    model = build_matrix_model("H", tokenizer, config)
    ids = torch.tensor([[tokenizer.bos_id, tokenizer.token_to_id["a"]]], dtype=torch.long)
    _, state = model.forward_chunk(ids)
    assert torch.equal(state.harmonic, torch.zeros_like(state.harmonic))
    assert state.fast.abs().sum().item() > 0.0


def test_configuration_audit_and_short_matched_training_pass():
    audit = configuration_audit_gate((1,))
    training = train_matrix_gate((1,), steps=2)
    assert audit["passed"]
    assert training["passed"]
    assert set(training["details"]["summary"]) == {"T", "V2", "U", "G", "H", "E", "C", "F"}


def test_scaling_and_streaming_allocation_guards_pass_at_nano_lengths():
    scaling = scaling_gate((8, 16, 32), seed=1)
    streaming = streaming_allocation_gate(seed=1, length=32)
    assert scaling["passed"]
    assert streaming["passed"]
    assert {record["persistent_state_bytes"] for record in scaling["details"]["records"]} == {192}
    assert streaming["details"]["runtime_dense_sequence_allocations"] == []


def test_long_context_negative_result_is_recorded_not_hidden():
    result = long_context_gate(seed=1, delay=8)
    assert result["name"] == "long_context_harmonic_retention"
    assert result["details"]["full_harmonic_norm"] >= result["details"]["no_harmonic_norm"]
    assert result["details"]["scope"] == "synthetic_memory_diagnostic"


def test_stage_e_fresh_reproducibility_gate():
    result = reproducibility_gate(seed=1, steps=8)
    assert result["passed"]
    assert result["details"]["loss_max_abs"] <= 1e-6
