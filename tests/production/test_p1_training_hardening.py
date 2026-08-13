"""Offline P1 production-training hardening regression tests."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from cdi.v3 import (
    ArtifactLineage,
    DataManifest,
    EvaluationCard,
    GovernedDocument,
    LocalSyntheticCorpus,
    P1DataPolicy,
    ProductionRunConfig,
    ReleaseBoundary,
    StageDConfig,
    assert_core_optionality,
    build_envelope,
    build_model,
    checkpoint_payload,
    load_verified,
    optimizer_for,
    restore_checkpoint,
    save_atomic,
    train_steps,
)
from cdi.v3.production.lineage import assert_compatible
from cdi.v3.training import deterministic_batches, pack_documents, parameter_fingerprint, seed_everything


def _manifest() -> DataManifest:
    documents = [
        GovernedDocument("train", "synthetic train text", "local://synthetic/train", "CC0-1.0", "ephemeral"),
        GovernedDocument("validation", "synthetic validation text", "local://synthetic/validation", "CC0-1.0", "ephemeral"),
        GovernedDocument("test", "synthetic test text", "local://synthetic/test", "CC0-1.0", "ephemeral"),
    ]
    return DataManifest.build(documents, {"train": ["train"], "validation": ["validation"], "test": ["test"]})


def _resources(seed: int = 7):
    config = StageDConfig.nano(seed=seed)
    corpus = LocalSyntheticCorpus.default()
    tokenizer = corpus.tokenizer(config)
    split = corpus.split(seed)
    train_examples, _ = pack_documents(split["train"], tokenizer, config.chunk_length)
    batches = deterministic_batches(train_examples, tokenizer, config)
    return config, corpus, tokenizer, batches


def test_p1_config_is_strictly_offline_cpu_and_capability_free() -> None:
    config = ProductionRunConfig()
    config.validate()
    assert config.fingerprint
    with pytest.raises(ValueError):
        replace(config, external_side_effects_enabled=True).validate()
    with pytest.raises(ValueError):
        replace(config, capability_tools_enabled=True).validate()
    with pytest.raises(ValueError):
        replace(config, device="cuda").validate()
    with pytest.raises(ValueError):
        replace(ReleaseBoundary(), deployment_authorized=True).validate()


def test_governed_manifest_has_provenance_and_rejects_split_or_content_leakage() -> None:
    manifest = _manifest()
    manifest.assert_no_split_leakage()
    assert manifest.fingerprint
    duplicate = [
        GovernedDocument("a", "same", "local://a", "CC0", "ephemeral"),
        GovernedDocument("b", "same", "local://b", "CC0", "ephemeral"),
        GovernedDocument("c", "different", "local://c", "CC0", "ephemeral"),
    ]
    with pytest.raises(ValueError, match="Duplicate content"):
        DataManifest.build(duplicate, {"train": ["a"], "validation": ["b"], "test": ["c"]})
    real = [
        GovernedDocument("a", "one", "local://a", "unknown", "ephemeral", data_class="rights_cleared_pilot"),
        GovernedDocument("b", "two", "local://b", "unknown", "ephemeral"),
        GovernedDocument("c", "three", "local://c", "unknown", "ephemeral"),
    ]
    with pytest.raises(ValueError, match="not admitted"):
        DataManifest.build(real, {"train": ["a"], "validation": ["b"], "test": ["c"]}, P1DataPolicy())


def test_lineage_compatibility_is_explicit_and_fails_closed() -> None:
    lineage = ArtifactLineage("commit-test", "run", "corpus", "tokenizer", "model")
    assert lineage.fingerprint
    assert_compatible(lineage, lineage.as_dict())
    incompatible = dict(lineage.as_dict())
    incompatible["tokenizer_fingerprint"] = "different"
    with pytest.raises(ValueError, match="tokenizer"):
        assert_compatible(lineage, incompatible)


def test_atomic_checkpoint_integrity_and_lineage_round_trip(tmp_path: Path) -> None:
    config, corpus, tokenizer, batches = _resources()
    seed_everything(config.seed)
    model = build_model("dcss_cdi", tokenizer, config)
    optimizer = optimizer_for(model, config)
    _, optimizer, cursor = train_steps(model, batches, config, steps=2, optimizer=optimizer)
    stage_manifest = corpus.manifest(tokenizer, config)
    stage_d = checkpoint_payload(model, optimizer, tokenizer, stage_manifest, config, step=2, cursor=cursor)
    lineage = ArtifactLineage("commit-test", ProductionRunConfig().fingerprint, stage_manifest["fingerprint"], tokenizer.fingerprint, parameter_fingerprint(model))
    envelope = build_envelope(stage_d, lineage, ReleaseBoundary())
    path = tmp_path / "checkpoint.pt"
    written = save_atomic(path, envelope)
    assert Path(written["sidecar"]).is_file()
    restored = load_verified(path)
    assert restored["lineage_fingerprint"] == lineage.fingerprint
    with path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="integrity"):
        load_verified(path)


def test_deterministic_resume_reaches_identical_parameters(tmp_path: Path) -> None:
    config, corpus, tokenizer, batches = _resources(seed=11)
    total_steps, split_steps = 6, 3
    seed_everything(config.seed)
    uninterrupted = build_model("dcss_cdi", tokenizer, config)
    uninterrupted_optimizer = optimizer_for(uninterrupted, config)
    losses_full, _, cursor_full = train_steps(uninterrupted, batches, config, total_steps, uninterrupted_optimizer)
    full_fingerprint = parameter_fingerprint(uninterrupted)

    seed_everything(config.seed)
    interrupted = build_model("dcss_cdi", tokenizer, config)
    interrupted_optimizer = optimizer_for(interrupted, config)
    losses_first, interrupted_optimizer, cursor = train_steps(interrupted, batches, config, split_steps, interrupted_optimizer)
    stage_manifest = corpus.manifest(tokenizer, config)
    stage_d = checkpoint_payload(interrupted, interrupted_optimizer, tokenizer, stage_manifest, config, step=split_steps, cursor=cursor)
    lineage = ArtifactLineage("commit-test", ProductionRunConfig(seed=config.seed).fingerprint, stage_manifest["fingerprint"], tokenizer.fingerprint, parameter_fingerprint(interrupted))
    path = tmp_path / "resume.pt"
    save_atomic(path, build_envelope(stage_d, lineage, ReleaseBoundary()))

    resumed = build_model("dcss_cdi", tokenizer, config)
    resumed_optimizer = optimizer_for(resumed, config)
    restored = load_verified(path)["stage_d_payload"]
    step, restored_cursor = restore_checkpoint(
        restored,
        resumed,
        resumed_optimizer,
        tokenizer,
        expected_data_manifest=stage_manifest,
        expected_config=config,
    )
    assert (step, restored_cursor) == (split_steps, cursor)
    losses_second, _, cursor_resumed = train_steps(resumed, batches, config, total_steps - split_steps, resumed_optimizer, start_cursor=restored_cursor)
    assert cursor_full == cursor_resumed
    assert losses_full == losses_first + losses_second
    assert parameter_fingerprint(resumed) == full_fingerprint


def test_evaluation_card_and_core_optionality_are_bound_and_numerical() -> None:
    manifest = _manifest()
    card = EvaluationCard("synthetic-causal", "offline causal loss observation", manifest.fingerprint, ("loss", "perplexity"))
    assert card.fingerprint
    reference = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
    candidate = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
    assert assert_core_optionality(reference, candidate) == 0.0
    with pytest.raises(AssertionError):
        assert_core_optionality(reference, [torch.tensor([1.0, 2.0]), torch.tensor([3.1])], atol=1e-6)


def test_frozen_p1_config_and_readiness_harness(tmp_path: Path) -> None:
    from benchmarks.p1_readiness import run_all

    config = __import__("json").loads(Path("benchmarks/configs/p1_offline.json").read_text(encoding="utf-8"))
    assert config["phase"] == "P1"
    assert config["data_policy"]["real_corpus_training_authorized"] is False
    assert config["release_boundary"]["fine_tuning_authorized"] is False
    assert config["release_boundary"]["deployment_authorized"] is False
    report = run_all(tmp_path)
    assert report["status"] == "PASS"
    assert report["offline_only"] is True
    assert report["external_side_effects_enabled"] is False
    assert (tmp_path / "latest.json").is_file()
