from __future__ import annotations

from dataclasses import replace

import pytest

from cdi.v3.training import (
    LocalSyntheticCorpus,
    StageDConfig,
    build_model,
    checkpoint_payload,
    optimizer_for,
    restore_checkpoint,
)


def _resources():
    config = StageDConfig.nano(seed=17)
    corpus = LocalSyntheticCorpus.default()
    tokenizer = corpus.tokenizer(config)
    manifest = corpus.manifest(tokenizer, config)
    source = build_model("dcss_cdi", tokenizer, config)
    optimizer = optimizer_for(source, config)
    payload = checkpoint_payload(source, optimizer, tokenizer, manifest, config, step=3, cursor=3)
    target = build_model("dcss_cdi", tokenizer, config)
    target_optimizer = optimizer_for(target, config)
    return config, tokenizer, manifest, payload, target, target_optimizer


def test_bound_checkpoint_restores_only_with_exact_contract() -> None:
    config, tokenizer, manifest, payload, target, target_optimizer = _resources()
    step, cursor = restore_checkpoint(
        payload,
        target,
        target_optimizer,
        tokenizer,
        expected_data_manifest=manifest,
        expected_config=config,
    )
    assert (step, cursor) == (3, 3)


def test_bound_checkpoint_rejects_configuration_or_manifest_mismatch() -> None:
    config, tokenizer, manifest, payload, target, target_optimizer = _resources()
    with pytest.raises(ValueError, match="configuration"):
        restore_checkpoint(
            payload,
            target,
            target_optimizer,
            tokenizer,
            expected_data_manifest=manifest,
            expected_config=replace(config, learning_rate=config.learning_rate / 2),
        )
    with pytest.raises(ValueError, match="data manifest"):
        restore_checkpoint(
            payload,
            target,
            target_optimizer,
            tokenizer,
            expected_data_manifest={"format": "different-manifest"},
            expected_config=config,
        )


def test_bound_checkpoint_rejects_internal_manifest_tampering() -> None:
    config, tokenizer, manifest, payload, target, target_optimizer = _resources()
    tampered = dict(payload)
    changed_manifest = dict(payload["data_manifest"])
    changed_manifest["source"] = "tampered"
    tampered["data_manifest"] = changed_manifest
    with pytest.raises(ValueError, match="fingerprint"):
        restore_checkpoint(
            tampered,
            target,
            target_optimizer,
            tokenizer,
            expected_data_manifest=manifest,
            expected_config=config,
        )
