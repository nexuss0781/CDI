"""Regression tests for verified DCSS-CDI production inference."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest
import torch

from cdi.v3 import (
    ArtifactLineage,
    EthioBBPETokenizer,
    DCSSInferenceEngine,
    DCSSLanguageModel,
    GenerationConfig,
    ProductionRunConfig,
    ReleaseBoundary,
    StageCConfig,
    StageDConfig,
    TokenizerConfig,
    build_envelope,
    checkpoint_payload,
    optimizer_for,
    parameter_fingerprint,
    save_atomic,
)


def _checkpoint(tmp_path: Path, *, embedding_dim: int = 4, n_vertices: int = 4, band_width: int = 4) -> Path:
    texts = ("alpha beta gamma", "delta epsilon zeta", "cohomodynamic language model")
    tokenizer = EthioBBPETokenizer.from_pretrained(
        TokenizerConfig(max_chunk_length=8, embedding_dim=embedding_dim)
    )
    stage_d = StageDConfig.nano(seed=17)
    stage_c = replace(
        StageCConfig.nano(seed=17),
        input_width=embedding_dim,
        output_width=embedding_dim,
        n_vertices=n_vertices,
        band_width=band_width,
    )
    stage_c.validate()
    model = DCSSLanguageModel(tokenizer, stage_c)
    optimizer = optimizer_for(model, stage_d)
    manifest = {
        "format": "test-manifest-v1",
        "tokenizer_fingerprint": tokenizer.fingerprint,
    }
    manifest["fingerprint"] = sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = checkpoint_payload(model, optimizer, tokenizer, manifest, stage_d, step=3, cursor=0)
    lineage = ArtifactLineage(
        code_revision="inference-regression-test",
        run_config_fingerprint=ProductionRunConfig(seed=17).fingerprint,
        corpus_manifest_fingerprint=manifest["fingerprint"],
        tokenizer_fingerprint=tokenizer.fingerprint,
        model_fingerprint=parameter_fingerprint(model),
    )
    path = tmp_path / "inference.pt"
    save_atomic(path, build_envelope(payload, lineage, ReleaseBoundary()))
    return path


def test_verified_checkpoint_loads_and_greedy_generation_is_deterministic(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    engine = DCSSInferenceEngine(checkpoint, device="cpu")
    config = GenerationConfig(max_new_tokens=12, mode="greedy")
    first = engine.generate_ids("alpha", config)
    second = engine.generate_ids("alpha", config)
    assert torch.equal(first, second)
    assert first.numel() >= 1
    assert engine.metadata.global_step == 3
    assert engine.metadata.device == "cpu"


def test_complete_excludes_the_supplied_prompt(tmp_path: Path) -> None:
    engine = DCSSInferenceEngine(_checkpoint(tmp_path), device="cpu")
    prompt = "alpha"
    config = GenerationConfig(max_new_tokens=4, mode="greedy")
    full_sequence = engine.generate(prompt, config)
    assert engine.complete(prompt, config) == full_sequence[len(engine.tokenizer.normalize(prompt)):]


def test_shape_derived_restoration_supports_valid_nondefault_nano_checkpoint(tmp_path: Path) -> None:
    engine = DCSSInferenceEngine(_checkpoint(tmp_path, embedding_dim=16, n_vertices=3, band_width=2), device="cpu")
    assert engine.stage_c_config.input_width == 16
    assert engine.stage_c_config.n_vertices == 3
    assert engine.stage_c_config.band_width == 2
    assert engine.generate("alpha", GenerationConfig(max_new_tokens=4, mode="greedy"))


def test_sampling_is_seed_reproducible_and_stateful(tmp_path: Path) -> None:
    engine = DCSSInferenceEngine(_checkpoint(tmp_path), device="cpu")
    config = GenerationConfig(max_new_tokens=12, mode="sample", temperature=0.8, top_k=8, top_p=0.9, seed=123)
    assert torch.equal(engine.generate_ids("delta", config), engine.generate_ids("delta", config))


def test_tampered_sidecar_or_lineage_is_rejected(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    sidecar = checkpoint.with_suffix(".pt.sha256")
    sidecar.write_text("format=dcss-cdi-production-checkpoint-v1\nsha256=bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        DCSSInferenceEngine(checkpoint, device="cpu")


def test_invalid_generation_controls_and_overlong_prompts_are_rejected(tmp_path: Path) -> None:
    engine = DCSSInferenceEngine(_checkpoint(tmp_path), device="cpu")
    with pytest.raises(ValueError, match="temperature"):
        engine.generate("alpha", GenerationConfig(temperature=0.0))
    with pytest.raises(ValueError, match="max_prompt_tokens"):
        engine.generate("alpha beta", GenerationConfig(max_prompt_tokens=1))


def test_special_tokens_are_never_sampled(tmp_path: Path) -> None:
    engine = DCSSInferenceEngine(_checkpoint(tmp_path), device="cpu")
    logits = torch.zeros(engine.tokenizer.vocab_size)
    candidate = engine._filtered_logits(logits, GenerationConfig())
    assert not torch.isfinite(candidate[engine.tokenizer.pad_id])
    assert not torch.isfinite(candidate[engine.tokenizer.bos_id])
    assert not torch.isfinite(candidate[engine.tokenizer.doc_id])
