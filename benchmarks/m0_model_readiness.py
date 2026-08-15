from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import resource
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, Mapping

import torch

from cdi.v3 import (
    ArtifactLineage,
    DCSSLanguageModel,
    EthioBBPETokenizer,
    LocalSyntheticCorpus,
    ProductionRunConfig,
    ReleaseBoundary,
    StageDConfig,
    build_envelope,
    build_model,
    checkpoint_payload,
    load_verified,
    optimizer_for,
    parameter_fingerprint,
    restore_checkpoint,
    save_atomic,
    train_steps,
)
from cdi.v3.training import deterministic_batches, model_loss, pack_documents, seed_everything


HARD_MEMORY_GIB = 11.0
OPERATING_MEMORY_GIB = 8.5


def peak_rss_gib() -> float:
    """Return Linux process peak RSS in GiB."""
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def _gate(name: str, fn: Callable[[], Mapping[str, Any]]) -> Dict[str, Any]:
    before = peak_rss_gib()
    try:
        details = dict(fn())
        after = peak_rss_gib()
        details["peak_rss_gib"] = max(before, after)
        return {"name": name, "status": "PASS", "passed": True, "details": details}
    except Exception as exc:  # The report must retain all failed gates for diagnosis.
        return {
            "name": name,
            "status": "FAIL",
            "passed": False,
            "details": {
                "peak_rss_gib": peak_rss_gib(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        }


def _resources(seed: int = 42):
    config = StageDConfig.nano(seed=seed)
    corpus = LocalSyntheticCorpus.default()
    tokenizer = corpus.tokenizer(config)
    splits = corpus.split(seed)
    examples, _ = pack_documents(splits["train"], tokenizer, config.chunk_length)
    batches = deterministic_batches(examples, tokenizer, config)
    manifest = corpus.manifest(tokenizer, config)
    return config, corpus, tokenizer, batches, manifest


def model_load_gate() -> Mapping[str, Any]:
    config, _, tokenizer, _, _ = _resources()
    seed_everything(config.seed)
    model = build_model("dcss_cdi", tokenizer, config)
    inventory = model.parameter_inventory()
    if inventory["total_parameters"] <= 0:
        raise AssertionError("The CDI model has no parameters.")
    artifact = tokenizer.artifact()
    restored_tokenizer = EthioBBPETokenizer.from_artifact(artifact)
    if restored_tokenizer.fingerprint != tokenizer.fingerprint:
        raise AssertionError("Tokenizer artifact did not reproduce its fingerprint.")
    tokenizer.assert_fingerprint(restored_tokenizer.fingerprint)
    return {
        "model_parameters": inventory["total_parameters"],
        "parameter_groups": inventory["groups"],
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_reload": "PASS",
    }


def forward_inference_gate() -> Mapping[str, Any]:
    config, _, tokenizer, batches, _ = _resources()
    seed_everything(config.seed)
    model = build_model("dcss_cdi", tokenizer, config)
    model.eval()
    batch = batches[0]
    with torch.no_grad():
        logits, state = model.forward_chunk(batch["input_ids"], attention_mask=batch["attention_mask"])
        prefix = batch["input_ids"][0, :3]
        generated = model.generate(prefix, max_new_tokens=2, mode="greedy")
    expected_shape = (*batch["input_ids"].shape, tokenizer.vocab_size)
    if tuple(logits.shape) != expected_shape:
        raise AssertionError(f"Unexpected logits shape: {tuple(logits.shape)} != {expected_shape}")
    if not bool(torch.isfinite(logits).all().item()):
        raise FloatingPointError("Forward logits contain non-finite values.")
    if not all(bool(torch.isfinite(tensor).all().item()) for tensor in state.tensors()):
        raise FloatingPointError("Forward state contains non-finite values.")
    perturbed = batch["input_ids"].clone()
    perturbed[:, -1] = tokenizer.unk_id
    with torch.no_grad():
        perturbed_logits, _ = model.forward_chunk(perturbed, attention_mask=batch["attention_mask"])
    causal_error = float((logits[:, :-1] - perturbed_logits[:, :-1]).abs().max().item())
    if causal_error > 1e-6:
        raise AssertionError(f"Future-token causality violation: max error {causal_error}")
    if generated.ndim != 1 or generated.numel() <= prefix.numel():
        raise AssertionError("Generation did not append new token IDs.")
    tokenizer.assert_ids_in_range(generated)
    return {
        "logits_shape": list(logits.shape),
        "state_tensors": len(state.tensors()),
        "causal_max_abs_error": causal_error,
        "generated_tokens": int(generated.numel() - prefix.numel()),
    }


def one_step_training_gate() -> Mapping[str, Any]:
    config, _, tokenizer, batches, _ = _resources()
    seed_everything(config.seed)
    model = build_model("dcss_cdi", tokenizer, config)
    optimizer = optimizer_for(model, config)
    before_fingerprint = parameter_fingerprint(model)
    before_state = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    declared_trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    losses, _, cursor = train_steps(model, batches, config, steps=1, optimizer=optimizer)
    after_fingerprint = parameter_fingerprint(model)
    if len(losses) != 1 or not bool(torch.isfinite(torch.tensor(losses)).all().item()):
        raise AssertionError("One-step training did not produce one finite loss.")
    if before_fingerprint == after_fingerprint:
        raise AssertionError("One-step training did not change the model parameters.")
    changed_names = {
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(before_state[name], parameter.detach())
    }
    frozen_changed = sorted(changed_names.difference(declared_trainable))
    if frozen_changed:
        raise AssertionError(f"Frozen or undeclared parameters changed: {frozen_changed}")
    if not changed_names.intersection(declared_trainable):
        raise AssertionError("No declared trainable parameter changed during the update.")
    return {
        "loss": losses[0],
        "cursor": cursor,
        "parameter_fingerprint_changed": True,
        "changed_parameter_count": len(changed_names),
        "frozen_parameters_changed": frozen_changed,
        "trainable_parameters": len(declared_trainable),
    }


def checkpoint_reload_gate(output_dir: Path) -> Mapping[str, Any]:
    config, corpus, tokenizer, batches, manifest = _resources(seed=23)
    seed_everything(config.seed)
    model = build_model("dcss_cdi", tokenizer, config)
    optimizer = optimizer_for(model, config)
    train_steps(model, batches, config, steps=1, optimizer=optimizer)
    probe = batches[0]
    with torch.no_grad():
        expected_logits, _ = model.forward_chunk(probe["input_ids"], attention_mask=probe["attention_mask"])
    model_hash = parameter_fingerprint(model)
    stage_payload = checkpoint_payload(model, optimizer, tokenizer, manifest, config, step=1, cursor=1)
    production_config = ProductionRunConfig(seed=config.seed)
    lineage = ArtifactLineage(
        code_revision="m0-local",
        run_config_fingerprint=production_config.fingerprint,
        corpus_manifest_fingerprint=manifest["fingerprint"],
        tokenizer_fingerprint=tokenizer.fingerprint,
        model_fingerprint=model_hash,
    )
    envelope = build_envelope(stage_payload, lineage, ReleaseBoundary())
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "m0_candidate.pt"
    written = save_atomic(artifact_path, envelope)
    loaded = load_verified(artifact_path)
    restored_tokenizer = EthioBBPETokenizer.from_artifact(loaded["stage_d_payload"]["tokenizer_artifact"])
    restored = build_model("dcss_cdi", restored_tokenizer, config)
    restored_optimizer = optimizer_for(restored, config)
    step, cursor = restore_checkpoint(
        loaded["stage_d_payload"],
        restored,
        restored_optimizer,
        restored_tokenizer,
        expected_data_manifest=manifest,
        expected_config=config,
    )
    with torch.no_grad():
        restored_logits, _ = restored.forward_chunk(probe["input_ids"], attention_mask=probe["attention_mask"])
    reload_error = float((expected_logits - restored_logits).abs().max().item())
    if parameter_fingerprint(restored) != model_hash:
        raise AssertionError("Reloaded model fingerprint differs from saved model.")
    if reload_error > 1e-6:
        raise AssertionError(f"Reloaded inference differs: max error {reload_error}")
    if step != 1 or cursor != 1:
        raise AssertionError(f"Unexpected checkpoint position: step={step}, cursor={cursor}")
    # Verify integrity rejection by changing the checkpoint bytes without changing
    # the sidecar. Restore the original bytes from the verified artifact afterward.
    original_bytes = artifact_path.read_bytes()
    try:
        artifact_path.write_bytes(original_bytes + b"tamper")
        try:
            load_verified(artifact_path)
        except ValueError:
            tamper_rejected = True
        else:
            tamper_rejected = False
    finally:
        artifact_path.write_bytes(original_bytes)
    if not tamper_rejected:
        raise AssertionError("Tampered checkpoint was not rejected by integrity verification.")
    return {
        "artifact_path": str(artifact_path),
        "artifact_sha256": written["sha256"],
        "tokenizer_fingerprint": restored_tokenizer.fingerprint,
        "global_step": step,
        "cursor": cursor,
        "reload_max_abs_error": reload_error,
        "tamper_rejected": tamper_rejected,
    }


def memory_budget_gate() -> Mapping[str, Any]:
    peak = peak_rss_gib()
    if peak > HARD_MEMORY_GIB:
        raise MemoryError(f"Peak RSS {peak:.3f} GiB exceeded hard limit {HARD_MEMORY_GIB:.1f} GiB.")
    return {
        "peak_rss_gib": peak,
        "hard_limit_gib": HARD_MEMORY_GIB,
        "operating_target_gib": OPERATING_MEMORY_GIB,
        "operating_target_passed": peak <= OPERATING_MEMORY_GIB,
    }


def render(report: Mapping[str, Any]) -> str:
    rows = "\n".join(
        f"| {gate['name']} | {gate['status']} | {gate['details'].get('peak_rss_gib', 0.0):.3f} |"
        for gate in report["gates"]
    )
    return f"""# CDI M0 Model and Data Readiness Report

**Status:** `{report['status']}`  
**Module:** `M0 — Model and Data Readiness`  
**Model:** `dcss_cdi`  
**Mode:** local CPU float32 validation  
**Report fingerprint:** `{report['fingerprint']}`

## Gates

| Gate | Status | Peak RSS GiB |
|---|---:|---:|
{rows}

## Promotion decision

The checkpoint may become the parent for M1 only when every gate is `PASS` and the artifact reload, tokenizer binding, inference, one-step training, causality, integrity, and memory contracts are all satisfied.
"""


def run_all(output_dir: str | Path = "results/m0") -> Dict[str, Any]:
    directory = Path(output_dir)
    gates = [
        _gate("m0_model_load_and_tokenizer_binding", model_load_gate),
        _gate("m0_forward_inference_and_causality", forward_inference_gate),
        _gate("m0_one_step_training_and_gradient_health", one_step_training_gate),
        _gate("m0_checkpoint_reload_and_integrity", lambda: checkpoint_reload_gate(directory)),
        _gate("m0_memory_budget", memory_budget_gate),
    ]
    passed = all(gate["passed"] for gate in gates)
    report: Dict[str, Any] = {
        "format": "dcss-cdi-m0-model-readiness-v1",
        "module": "M0",
        "status": "PASS" if passed else "FAIL",
        "model": "dcss_cdi",
        "hard_memory_limit_gib": HARD_MEMORY_GIB,
        "operating_memory_target_gib": OPERATING_MEMORY_GIB,
        "gates": gates,
    }
    report["fingerprint"] = sha256(json.dumps(report, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "latest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (directory / "REPORT.md").write_text(render(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CDI Module M0 model and data readiness gates.")
    parser.add_argument("--output-dir", default="results/m0")
    args = parser.parse_args()
    report = run_all(args.output_dir)
    print(f"M0 {report['status']}; gates={sum(gate['passed'] for gate in report['gates'])}/{len(report['gates'])}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
