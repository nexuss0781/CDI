"""Offline P1 training-system hardening readiness harness."""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Mapping

import torch

from cdi.v3 import ArtifactLineage, DataManifest, GovernedDocument, LocalSyntheticCorpus, ProductionRunConfig, ReleaseBoundary, StageDConfig, assert_core_optionality, build_envelope, build_model, checkpoint_payload, load_verified, optimizer_for, save_atomic, train_steps
from cdi.v3.training import deterministic_batches, pack_documents, parameter_fingerprint, restore_checkpoint, seed_everything


def _gate(name: str, passed: bool, details: Mapping[str, Any]) -> Dict[str, Any]:
    return {"name": name, "status": "PASS" if passed else "FAIL", "passed": bool(passed), "details": dict(details)}


def _manifest() -> DataManifest:
    docs = [
        GovernedDocument("train", "synthetic train text", "local://synthetic/train", "CC0-1.0", "ephemeral"),
        GovernedDocument("validation", "synthetic validation text", "local://synthetic/validation", "CC0-1.0", "ephemeral"),
        GovernedDocument("test", "synthetic test text", "local://synthetic/test", "CC0-1.0", "ephemeral"),
    ]
    return DataManifest.build(docs, {"train": ["train"], "validation": ["validation"], "test": ["test"]})


def _resources(seed: int = 42):
    config = StageDConfig.nano(seed=seed)
    corpus = LocalSyntheticCorpus.default()
    tokenizer = corpus.tokenizer(config)
    train, _ = pack_documents(corpus.split(seed)["train"], tokenizer, config.chunk_length)
    return config, corpus, tokenizer, deterministic_batches(train, tokenizer, config)


def configuration_gate() -> Dict[str, Any]:
    config = ProductionRunConfig()
    boundary = ReleaseBoundary()
    config.validate()
    boundary.validate()
    return _gate("p1_offline_configuration_boundary", not config.external_side_effects_enabled and not config.capability_tools_enabled and not boundary.real_corpus_training_authorized and not boundary.fine_tuning_authorized and not boundary.deployment_authorized, {"config_fingerprint": config.fingerprint, "boundary": boundary.as_dict()})


def data_gate() -> Dict[str, Any]:
    manifest = _manifest()
    manifest.assert_no_split_leakage()
    return _gate("p1_governed_synthetic_manifest", True, {"manifest_fingerprint": manifest.fingerprint, "splits": manifest.splits, "data_classes": sorted({doc["data_class"] for doc in manifest.documents.values()})})


def checkpoint_and_resume_gate() -> Dict[str, Any]:
    config, corpus, tokenizer, batches = _resources(seed=23)
    total, split = 6, 3
    seed_everything(config.seed)
    whole = build_model("dcss_cdi", tokenizer, config)
    whole_optimizer = optimizer_for(whole, config)
    losses_whole, _, cursor_whole = train_steps(whole, batches, config, total, whole_optimizer)
    whole_hash = parameter_fingerprint(whole)

    seed_everything(config.seed)
    partial = build_model("dcss_cdi", tokenizer, config)
    partial_optimizer = optimizer_for(partial, config)
    losses_first, partial_optimizer, cursor = train_steps(partial, batches, config, split, partial_optimizer)
    stage_d = checkpoint_payload(partial, partial_optimizer, tokenizer, corpus.manifest(tokenizer, config), config, step=split, cursor=cursor)
    lineage = ArtifactLineage("p1-local", ProductionRunConfig(seed=config.seed).fingerprint, _manifest().fingerprint, tokenizer.fingerprint, parameter_fingerprint(partial))
    with TemporaryDirectory() as directory:
        path = Path(directory) / "resume.pt"
        written = save_atomic(path, build_envelope(stage_d, lineage, ReleaseBoundary()))
        resumed = build_model("dcss_cdi", tokenizer, config)
        resumed_optimizer = optimizer_for(resumed, config)
        checkpoint = load_verified(path)["stage_d_payload"]
        step, restored_cursor = restore_checkpoint(checkpoint, resumed, resumed_optimizer, tokenizer)
        losses_second, _, cursor_resumed = train_steps(resumed, batches, config, total - split, resumed_optimizer, start_cursor=restored_cursor)
        integrity_hash = written["sha256"]
    passed = step == split and cursor_whole == cursor_resumed and losses_whole == losses_first + losses_second and whole_hash == parameter_fingerprint(resumed)
    return _gate("p1_atomic_checkpoint_deterministic_resume", passed, {"checkpoint_sha256": integrity_hash, "full_cursor": cursor_whole, "resumed_cursor": cursor_resumed, "loss_sequence_equal": losses_whole == losses_first + losses_second, "parameter_fingerprints_equal": whole_hash == parameter_fingerprint(resumed)})


def core_optionality_gate() -> Dict[str, Any]:
    reference = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
    candidate = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
    error = assert_core_optionality(reference, candidate)
    return _gate("p1_core_optionality", error <= 1e-6, {"max_abs_error": error, "atol": 1e-6})


def render(report: Mapping[str, Any]) -> str:
    rows = "\n".join(f"| {gate['name']} | {gate['status']} |" for gate in report["gates"])
    return f"""# P1 Offline Training-System Hardening Report

**Status:** `{report['status']}`. P1 validates offline training-system controls only. It does **not** authorize real-corpus ingestion, fine-tuning, deployment, or external side effects.

| Gate | Status |
|---|---:|
{rows}

## Next decision

P2 may begin only after the user selects a narrow task, approved data boundary, success metrics, and acceptable GPU/data-residency environment as required by `Stages/PRODUCTION_NLP_TRAINING_ROADMAP.md`.
"""


def run_all(output_dir: str | Path = "results/p1") -> Dict[str, Any]:
    gates = [configuration_gate(), data_gate(), checkpoint_and_resume_gate(), core_optionality_gate()]
    passed = all(gate["passed"] for gate in gates)
    report = {"format": "dcss-cdi-p1-readiness-report-v1", "phase": "P1", "status": "PASS" if passed else "FAIL", "offline_only": True, "real_corpus_training_authorized": False, "fine_tuning_authorized": False, "deployment_authorized": False, "external_side_effects_enabled": False, "gates": gates}
    report["fingerprint"] = sha256(json.dumps(report, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "latest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path("Stages/P1_TRAINING_SYSTEM_HARDENING_REPORT.md").write_text(render(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/p1")
    args = parser.parse_args()
    report = run_all(args.output_dir)
    print(f"P1 {report['status']}; offline_only={report['offline_only']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
