"""Matched real-data CDI pilot using EthioBBPE and the Synaxarium corpus.

This is deliberately a bounded, CPU-safe architecture test.  It uses public
Amharic Synaxarium documents, splits at document level before tokenization, and
trains DCSS/CDI, a GRU baseline, and a causal Transformer under the same token
budget.  It is not a claim of production language-model quality.
"""
from __future__ import annotations

import argparse
import gc
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
from statistics import mean
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

import torch
from datasets import load_dataset

from cdi.v3 import (
    DCSSLanguageModel,
    EthioBBPETokenizer,
    LegacyCDIV2Adapter,
    P2DataPolicy,
    StageCConfig,
    StageDConfig,
    TinyTransformerBaseline,
    TokenizerConfig,
)
from cdi.v3.production.data import DataManifest, GovernedDocument
from cdi.v3.training import CorpusDocument, PackedExample, collate_examples, optimizer_for, seed_everything, train_steps


DATASET_ID = "Nexuss0781/synaxarium"
DATASET_URL = "https://huggingface.co/datasets/Nexuss0781/synaxarium"
DATASET_LICENSE = "MIT"
TEXT_COLUMN = "መጽሃፍ"
MONTH_COLUMN = "ወር"
DAY_COLUMN = "ቀን"
MODEL_NAMES = ("dcss_cdi", "gru_baseline", "transformer")


@dataclass(frozen=True)
class PilotConfig:
    """Fixed comparison budget for the empirical architecture pilot."""

    seeds: tuple[int, ...] = (11, 29, 47)
    steps: int = 30
    document_limit: int = 60
    chunks_per_document: int = 8
    chunk_length: int = 16
    batch_size: int = 2
    # Zero evaluates every held-out batch; a positive value fixes a bounded subset.
    eval_batches: int = 12
    shuffle_training_batches: bool = False
    learning_rate: float = 0.01
    relative_loss_tolerance: float = 0.10
    # None disables host-RSS enforcement. A positive value is interpreted as GiB.
    max_host_memory_gb: float | None = None
    output_dir: str = "results/ethiobbpe_synaxarium_pilot"

    def validate(self) -> None:
        if len(self.seeds) < 3 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("The proof pilot requires at least three unique seeds.")
        if self.steps <= 0 or self.document_limit < 12 or self.chunks_per_document <= 0:
            raise ValueError("Pilot steps, document_limit, and chunks_per_document must be positive; document_limit must be at least 12.")
        if self.chunk_length < 2 or self.batch_size <= 0 or self.eval_batches < 0:
            raise ValueError("Pilot chunk_length and batch_size must be positive; eval_batches must be zero or positive.")
        if not 0.0 < self.relative_loss_tolerance < 1.0:
            raise ValueError("relative_loss_tolerance must lie in (0, 1).")
        if self.max_host_memory_gb is not None and not self.max_host_memory_gb > 0.0:
            raise ValueError("max_host_memory_gb must be positive when set.")

    def as_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "seeds": list(self.seeds)}



class HostMemoryLimitExceeded(RuntimeError):
    """Raised before a configured host-RSS budget can end a Colab runtime."""


def host_resident_memory_bytes() -> int:
    """Return the conservative process/container resident-memory estimate in bytes."""

    process_rss = 0
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                fields = line.split()
                if len(fields) >= 2:
                    process_rss = int(fields[1]) * 1024
                    break
    cgroup_current = Path("/sys/fs/cgroup/memory.current")
    if cgroup_current.exists():
        try:
            return max(process_rss, int(cgroup_current.read_text(encoding="utf-8").strip()))
        except ValueError:
            pass
    if process_rss:
        return process_rss
    import resource

    # Linux reports KiB; macOS reports bytes. This fallback is a high-water mark,
    # which is intentionally conservative for a fail-closed safety threshold.
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if platform.system() != "Darwin" else value


class HostMemoryMonitor:
    """Measure and enforce an optional host-resident-memory limit in GiB."""

    def __init__(self, maximum_gb: float | None) -> None:
        self.maximum_bytes = None if maximum_gb is None else int(maximum_gb * (1024 ** 3))
        self.peak_bytes = 0
        self.last_bytes = 0

    def check(self, stage: str) -> int:
        current = host_resident_memory_bytes()
        self.last_bytes = current
        self.peak_bytes = max(self.peak_bytes, current)
        if self.maximum_bytes is not None and current >= self.maximum_bytes:
            raise HostMemoryLimitExceeded(
                f"Host RSS reached {current / (1024 ** 3):.3f} GiB at {stage}; "
                f"configured CCT limit is {self.maximum_bytes / (1024 ** 3):.3f} GiB."
            )
        return current

    def as_dict(self) -> Dict[str, float | None]:
        return {
            "maximum_gb": None if self.maximum_bytes is None else self.maximum_bytes / (1024 ** 3),
            "last_gb": self.last_bytes / (1024 ** 3),
            "peak_gb": self.peak_bytes / (1024 ** 3),
        }


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _stable_rank(identifier: str) -> str:
    return sha256(identifier.encode("utf-8")).hexdigest()


def load_governed_synaxarium(config: PilotConfig) -> tuple[list[GovernedDocument], DataManifest]:
    """Download only public corpus metadata/text and create a split-isolated manifest."""

    source = load_dataset(DATASET_ID, split="train")
    rows: list[tuple[str, str]] = []
    for row in source:
        text = str(row[TEXT_COLUMN]).strip()
        month = str(row[MONTH_COLUMN]).strip()
        day = int(row[DAY_COLUMN])
        identifier = f"synaxarium-{month}-{day:02d}"
        if text:
            rows.append((identifier, text))
    unique_rows: list[tuple[str, str]] = []
    seen_content_hashes: set[str] = set()
    for identifier, text in sorted(rows, key=lambda row: _stable_rank(row[0])):
        content_hash = sha256(text.encode("utf-8")).hexdigest()
        if content_hash not in seen_content_hashes:
            seen_content_hashes.add(content_hash)
            unique_rows.append((identifier, text))
        if len(unique_rows) >= config.document_limit:
            break
    if len(unique_rows) < config.document_limit:
        raise ValueError(f"Expected at least {config.document_limit} unique usable Synaxarium documents; found {len(unique_rows)}.")

    documents = [
        GovernedDocument(
            identifier=identifier,
            text=text,
            source_uri=f"{DATASET_URL}/viewer/default/train",
            license_id=DATASET_LICENSE,
            retention_policy="public_mit_research_pilot",
            data_class="rights_cleared_pilot",
            pii_review="public_historical_religious_text_no_personal_profile_data",
        )
        for identifier, text in unique_rows
    ]
    count = len(documents)
    train_end = int(count * 0.70)
    validation_end = train_end + int(count * 0.15)
    splits = {
        "train": [document.identifier for document in documents[:train_end]],
        "validation": [document.identifier for document in documents[train_end:validation_end]],
        "test": [document.identifier for document in documents[validation_end:]],
    }
    manifest = DataManifest.build(documents, splits, policy=P2DataPolicy())
    manifest.assert_no_split_leakage()
    return documents, manifest


def _split_documents(documents: Sequence[GovernedDocument], manifest: DataManifest) -> Dict[str, list[CorpusDocument]]:
    text_by_id = {document.identifier: document.text for document in documents}
    return {
        name: [CorpusDocument(identifier, text_by_id[identifier]) for identifier in identifiers]
        for name, identifiers in manifest.splits.items()
    }


def pack_limited_documents(
    documents: Sequence[CorpusDocument],
    tokenizer: EthioBBPETokenizer,
    chunk_length: int,
    chunks_per_document: int,
) -> list[PackedExample]:
    """Tokenize each document independently and retain a fixed causal budget."""

    examples: list[PackedExample] = []
    for document in documents:
        ids = tokenizer.encode(document.text, add_special_tokens=True).ids
        document_chunks = 0
        for start in range(0, len(ids), chunk_length):
            chunk = ids[start:start + chunk_length]
            if len(chunk) >= 2:
                examples.append(PackedExample(tuple(chunk), document.identifier))
                document_chunks += 1
                if document_chunks >= chunks_per_document:
                    break
    if not examples:
        raise ValueError("No causal chunks were produced from the selected real corpus documents.")
    return examples


def make_batches(
    examples: Sequence[PackedExample],
    tokenizer: EthioBBPETokenizer,
    config: PilotConfig,
) -> list[Dict[str, torch.Tensor]]:
    batches: list[Dict[str, torch.Tensor]] = []
    for start in range(0, len(examples), config.batch_size):
        rows = list(examples[start:start + config.batch_size])
        while len(rows) < config.batch_size:
            rows.append(examples[len(rows) % len(examples)])
        batches.append(collate_examples(rows, tokenizer, config.chunk_length))
    return batches


def build_model(name: str, tokenizer: EthioBBPETokenizer, seed: int) -> torch.nn.Module:
    if name == "dcss_cdi":
        return DCSSLanguageModel(tokenizer, StageCConfig.nano(seed=seed))
    if name == "dcss_geometry_free":
        return DCSSLanguageModel(tokenizer, StageCConfig.nano(seed=seed, geometry_ablation=True))
    if name == "dcss_mean_readout_control":
        return DCSSLanguageModel(tokenizer, StageCConfig.nano(seed=seed, contrast_readout_ablation=True))
    if name == "dcss_harmonic_disabled":
        return DCSSLanguageModel(tokenizer, StageCConfig.nano(seed=seed, harmonic_ablation=True))
    if name == "dcss_residual_cdi":
        return DCSSLanguageModel(tokenizer, StageCConfig.nano(seed=seed, token_residual_enabled=True))
    if name == "dcss_residual_control":
        return DCSSLanguageModel(
            tokenizer,
            StageCConfig.nano(seed=seed, token_residual_enabled=True, token_residual_ablation=True),
        )
    if name == "dcss_fused_residual_cdi":
        return DCSSLanguageModel(
            tokenizer,
            StageCConfig.nano(seed=seed, token_residual_enabled=True, residual_fusion_enabled=True),
        )
    if name == "dcss_fusion_control":
        return DCSSLanguageModel(
            tokenizer,
            StageCConfig.nano(
                seed=seed,
                token_residual_enabled=True,
                residual_fusion_enabled=True,
                residual_fusion_ablation=True,
            ),
        )
    if name == "gru_baseline":
        return LegacyCDIV2Adapter(tokenizer, width=tokenizer.config.embedding_dim, dtype=torch.float32)
    if name == "transformer":
        return TinyTransformerBaseline(tokenizer, width=tokenizer.config.embedding_dim, dtype=torch.float32)
    raise ValueError(f"Unknown model name: {name}")


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def evaluate_model(
    model: torch.nn.Module,
    batches: Sequence[Mapping[str, torch.Tensor]],
    maximum_batches: int,
    *,
    memory_check: Callable[[str], int] | None = None,
) -> Dict[str, float]:
    """Return token-weighted causal metrics over a fixed held-out batch budget."""

    previous_mode = model.training
    model.eval()
    loss_total = 0.0
    correct = 0
    tokens = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(batches[: len(batches) if maximum_batches == 0 else min(maximum_batches, len(batches))]):
            if memory_check is not None:
                memory_check(f"evaluation_batch_{batch_index}_before")
            report = model.causal_loss(batch["input_ids"], batch["attention_mask"])
            token_count = int(report.token_count)
            loss_total += float(report.loss.detach().cpu()) * token_count
            predictions = report.logits.argmax(dim=-1)
            correct += int(((predictions == report.targets) & report.loss_mask).sum().item())
            tokens += token_count
            if memory_check is not None:
                memory_check(f"evaluation_batch_{batch_index}_after")
    if previous_mode:
        model.train()
    if tokens <= 0:
        raise ValueError("Evaluation produced no active tokens.")
    loss = loss_total / tokens
    return {
        "loss": loss,
        "perplexity": float(torch.exp(torch.tensor(loss)).item()),
        "token_accuracy": correct / tokens,
        "token_count": float(tokens),
    }


def run_one(
    name: str,
    seed: int,
    tokenizer: EthioBBPETokenizer,
    train_batches: Sequence[Mapping[str, torch.Tensor]],
    validation_batches: Sequence[Mapping[str, torch.Tensor]],
    test_batches: Sequence[Mapping[str, torch.Tensor]],
    config: PilotConfig,
    *,
    memory_monitor: HostMemoryMonitor | None = None,
    post_training_diagnostics: Callable[[torch.nn.Module, Mapping[str, torch.Tensor], Callable[[str], int] | None], Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    if memory_monitor is not None:
        memory_monitor.check(f"{name}_seed_{seed}_before_model")
    seed_everything(seed)
    model = build_model(name, tokenizer, seed)
    train_config = StageDConfig(
        seed=seed,
        chunk_length=config.chunk_length,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
    )
    train_config.validate()
    memory_check = None if memory_monitor is None else memory_monitor.check
    initial_validation = evaluate_model(model, validation_batches, config.eval_batches, memory_check=memory_check)
    started = time.perf_counter()
    losses, _, _ = train_steps(
        model,
        train_batches,
        train_config,
        steps=config.steps,
        shuffle_each_epoch=config.shuffle_training_batches,
        memory_check=memory_check,
    )
    elapsed_seconds = time.perf_counter() - started
    final_validation = evaluate_model(model, validation_batches, config.eval_batches, memory_check=memory_check)
    held_out_test = evaluate_model(model, test_batches, config.eval_batches, memory_check=memory_check)
    diagnostics = None
    if post_training_diagnostics is not None:
        if not validation_batches:
            raise ValueError("Post-training diagnostics require at least one held-out validation batch.")
        diagnostics = post_training_diagnostics(model, validation_batches[0], memory_check)
    return {
        "seed": seed,
        "model": name,
        "parameter_count": parameter_count(model),
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "train_loss_decreased": bool(losses[-1] < losses[0]),
        "initial_validation": initial_validation,
        "validation": final_validation,
        "test": held_out_test,
        "elapsed_seconds": elapsed_seconds,
        "tokens_processed": config.steps * config.batch_size * (config.chunk_length - 1),
        "training_batch_order": "deterministic_per_epoch_shuffle" if config.shuffle_training_batches else "fixed_cyclic_order",
        "evaluation_scope": "all_held_out_batches" if config.eval_batches == 0 else f"first_{config.eval_batches}_held_out_batches",
        "host_memory": None if memory_monitor is None else memory_monitor.as_dict(),
        "post_training_diagnostics": diagnostics,
    }


def aggregate(records: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["model"]), []).append(record)
    summary: Dict[str, Dict[str, float]] = {}
    for name, entries in grouped.items():
        summary[name] = {
            "runs": float(len(entries)),
            "parameter_count": float(entries[0]["parameter_count"]),
            "mean_validation_loss": mean(float(entry["validation"]["loss"]) for entry in entries),
            "mean_validation_perplexity": mean(float(entry["validation"]["perplexity"]) for entry in entries),
            "mean_validation_accuracy": mean(float(entry["validation"]["token_accuracy"]) for entry in entries),
            "mean_test_loss": mean(float(entry["test"]["loss"]) for entry in entries),
            "mean_test_perplexity": mean(float(entry["test"]["perplexity"]) for entry in entries),
            "mean_elapsed_seconds": mean(float(entry["elapsed_seconds"]) for entry in entries),
            "mean_tokens_per_second": mean(float(entry["tokens_processed"]) / max(float(entry["elapsed_seconds"]), 1e-9) for entry in entries),
            "all_train_loss_decreased": float(all(bool(entry["train_loss_decreased"]) for entry in entries)),
        }
    return summary


def _record_values_are_finite(record: Mapping[str, Any]) -> bool:
    values = [
        record.get("train_loss_first"),
        record.get("train_loss_last"),
        record.get("elapsed_seconds"),
        record.get("tokens_processed"),
    ]
    for split_name in ("initial_validation", "validation", "test"):
        split = record.get(split_name, {})
        if not isinstance(split, Mapping):
            return False
        values.extend(split.get(name) for name in ("loss", "perplexity", "token_accuracy", "token_count"))
    return all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values)


def architecture_decision(
    summary: Mapping[str, Mapping[str, float]],
    config: PilotConfig,
    records: Sequence[Mapping[str, Any]],
    *,
    required_models: Sequence[str] = MODEL_NAMES,
    primary_model_name: str = "dcss_cdi",
) -> Dict[str, Any]:
    """Apply the complete CCT transition gate to the seed-level pilot evidence.

    A mean-only best-baseline comparison is insufficient: CDI must learn with
    finite values, stay within the declared mean Transformer-loss tolerance,
    and match or beat GRU validation loss in every declared seed.
    """
    if primary_model_name not in summary:
        raise ValueError(f"Primary CDI model is absent from summary: {primary_model_name}")
    dcss = summary[primary_model_name]
    transformer = summary["transformer"]
    gru = summary["gru_baseline"]
    transformer_gap = (dcss["mean_validation_loss"] / transformer["mean_validation_loss"]) - 1.0
    gru_gap = (dcss["mean_validation_loss"] / gru["mean_validation_loss"]) - 1.0
    by_seed: Dict[int, Dict[str, Mapping[str, Any]]] = {}
    for record in records:
        seed = record.get("seed")
        model = record.get("model")
        if isinstance(seed, int) and isinstance(model, str):
            by_seed.setdefault(seed, {})[model] = record
    required_models = set(required_models)
    expected_seeds = set(config.seeds)
    complete_seed_matrix = set(by_seed) == expected_seeds and all(set(rows) == required_models for rows in by_seed.values())
    finite = complete_seed_matrix and all(_record_values_are_finite(record) for rows in by_seed.values() for record in rows.values())
    learning = finite and all(bool(by_seed[seed][primary_model_name].get("train_loss_decreased")) for seed in config.seeds)
    transformer_tolerance = finite and transformer_gap <= config.relative_loss_tolerance
    per_seed_gru = {
        str(seed): float(by_seed[seed][primary_model_name]["validation"]["loss"]) <= float(by_seed[seed]["gru_baseline"]["validation"]["loss"])
        for seed in config.seeds
    } if finite else {}
    gru_gate = finite and bool(per_seed_gru) and all(per_seed_gru.values())
    passed = learning and transformer_tolerance and gru_gate
    return {
        "primary_model_name": primary_model_name,
        "finite_values_gate": finite,
        "complete_seed_matrix_gate": complete_seed_matrix,
        "learning_gate": learning,
        "transformer_tolerance_gate": transformer_tolerance,
        "gru_per_seed_gate": gru_gate,
        "gru_per_seed": per_seed_gru,
        "dcss_transformer_relative_validation_loss_gap": transformer_gap,
        "dcss_gru_relative_validation_loss_gap": gru_gap,
        "tolerance": config.relative_loss_tolerance,
        "verdict": "EARNED_NEXT_PILOT" if passed else "REDESIGN_BEFORE_SCALE",
        "scope": "This CCT verdict applies only to the current configuration, governed manifest, and declared bounded token budget.",
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    rows = []
    for name, values in report["summary"].items():
        rows.append(
            "| {name} | {params:,.0f} | {val_loss:.4f} | {val_ppl:.1f} | {test_loss:.4f} | {speed:.1f} |".format(
                name=name,
                params=values["parameter_count"],
                val_loss=values["mean_validation_loss"],
                val_ppl=values["mean_validation_perplexity"],
                test_loss=values["mean_test_loss"],
                speed=values["mean_tokens_per_second"],
            )
        )
    decision = report["decision"]
    return f"""# EthioBBPE Synaxarium Matched CDI Pilot

**Verdict:** `{decision['verdict']}`. This is a bounded, real-data architecture pilot, not a production language-model claim.

The pilot used document-isolated Amharic readings from [`{DATASET_ID}`]({DATASET_URL}), tokenized by the exact EthioBBPE artifact recorded in the result JSON. It trained each model for the same number of causal token positions under the same three seeds.

| Model | Parameters | Mean validation loss | Mean validation perplexity | Mean test loss | Mean tokens/sec |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Decision gates

| Gate | Result | Evidence |
|---|---|---|
| Complete finite evidence | {'PASS' if decision['finite_values_gate'] else 'FAIL'} | Three seeds and all required model records are present with finite metrics. |
| Learning | {'PASS' if decision['learning_gate'] else 'FAIL'} | CDI training loss decreased in every seed. |
| Transformer tolerance | {'PASS' if decision['transformer_tolerance_gate'] else 'FAIL'} | CDI-to-Transformer mean validation-loss gap: `{decision['dcss_transformer_relative_validation_loss_gap']:.2%}`; tolerance: `{decision['tolerance']:.0%}`. |
| GRU relation | {'PASS' if decision['gru_per_seed_gate'] else 'FAIL'} | CDI matches or beats GRU in every seed: `{decision['gru_per_seed']}`; mean gap: `{decision['dcss_gru_relative_validation_loss_gap']:.2%}`. |
| Split isolation | PASS | The governed manifest's document and content-hash leakage checks passed before training. |

> {decision['scope']}

## Reproducibility

| Field | Value |
|---|---|
| Dataset | `{DATASET_ID}` |
| Dataset license asserted by source card | `{DATASET_LICENSE}` |
| Tokenizer | `EthioBBPE` artifact fingerprint `{report['tokenizer_fingerprint']}` |
| Seeds | `{report['config']['seeds']}` |
| Training steps per model/seed | `{report['config']['steps']}` |
| Causal token positions per model/seed | `{report['records'][0]['tokens_processed']}` |
| Training batch order | `{report['records'][0]['training_batch_order']}` |
| Held-out evaluation scope | `{report['records'][0]['evaluation_scope']}` |
| Data manifest | `{report['data_manifest_fingerprint']}` |
| Code revision | `{report['code_revision']}` |

## Interpretation

If the verdict is `REDESIGN_BEFORE_SCALING`, do not solve the result by adding a large corpus. Change the current DCSS state/readout design, then rerun this exact protocol. If the verdict is `EARNED_NEXT_PILOT`, extend the **same** protocol to a larger document and token budget before any production-scale pretraining.

## References

[1]: {DATASET_URL} "Synaxarium dataset card"
[2]: https://huggingface.co/Nexuss0781/Ethio-BBPE "EthioBBPE model artifact"
"""


def run(
    config: PilotConfig,
    *,
    model_names: Sequence[str] = MODEL_NAMES,
    primary_model_name: str = "dcss_cdi",
    post_training_diagnostics: Callable[[torch.nn.Module, Mapping[str, torch.Tensor], Callable[[str], int] | None], Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    config.validate()
    memory_monitor = HostMemoryMonitor(config.max_host_memory_gb)
    memory_monitor.check("before_corpus_load")
    documents, manifest = load_governed_synaxarium(config)
    partitions = _split_documents(documents, manifest)
    tokenizer = EthioBBPETokenizer.from_pretrained(
        TokenizerConfig(max_chunk_length=config.chunk_length, embedding_dim=4)
    )
    packed = {
        name: pack_limited_documents(partition, tokenizer, config.chunk_length, config.chunks_per_document)
        for name, partition in partitions.items()
    }
    batches = {name: make_batches(examples, tokenizer, config) for name, examples in packed.items()}
    memory_monitor.check("after_data_preparation")
    records: list[Dict[str, Any]] = []
    for seed in config.seeds:
        for name in model_names:
            records.append(
                run_one(
                    name,
                    seed,
                    tokenizer,
                    batches["train"],
                    batches["validation"],
                    batches["test"],
                    config,
                    memory_monitor=memory_monitor,
                    post_training_diagnostics=post_training_diagnostics,
                )
            )
            gc.collect()
            memory_monitor.check(f"{name}_seed_{seed}_after_release")
    summary = aggregate(records)
    report: Dict[str, Any] = {
        "format": "dcss-cdi-ethiobbpe-synaxarium-pilot-v1",
        "status": "COMPLETE",
        "config": config.as_dict(),
        "dataset": {"id": DATASET_ID, "url": DATASET_URL, "license_asserted_by_source_card": DATASET_LICENSE, "document_count": len(documents)},
        "data_manifest_fingerprint": manifest.fingerprint,
        "data_manifest": manifest.as_dict(),
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "packed_example_counts": {name: len(examples) for name, examples in packed.items()},
        "records": records,
        "summary": summary,
        "decision": architecture_decision(
            summary,
            config,
            records,
            required_models=model_names,
            primary_model_name=primary_model_name,
        ),
        "code_revision": _git_revision(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
        "host_memory": memory_monitor.as_dict(),
    }
    report["fingerprint"] = _canonical_digest(report)
    directory = Path(config.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (directory / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def _git_revision() -> str:
    head = Path(".git")
    if not head.exists():
        return "unknown"
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    parser.add_argument("--document-limit", type=int, default=60)
    parser.add_argument("--chunks-per-document", type=int, default=8)
    parser.add_argument("--chunk-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batches", type=int, default=12, help="Zero evaluates every held-out batch; a positive value evaluates a bounded prefix.")
    parser.add_argument("--shuffle-training-batches", action="store_true", help="Use a deterministic per-epoch shuffled batch order.")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--relative-loss-tolerance", type=float, default=0.10)
    parser.add_argument("--max-host-memory-gb", type=float, default=None, help="Optional fail-closed process/container RSS limit in GiB.")
    parser.add_argument("--output-dir", default="results/ethiobbpe_synaxarium_pilot")
    args = parser.parse_args()
    config = PilotConfig(
        seeds=tuple(args.seeds),
        steps=args.steps,
        document_limit=args.document_limit,
        chunks_per_document=args.chunks_per_document,
        chunk_length=args.chunk_length,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        shuffle_training_batches=args.shuffle_training_batches,
        learning_rate=args.learning_rate,
        relative_loss_tolerance=args.relative_loss_tolerance,
        max_host_memory_gb=args.max_host_memory_gb,
        output_dir=args.output_dir,
    )
    report = run(config)
    print(f"Pilot {report['decision']['verdict']}; report={Path(config.output_dir) / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
