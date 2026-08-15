#!/usr/bin/env bash
# CDI Module 1 staged Colab/Google Drive training pipeline.
#
# This script runs exactly one selected submodule at a time. It prepares bounded
# English chunks, trains from the approved parent checkpoint when required,
# validates the selected competency, persists every artifact under Google Drive,
# and stops. It never advances automatically.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# In Colab, mount Drive before running this script. For local dry-runs, set
# CDI_DRIVE_ROOT to a writable directory such as /tmp/cdi-drive.
DRIVE_ROOT="${CDI_DRIVE_ROOT:-/content/drive/MyDrive/CDI}"
STAGE="${CDI_STAGE:-m1.1}"
PARENT_STAGE="${CDI_PARENT_STAGE:-}"
case "${PARENT_STAGE,,}" in
  m1.1) PARENT_STAGE="M1.1" ;;
  m1.2) PARENT_STAGE="M1.2" ;;
  "") ;;
  *) echo "ERROR: unsupported CDI_PARENT_STAGE=$PARENT_STAGE; expected M1.1 or M1.2" >&2; exit 2 ;;
esac
PARENT_DATA_VARIANT="${CDI_PARENT_DATA_VARIANT:-base}"
if [[ "$STAGE" == "m1.2" ]]; then
  BASE_ROOT="${DRIVE_ROOT}/module1/M1.2"
else
  STAGE="m1.1"
  BASE_ROOT="${DRIVE_ROOT}/module1/M1.1"
fi
if [[ "${CDI_NEW_SESSION:-0}" == "1" ]]; then
  SESSION_ID="${CDI_SESSION_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
  RUN_ROOT="${BASE_ROOT}/sessions/${SESSION_ID}"
else
  SESSION_ID="initial"
  RUN_ROOT="$BASE_ROOT"
fi
DATA_VARIANT="${CDI_DATA_VARIANT:-base}"
if [[ "$DATA_VARIANT" == "base" ]]; then
  DATA_ROOT="${BASE_ROOT}/dataset"
else
  DATA_ROOT="${BASE_ROOT}/dataset_${DATA_VARIANT}"
fi
CHECKPOINT_ROOT="${RUN_ROOT}/checkpoints"
REPORT_ROOT="${RUN_ROOT}/reports"
LOG_ROOT="${RUN_ROOT}/logs"
CACHE_ROOT="${BASE_ROOT}/cache"

mkdir -p "$RUN_ROOT" "$DATA_ROOT" "$CHECKPOINT_ROOT" "$REPORT_ROOT" "$LOG_ROOT" "$CACHE_ROOT"
cp "$ROOT/bash.sh" "$RUN_ROOT/bash.sh.snapshot"

if [[ "${CDI_SKIP_INSTALL:-0}" != "1" ]]; then
  if ! python -c 'import torch, datasets, ethiobbpe' >/dev/null 2>&1; then
    python -m pip install -q -r "$ROOT/requirements.txt"
  fi
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CDI_RUN_ROOT="$RUN_ROOT"
export CDI_STAGE="$STAGE"
export CDI_PARENT_STAGE="$PARENT_STAGE"
export CDI_PARENT_DATA_VARIANT="$PARENT_DATA_VARIANT"
export CDI_SESSION_ID="$SESSION_ID"
export CDI_DATA_VARIANT="$DATA_VARIANT"
export CDI_DATA_ROOT="$DATA_ROOT"
export CDI_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"
export CDI_REPORT_ROOT="$REPORT_ROOT"
export CDI_LOG_ROOT="$LOG_ROOT"
export CDI_CACHE_ROOT="$CACHE_ROOT"

python - <<'PY'
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import sys
import time
from typing import Any, Iterable

import torch
from datasets import load_dataset

from cdi.v3 import DCSSLanguageModel, EthioBBPETokenizer, StageCConfig, TokenizerConfig

RUN_ROOT = Path(os.environ["CDI_RUN_ROOT"])
STAGE = os.environ["CDI_STAGE"]
PARENT_STAGE = os.environ["CDI_PARENT_STAGE"]
PARENT_DATA_VARIANT = os.environ["CDI_PARENT_DATA_VARIANT"]
SESSION_ID = os.environ["CDI_SESSION_ID"]
DATA_VARIANT = os.environ["CDI_DATA_VARIANT"]
SUBMODULE = "M1.2" if STAGE == "m1.2" else "M1.1"
REPORT_NAME = f"{SUBMODULE}_REPORT.md"
DATA_ROOT = Path(os.environ["CDI_DATA_ROOT"])
CHECKPOINT_ROOT = Path(os.environ["CDI_CHECKPOINT_ROOT"])
REPORT_ROOT = Path(os.environ["CDI_REPORT_ROOT"])
LOG_ROOT = Path(os.environ["CDI_LOG_ROOT"])
CACHE_ROOT = Path(os.environ["CDI_CACHE_ROOT"])

CONFIG = {
    "format": "dcss-cdi-module1-training-config-v1",
    "module": "M1",
    "submodule": "M1.1",
    "objective": "English token-frequency and local next-token learning",
    "dataset": {
        "name": "wikitext",
        "config": "wikitext-2-raw-v1",
        "revision": "main",
        "source": "https://huggingface.co/datasets/Salesforce/wikitext",
        "train_tokens": 50000,
        "finetune_tokens": 10000,
        "validation_tokens": 10000,
        "test_tokens": 10000,
        "split_seed": 42,
        "document_disjoint": True,
    },
    "model": {
        "family": "dcss_cdi",
        "tier": "nano",
        "embedding_dim": 4,
        "state_dim": 48,
        "vocab_size": 16000,
        "dtype": "float32",
        "device": "auto",
    },
    "optimization": {
        "pretrain_learning_rate": 0.01,
        "finetune_learning_rate": 0.005,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "chunk_length": 64,
        "seed": 42,
        "pretrain_epochs": 1,
        "finetune_epochs": 1,
    },
    "resource": {
        "hard_memory_gib": 11.0,
        "operating_memory_gib": 8.5,
        "checkpoint_every_steps": 25,
    },
    "competency": {
        "minimum_validation_loss_improvement": 0.0,
        "maximum_repetition_rate": 0.75,
        "required_reload_max_abs_error": 1e-6,
    },
    "execution": {
        "stop_after_submodule": True,
        "auto_advance": False,
        "require_user_verdict_before_next_stage": True,
        "compiled": os.environ.get("CDI_COMPILE", "0") == "1",
        "compile_mode": os.environ.get("CDI_COMPILE_MODE", "reduce-overhead"),
        "compile_fullgraph": True,
    },
}

if STAGE == "m1.2":
    CONFIG["submodule"] = "M1.2"
    CONFIG["objective"] = "Local English sentence and short-passage continuation"
    CONFIG["dataset"]["split_seed"] = 137
    CONFIG["optimization"]["pretrain_learning_rate"] = 0.002
    CONFIG["optimization"]["finetune_learning_rate"] = 0.001
    CONFIG["competency"]["maximum_test_gap"] = 0.02
    CONFIG["competency"]["minimum_prompt_pass_fraction"] = 0.75
    if DATA_VARIANT == "r2":
        CONFIG["dataset"]["split_seed"] = 2718
        CONFIG["dataset"]["train_tokens"] = 100000
        CONFIG["dataset"]["finetune_tokens"] = 20000
        CONFIG["optimization"]["max_steps"] = 500


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def text_rows(dataset_split: Iterable[dict[str, Any]]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for index, row in enumerate(dataset_split):
        text = str(row.get("text", "")).strip()
        if len(text) < 20:
            continue
        rows.append((f"wikitext-{index:07d}", text))
    return rows


def token_count(text: str, tokenizer: EthioBBPETokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True).ids)


def collect_documents(rows: list[tuple[str, str]], tokenizer: EthioBBPETokenizer, target_tokens: int, start: int = 0) -> tuple[list[dict[str, Any]], int, int]:
    selected: list[dict[str, Any]] = []
    total = 0
    cursor = start
    while cursor < len(rows) and total < target_tokens:
        identifier, text = rows[cursor]
        count = token_count(text, tokenizer)
        cursor += 1
        if count < 3:
            continue
        selected.append({"id": identifier, "text": text, "token_count": count})
        total += count
    return selected, total, cursor


def chunk_documents(documents: list[dict[str, Any]], tokenizer: EthioBBPETokenizer, chunk_length: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for document in documents:
        ids = list(tokenizer.encode(document["text"], add_special_tokens=True).ids)
        for start in range(0, max(0, len(ids) - 1), chunk_length):
            chunk = ids[start : start + chunk_length]
            if len(chunk) < 3:
                continue
            chunks.append({
                "id": f"{document['id']}::{start}",
                "document_id": document["id"],
                "input_ids": chunk,
                "token_count": len(chunk) - 1,
            })
    return chunks


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def prepare_dataset() -> dict[str, Any]:
    manifest_path = DATA_ROOT / "manifest.json"
    expected_config = {
        "dataset": CONFIG["dataset"],
        "data_variant": DATA_VARIANT,
        "chunk_length": CONFIG["optimization"]["chunk_length"],
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config") == expected_config and all((DATA_ROOT / name).is_file() for name in ("pretrain.jsonl", "finetune.jsonl", "validation.jsonl", "test.jsonl")):
            print("DATASET: using persisted Drive dataset", manifest["fingerprint"])
            return manifest

    print("DATASET: downloading or loading WikiText-2 raw dataset into Drive cache")
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        revision="main",
        cache_dir=str(CACHE_ROOT / "datasets"),
    )
    tokenizer_config = TokenizerConfig(max_chunk_length=CONFIG["optimization"]["chunk_length"], embedding_dim=4)
    tokenizer = EthioBBPETokenizer.from_pretrained(tokenizer_config, cache_dir=CACHE_ROOT / "tokenizer")
    train_rows = text_rows(dataset["train"])
    validation_rows = text_rows(dataset["validation"])
    test_rows = text_rows(dataset["test"])
    seed = CONFIG["dataset"]["split_seed"]
    if DATA_VARIANT != "base":
        previous_dataset_name = "dataset" if PARENT_DATA_VARIANT == "base" else f"dataset_{PARENT_DATA_VARIANT}"
        parent_manifest_stage = PARENT_STAGE or ("M1.2" if STAGE == "m1.2" else "M1.1")
        previous_manifest_path = Path(os.environ["CDI_DRIVE_ROOT"]) / "module1" / parent_manifest_stage / previous_dataset_name / "manifest.json"
        if not previous_manifest_path.is_file():
            raise FileNotFoundError(f"Parent dataset manifest is required before creating {DATA_VARIANT}: {previous_manifest_path}")
        previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
        excluded_ids = {document_id for ids in previous_manifest.get("document_ids", {}).values() for document_id in ids}
        train_rows = [row for row in train_rows if row[0] not in excluded_ids]
        validation_rows = [row for row in validation_rows if row[0] not in excluded_ids]
        test_rows = [row for row in test_rows if row[0] not in excluded_ids]
        if not train_rows or not validation_rows or not test_rows:
            raise RuntimeError("The new M1.2 data variant has no documents after excluding the prior M1.2 corpus.")
    random.Random(seed).shuffle(train_rows)
    random.Random(seed + 1).shuffle(validation_rows)
    random.Random(seed + 2).shuffle(test_rows)

    pretrain_docs, pretrain_doc_tokens, cursor = collect_documents(train_rows, tokenizer, CONFIG["dataset"]["train_tokens"], 0)
    finetune_docs, finetune_doc_tokens, _ = collect_documents(train_rows, tokenizer, CONFIG["dataset"]["finetune_tokens"], cursor)
    validation_docs, validation_doc_tokens, _ = collect_documents(validation_rows, tokenizer, CONFIG["dataset"]["validation_tokens"], 0)
    test_docs, test_doc_tokens, _ = collect_documents(test_rows, tokenizer, CONFIG["dataset"]["test_tokens"], 0)
    if not all((pretrain_docs, finetune_docs, validation_docs, test_docs)):
        raise RuntimeError("WikiText-2 did not provide enough non-empty documents for the declared budgets.")

    splits = {
        "pretrain": chunk_documents(pretrain_docs, tokenizer, CONFIG["optimization"]["chunk_length"]),
        "finetune": chunk_documents(finetune_docs, tokenizer, CONFIG["optimization"]["chunk_length"]),
        "validation": chunk_documents(validation_docs, tokenizer, CONFIG["optimization"]["chunk_length"]),
        "test": chunk_documents(test_docs, tokenizer, CONFIG["optimization"]["chunk_length"]),
    }
    for name, rows in splits.items():
        if not rows:
            raise RuntimeError(f"No causal chunks were created for {name}.")
        write_jsonl(DATA_ROOT / f"{name}.jsonl", rows)

    manifest = {
        "format": "dcss-cdi-module1-dataset-manifest-v1",
        "config": expected_config,
        "data_variant": DATA_VARIANT,
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "vocab_size": tokenizer.vocab_size,
        "source_splits": {"pretrain": "train", "finetune": "train", "validation": "validation", "test": "test"},
        "document_ids": {name: sorted({row["document_id"] for row in rows}) for name, rows in splits.items()},
        "document_disjoint": True,
        "document_token_counts": {
            "pretrain": pretrain_doc_tokens,
            "finetune": finetune_doc_tokens,
            "validation": validation_doc_tokens,
            "test": test_doc_tokens,
        },
        "chunk_counts": {name: len(rows) for name, rows in splits.items()},
        "causal_token_counts": {name: sum(row["token_count"] for row in rows) for name, rows in splits.items()},
        "created_at_unix": time.time(),
    }
    manifest["fingerprint"] = sha256_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    write_json(manifest_path, manifest)
    tokenizer.save(DATA_ROOT / "tokenizer.json")
    print("DATASET: prepared and persisted", manifest["fingerprint"])
    return manifest


def read_chunks(path: Path) -> list[list[int]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows.append([int(value) for value in row["input_ids"]])
    if not rows:
        raise RuntimeError(f"No chunks found in {path}")
    return rows


class CompiledDenseCDI(torch.nn.Module):
    """Fixed-shape compiled wrapper for the exact dense CDI recurrence."""

    def __init__(self, model: DCSSLanguageModel, compile_mode: str) -> None:
        super().__init__()
        self.model = model
        self.compile_mode = compile_mode

    def forward(self, input_ids: torch.Tensor):
        return self.model.forward_chunk_active(
            input_ids,
            return_state=False,
            runtime_guard_mode="deferred",
        )


def build_runner(model: DCSSLanguageModel, compile_enabled: bool, compile_mode: str) -> torch.nn.Module:
    if not compile_enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("Compiled CDI training was requested, but this PyTorch build has no torch.compile.")
    print(f"COMPILE: enabling fixed-shape CDI training with mode={compile_mode!r}, fullgraph=True")
    return torch.compile(
        CompiledDenseCDI(model, compile_mode),
        mode=compile_mode,
        dynamic=False,
        fullgraph=True,
    )


def check_deferred_metrics(metrics: tuple[torch.Tensor, torch.Tensor, torch.Tensor], model: DCSSLanguageModel) -> None:
    spectral_violation, max_geometry_energy, max_state_norm = metrics
    if bool(spectral_violation.detach().item()):
        raise FloatingPointError("Deferred spectral-envelope guard failed.")
    if bool((max_geometry_energy > model.ssm.cell.stage_b_config.energy_limit).detach().item()):
        raise FloatingPointError("Deferred geometry-energy guard failed.")
    if bool((max_state_norm > model.ssm.cell.config.state_norm_bound).detach().item()):
        raise FloatingPointError("Deferred state-norm guard failed.")


def batch(rows: list[list[int]], batch_size: int, index: int, device: str, fixed_length: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    selected = [rows[(index + offset) % len(rows)] for offset in range(batch_size)]
    length = fixed_length or max(len(row) for row in selected)
    if any(len(row) > length for row in selected):
        raise ValueError(f"A chunk of length {max(len(row) for row in selected)} exceeds fixed compiled length {length}.")
    ids = [row + [0] * (length - len(row)) for row in selected]
    mask = [[True] * len(row) + [False] * (length - len(row)) for row in selected]
    return torch.tensor(ids, dtype=torch.long, device=device), torch.tensor(mask, dtype=torch.bool, device=device)


def loss_tensor(
    runner: torch.nn.Module,
    model: DCSSLanguageModel,
    ids: torch.Tensor,
    mask: torch.Tensor,
    compiled: bool,
) -> torch.Tensor:
    if compiled:
        if not bool(mask.all().item()):
            raise ValueError("Compiled CDI training requires all-active fixed-length chunks.")
        logits, _, metrics = runner(ids[:, :-1])
        check_deferred_metrics(metrics, model)
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            ids[:, 1:].reshape(-1),
        )
    return model.causal_loss(ids, mask).loss


def build_model(tokenizer: EthioBBPETokenizer, device: str, seed: int) -> DCSSLanguageModel:
    config = StageCConfig.nano(seed=seed)
    config = config.__class__(**{**config.as_dict(), "device": device})
    config.validate()
    return DCSSLanguageModel(tokenizer, config).to(device)


def loss_on(
    model: DCSSLanguageModel,
    runner: torch.nn.Module,
    rows: list[list[int]],
    batch_size: int,
    device: str,
    compiled: bool,
    max_batches: int | None = None,
) -> float:
    model.eval()
    runner.eval()
    total = 0.0
    tokens = 0
    fixed_length = CONFIG["optimization"]["chunk_length"] if compiled else None
    with torch.no_grad():
        count = len(rows) if max_batches is None else min(len(rows), max_batches)
        for index in range(count):
            ids, mask = batch(rows, batch_size, index, device, fixed_length=fixed_length)
            loss = loss_tensor(runner, model, ids, mask, compiled)
            token_count = int(mask[:, 1:].sum().item())
            total += float(loss.detach().cpu()) * token_count
            tokens += token_count
    if tokens <= 0:
        raise RuntimeError("Evaluation produced zero causal target tokens.")
    return total / tokens


def save_checkpoint(path: Path, model: DCSSLanguageModel, optimizer: torch.optim.Optimizer, metadata: dict[str, Any]) -> str:
    payload = {
        "format": f"dcss-cdi-module1-{STAGE}-checkpoint-v1",
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "metadata": metadata,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def train_phase(
    name: str,
    model: DCSSLanguageModel,
    runner: torch.nn.Module,
    rows: list[list[int]],
    tokenizer: EthioBBPETokenizer,
    device: str,
    learning_rate: float,
    batch_size: int,
    token_budget: int,
    checkpoint_path: Path,
    manifest: dict[str, Any],
    starting_step: int,
    compiled: bool,
) -> tuple[float, float, int, str]:
    optimizer = torch.optim.AdamW(
        [parameter for parameter in runner.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=CONFIG["optimization"]["weight_decay"],
    )
    before = loss_on(model, runner, rows, batch_size, device, compiled)
    model.train()
    runner.train()
    step = starting_step
    tokens_per_step = batch_size * (CONFIG["optimization"]["chunk_length"] - 1)
    requested_steps = max(1, math.ceil(token_budget / tokens_per_step))
    max_steps = int(CONFIG["optimization"].get("max_steps", 250))
    phase_steps = min(requested_steps, max_steps)
    order_rng = random.Random(CONFIG["optimization"]["seed"] + starting_step)
    fixed_length = CONFIG["optimization"]["chunk_length"] if compiled else None
    for local_step in range(phase_steps):
        position = order_rng.randrange(len(rows))
        ids, mask = batch(rows, batch_size, position, device, fixed_length=fixed_length)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_tensor(runner, model, ids, mask, compiled)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"{name} produced non-finite loss at step {step}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["optimization"]["gradient_clip_norm"])
        optimizer.step()
        step += 1
        if step % CONFIG["resource"]["checkpoint_every_steps"] == 0:
            save_checkpoint(checkpoint_path, model, optimizer, {"phase": name, "step": step, "token_budget": token_budget, "requested_steps": requested_steps, "manifest": manifest["fingerprint"], "compiled": compiled})
    after = loss_on(model, runner, rows, batch_size, device, compiled)
    digest = save_checkpoint(checkpoint_path, model, optimizer, {"phase": name, "step": step, "token_budget": token_budget, "requested_steps": requested_steps, "manifest": manifest["fingerprint"], "compiled": compiled})
    print(f"{name}: loss {before:.6f} -> {after:.6f}; steps={phase_steps}; budget={token_budget}; compiled={compiled}; checkpoint={digest}")
    return before, after, step, digest


def load_parent_checkpoint(model: DCSSLanguageModel, device: str, parent_stage: str) -> Path:
    parent_root = Path(os.environ["CDI_DRIVE_ROOT"]) / "module1" / parent_stage
    checkpoint_name = "m1_2_candidate.pt" if parent_stage == "M1.2" else "m1_1_candidate.pt"
    candidates = sorted(parent_root.glob(f"**/checkpoints/{checkpoint_name}"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No {parent_stage} checkpoint found below {parent_root}")
    checkpoint = candidates[0]
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    return checkpoint


def competency_test(model: DCSSLanguageModel, tokenizer: EthioBBPETokenizer, validation_rows: list[list[int]], test_rows: list[list[int]], batch_size: int, device: str, initial_validation_loss: float, final_validation_loss: float, checkpoint_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    reload_model = build_model(tokenizer, device, CONFIG["optimization"]["seed"])
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    reload_model.load_state_dict(payload["model_state"], strict=True)
    reloaded_validation_loss = loss_on(reload_model, validation_rows, batch_size, device)
    reload_error = abs(reloaded_validation_loss - final_validation_loss)
    if reload_error > CONFIG["competency"]["required_reload_max_abs_error"]:
        raise AssertionError(f"Reloaded validation loss differs by {reload_error}")

    prompt_ids = torch.tensor(validation_rows[0][: min(8, len(validation_rows[0]))], dtype=torch.long, device=device)
    with torch.no_grad():
        generated = reload_model.generate(prompt_ids, max_new_tokens=16, mode="greedy")
    tokenizer.assert_ids_in_range(generated)
    generated_list = generated.detach().cpu().tolist()
    continuation = generated_list[len(prompt_ids) :]
    repetition_rate = 1.0
    if continuation:
        repetition_rate = 1.0 - (len(set(continuation)) / len(continuation))
    finite = bool(torch.isfinite(generated.float()).all().item())
    results = {
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": final_validation_loss,
        "test_loss": loss_on(reload_model, test_rows, batch_size, device),
        "reloaded_validation_loss": reloaded_validation_loss,
        "reload_loss_abs_error": reload_error,
        "generated_token_count": len(continuation),
        "generated_repetition_rate": repetition_rate,
        "generated_tokens_finite": finite,
        "generated_text": tokenizer.decode(generated_list),
        "dataset_manifest": manifest["fingerprint"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "peak_rss_gib": rss_gib(),
    }
    results["criteria"] = {
        "validation_improved": final_validation_loss <= initial_validation_loss - CONFIG["competency"]["minimum_validation_loss_improvement"],
        "reload_equivalent": reload_error <= CONFIG["competency"]["required_reload_max_abs_error"],
        "finite_generation": finite,
        "memory_within_hard_limit": rss_gib() <= CONFIG["resource"]["hard_memory_gib"],
    }
    results["observations"] = {
        "repetition_rate": repetition_rate,
        "repetition_within_future_continuation_target": repetition_rate <= CONFIG["competency"]["maximum_repetition_rate"],
    }
    results["status"] = "PASS" if all(results["criteria"].values()) else "FAIL"
    return results


def competency_test_m1_2(model: DCSSLanguageModel, tokenizer: EthioBBPETokenizer, validation_rows: list[list[int]], test_rows: list[list[int]], batch_size: int, device: str, initial_validation_loss: float, final_validation_loss: float, checkpoint_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    reload_model = build_model(tokenizer, device, CONFIG["optimization"]["seed"])
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    reload_model.load_state_dict(payload["model_state"], strict=True)
    reloaded_validation_loss = loss_on(reload_model, validation_rows, batch_size, device)
    reload_error = abs(reloaded_validation_loss - final_validation_loss)
    test_loss = loss_on(reload_model, test_rows, batch_size, device)
    test_gap = (test_loss - final_validation_loss) / max(abs(final_validation_loss), 1e-6)
    prompt_count = min(12, len(validation_rows))
    prompt_records = []
    with torch.no_grad():
        for row in validation_rows[:prompt_count]:
            prompt_length = min(8, len(row))
            prompt = torch.tensor(row[:prompt_length], dtype=torch.long, device=device)
            generated = reload_model.generate(prompt, max_new_tokens=16, mode="greedy")
            tokenizer.assert_ids_in_range(generated)
            generated_ids = generated.detach().cpu().tolist()
            continuation = generated_ids[prompt_length:]
            repetition_rate = 1.0 if not continuation else 1.0 - (len(set(continuation)) / len(continuation))
            prompt_records.append({
                "repetition_rate": repetition_rate,
                "valid": bool(torch.isfinite(generated.float()).all().item()),
                "continuation_token_count": len(continuation),
                "text": tokenizer.decode(generated_ids),
            })
    average_repetition = sum(record["repetition_rate"] for record in prompt_records) / max(len(prompt_records), 1)
    valid_fraction = sum(record["valid"] for record in prompt_records) / max(len(prompt_records), 1)
    prompt_pass_fraction = sum(record["repetition_rate"] <= CONFIG["competency"]["maximum_repetition_rate"] for record in prompt_records) / max(len(prompt_records), 1)
    results = {
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": final_validation_loss,
        "test_loss": test_loss,
        "test_gap_fraction": test_gap,
        "reloaded_validation_loss": reloaded_validation_loss,
        "reload_loss_abs_error": reload_error,
        "prompt_count": prompt_count,
        "average_repetition_rate": average_repetition,
        "prompt_pass_fraction": prompt_pass_fraction,
        "finite_prompt_fraction": valid_fraction,
        "prompt_records": prompt_records,
        "dataset_manifest": manifest["fingerprint"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "peak_rss_gib": rss_gib(),
    }
    results["criteria"] = {
        "validation_improved": final_validation_loss < initial_validation_loss,
        "test_gap_within_limit": test_gap <= CONFIG["competency"]["maximum_test_gap"],
        "prompt_continuation_pass_fraction": prompt_pass_fraction >= CONFIG["competency"]["minimum_prompt_pass_fraction"],
        "repetition_controlled": average_repetition <= CONFIG["competency"]["maximum_repetition_rate"],
        "finite_prompts": valid_fraction >= 1.0,
        "reload_equivalent": reload_error <= CONFIG["competency"]["required_reload_max_abs_error"],
        "memory_within_hard_limit": rss_gib() <= CONFIG["resource"]["hard_memory_gib"],
    }
    results["status"] = "PASS" if all(results["criteria"].values()) else "FAIL"
    return results


def main() -> int:
    if not Path("/content/drive/MyDrive").exists() and "CDI_DRIVE_ROOT" not in os.environ:
        print("ERROR: Google Drive is not mounted. Run drive.mount('/content/drive') first, or set CDI_DRIVE_ROOT for a local dry run.", file=sys.stderr)
        return 2
    existing_status = RUN_ROOT / "reports" / f"{STAGE}_status.json"
    if existing_status.exists() and os.environ.get("CDI_FORCE_RERUN", "0") != "1":
        previous = json.loads(existing_status.read_text(encoding="utf-8"))
        if previous.get("status") in {"PASS", "FAIL"}:
            print(f"{SUBMODULE} already has a persisted result:", previous["status"])
            print("Review the report before rerunning. Use a new session for another approved run.")
            return 0

    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(CONFIG["optimization"]["seed"])
    manifest = prepare_dataset()
    tokenizer = EthioBBPETokenizer.load(DATA_ROOT / "tokenizer.json")
    train_rows = read_chunks(DATA_ROOT / "pretrain.jsonl")
    finetune_rows = read_chunks(DATA_ROOT / "finetune.jsonl")
    validation_rows = read_chunks(DATA_ROOT / "validation.jsonl")
    test_rows = read_chunks(DATA_ROOT / "test.jsonl")
    batch_size = 8 if device == "cuda" else 2
    model = build_model(tokenizer, device, CONFIG["optimization"]["seed"])
    parent_checkpoint = None
    if STAGE != "m1.1":
        parent_stage = PARENT_STAGE or ("M1.2" if STAGE == "m1.2" and DATA_VARIANT == "r2" else "M1.1")
        parent_checkpoint = load_parent_checkpoint(model, device, parent_stage)
    compiled = bool(CONFIG["execution"]["compiled"])
    runner = build_runner(model, compiled, str(CONFIG["execution"]["compile_mode"]))
    initial_validation_loss = loss_on(model, runner, validation_rows, batch_size, device, compiled)
    pretrain_name = "m1_2_adaptation" if STAGE == "m1.2" else "pretrain"
    pretrain_path = CHECKPOINT_ROOT / ("m1_2_adaptation.pt" if STAGE == "m1.2" else "m1_1_pretrain.pt")
    candidate_path = CHECKPOINT_ROOT / ("m1_2_candidate.pt" if STAGE == "m1.2" else "m1_1_candidate.pt")
    pretrain_before, pretrain_after, step, pretrain_digest = train_phase(
        pretrain_name, model, runner, train_rows, tokenizer, device, CONFIG["optimization"]["pretrain_learning_rate"], batch_size, CONFIG["dataset"]["train_tokens"], pretrain_path, manifest, 0, compiled
    )
    finetune_before, finetune_after, step, finetune_digest = train_phase(
        "finetune", model, runner, finetune_rows, tokenizer, device, CONFIG["optimization"]["finetune_learning_rate"], batch_size, CONFIG["dataset"]["finetune_tokens"], candidate_path, manifest, step, compiled
    )
    final_validation_loss = loss_on(model, runner, validation_rows, batch_size, device, compiled)
    competency = (
        competency_test_m1_2(model, tokenizer, validation_rows, test_rows, batch_size, device, initial_validation_loss, final_validation_loss, candidate_path, manifest)
        if STAGE == "m1.2"
        else competency_test(model, tokenizer, validation_rows, test_rows, batch_size, device, initial_validation_loss, final_validation_loss, candidate_path, manifest)
    )
    report = {
        "format": f"dcss-cdi-module1-{STAGE}-report-v1",
        "module": "M1",
        "submodule": SUBMODULE,
        "stage": STAGE,
        "session_id": SESSION_ID,
        "parent_stage": parent_stage if parent_checkpoint else None,
        "parent_checkpoint": str(parent_checkpoint) if parent_checkpoint else None,
        "status": competency["status"],
        "execution": {
            "compiled": compiled,
            "compile_mode": CONFIG["execution"]["compile_mode"] if compiled else None,
            "compile_fullgraph": CONFIG["execution"]["compile_fullgraph"] if compiled else None,
            "fixed_batch_size": batch_size if compiled else None,
            "fixed_sequence_length": CONFIG["optimization"]["chunk_length"] if compiled else None,
            "loss_path": "full_vocabulary_cross_entropy_outside_compiled_recurrence" if compiled else "model_causal_loss",
        },
        "device": device,
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "dataset_manifest": manifest,
        "config": CONFIG,
        "batch_size": batch_size,
        "pretrain": {"before_loss": pretrain_before, "after_loss": pretrain_after, "checkpoint_sha256": pretrain_digest},
        "finetune": {"before_loss": finetune_before, "after_loss": finetune_after, "checkpoint_sha256": finetune_digest},
        "competency": competency,
        "steps": step,
        "elapsed_seconds": time.time() - start,
        "peak_rss_gib": rss_gib(),
        "next_stage_locked": True,
        "requires_user_verdict": True,
    }
    report["fingerprint"] = sha256_bytes(json.dumps(report, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8"))
    latest_path = REPORT_ROOT / f"{STAGE}_latest.json"
    status_path = REPORT_ROOT / f"{STAGE}_status.json"
    report_path = REPORT_ROOT / REPORT_NAME
    write_json(latest_path, report)
    write_json(status_path, {"module": "M1", "submodule": SUBMODULE, "session_id": SESSION_ID, "status": report["status"], "report": str(latest_path), "next_stage_locked": True})
    criteria_text = "\n".join(f"- **{name}:** `{value}`" for name, value in competency["criteria"].items())
    if STAGE == "m1.2":
        evidence_text = "\n".join(f"- `{record['text']}` — repetition rate `{record['repetition_rate']:.3f}`" for record in competency["prompt_records"])
        extra_text = "\n\n## Fixed-prompt evidence\n\n" + evidence_text
    else:
        extra_text = "\n\n## Deferred observations\n\n" + "\n".join(f"- **{name}:** `{value}`" for name, value in competency["observations"].items()) + "\n\n## Generated continuation\n\n````text\n" + competency["generated_text"] + "\n````"
    report_path.write_text(
        f"# CDI {SUBMODULE} Competency Report\n\n"
        f"**Status:** `{report['status']}`  \\n"
        f"**Session:** `{SESSION_ID}`  \\n"
        f"**Parent checkpoint:** `{parent_checkpoint or 'fresh model'}`  \\n"
        f"**Device:** `{device}`  \\n"
        f"**Peak RSS:** `{report['peak_rss_gib']:.3f} GiB`  \n"
        f"**Validation loss:** `{final_validation_loss:.6f}`  \n"
        f"**Test loss:** `{competency['test_loss']:.6f}`  \n"
        f"**Checkpoint:** `{competency['checkpoint_sha256']}`\n\n"
        "## Competency criteria\n\n" + criteria_text + extra_text + "\n\n**Next stage remains locked until the user reviews this report and returns the result for joint verdict.**\n",
        encoding="utf-8",
    )
    print(f"{SUBMODULE} {report['status']}; report={report_path}")
    print("STOP: no later Module 1 submodule will run automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
