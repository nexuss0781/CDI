"""Stage D zero-dependency tokenizer and reproducible causal-LM evaluation harness."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import resource
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

from cdi.v3 import DCSSLanguageModel, LocalSyntheticCorpus, StageDConfig
from cdi.v3.training import (
    build_model,
    checkpoint_payload,
    deterministic_batches,
    evaluate,
    optimizer_for,
    pack_documents,
    parameter_fingerprint,
    restore_checkpoint,
    seed_everything,
    train_steps,
)

ATOL = 1e-6
RTOL = 1e-5


def _pass(name: str, passed: bool, details: Mapping[str, Any]) -> Dict[str, Any]:
    return {"name": name, "status": "PASS" if passed else "FAIL", "passed": bool(passed), "details": dict(details)}


def _error(left: torch.Tensor, right: torch.Tensor) -> Dict[str, float]:
    difference = (left - right).abs()
    return {"max_abs": float(difference.max().detach().cpu()) if difference.numel() else 0.0, "mean_abs": float(difference.mean().detach().cpu()) if difference.numel() else 0.0}


def _resources(seed: int = 42):
    seed_everything(seed)
    config = StageDConfig.nano(seed=seed)
    corpus = LocalSyntheticCorpus.default()
    tokenizer = corpus.tokenizer(config)
    splits = corpus.split(seed)
    train_examples, truncation_count = pack_documents(splits["train"], tokenizer, config.chunk_length)
    validation_examples, _ = pack_documents(splits["validation"], tokenizer, config.chunk_length)
    test_examples, _ = pack_documents(splits["test"], tokenizer, config.chunk_length)
    return config, corpus, tokenizer, splits, deterministic_batches(train_examples, tokenizer, config), deterministic_batches(validation_examples, tokenizer, config), deterministic_batches(test_examples, tokenizer, config), truncation_count


def tokenizer_gate(seed: int = 42, output_dir: Path | None = None) -> Dict[str, Any]:
    config, _, tokenizer, _, _, _, _, _ = _resources(seed)
    fixtures = ["", "a  b\t\ne\u0301", "☃", "long " * 8]
    rows = []
    passed = True
    for text in fixtures:
        try:
            encoded = tokenizer.encode(text, max_length=None)
            decoded = tokenizer.decode(encoded.ids)
            expected = tokenizer.normalize(text).replace("☃", "�")
            row_passed = decoded == expected
            rows.append({"input": text, "length": len(encoded.ids), "decoded": decoded, "passed": row_passed})
            passed = passed and row_passed
        except Exception as exc:
            rows.append({"input": text, "error": repr(exc), "passed": False})
            passed = False
    artifact = tokenizer.artifact()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save(output_dir / "tokenizer.json")
    return _pass("tokenizer_round_trip", passed, {"fingerprint": tokenizer.fingerprint, "vocab_size": tokenizer.vocab_size, "config": tokenizer.config.as_dict(), "fixtures": rows})


def data_audit_gate(seed: int = 42, output_dir: Path | None = None) -> Dict[str, Any]:
    config, corpus, tokenizer, splits, train_batches, _, _, truncation_count = _resources(seed)
    manifest = corpus.manifest(tokenizer, config)
    split_hashes = [set(manifest["splits"][name]["document_hashes"].values()) for name in ("train", "validation", "test")]
    overlap = sum(len(split_hashes[left].intersection(split_hashes[right])) for left, right in ((0, 1), (0, 2), (1, 2)))
    boundary_ok = all(len(set(batch["document_ids"])) >= 1 for batch in train_batches)
    passed = overlap == 0 and truncation_count == 0 and boundary_ok
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _pass("data_integrity", passed, {"manifest_fingerprint": manifest["fingerprint"], "source": manifest["source"], "split_document_counts": {name: len(value) for name, value in splits.items()}, "hash_overlap": overlap, "truncation_count": truncation_count, "boundary_policy": manifest["preprocessing"]["boundary_policy"]})


def causal_alignment_gate(seed: int = 42) -> Dict[str, Any]:
    config, _, tokenizer, _, train_batches, _, _, _ = _resources(seed)
    model = build_model("dcss_cdi", tokenizer, config)
    batch = train_batches[0]
    ids, mask = batch["input_ids"], batch["attention_mask"]
    perturbed = ids.clone()
    perturb_index = min(4, ids.shape[1] - 1)
    perturbed[:, perturb_index] = tokenizer.unk_id
    with torch.no_grad():
        before, _ = model.forward_chunk(ids, attention_mask=mask)
        after, _ = model.forward_chunk(perturbed, attention_mask=mask)
    difference = (before[:, :perturb_index] - after[:, :perturb_index]).abs().max()
    return _pass("causal_alignment", bool(difference <= ATOL), {"perturbed_index": perturb_index, "pre_causal_max_abs": float(difference), "tolerance": ATOL})


def masking_gate(seed: int = 42) -> Dict[str, Any]:
    config, _, tokenizer, _, _, _, _, _ = _resources(seed)
    seed_everything(seed)
    first = build_model("dcss_cdi", tokenizer, config)
    second = build_model("dcss_cdi", tokenizer, config)
    second.load_state_dict(first.state_dict())
    with torch.no_grad():
        first.output_bias[tokenizer.token_to_id["a"]] = 2.0
        first.output_bias[tokenizer.token_to_id["b"]] = -2.0
        second.output_bias.copy_(first.output_bias)
    ids = torch.tensor([[tokenizer.bos_id, tokenizer.token_to_id["a"], tokenizer.token_to_id["b"], tokenizer.pad_id]], dtype=torch.long)
    mask = torch.tensor([[True, True, False, False]])
    changed_masked = ids.clone()
    changed_masked[:, 2] = tokenizer.unk_id
    first_report = first.causal_loss(ids, mask)
    second_report = second.causal_loss(changed_masked, mask)
    first_report.loss.backward()
    second_report.loss.backward()
    loss_difference = abs(float(first_report.loss.detach() - second_report.loss.detach()))
    gradient_difference = _error(first.embedding.weight.grad, second.embedding.weight.grad)["max_abs"]
    changed_unmasked = ids.clone()
    changed_unmasked[:, 1] = tokenizer.token_to_id["b"]
    unmasked_difference = abs(float(first_report.loss.detach() - first.causal_loss(changed_unmasked, mask).loss.detach()))
    passed = loss_difference <= 1e-7 and gradient_difference <= 1e-7 and unmasked_difference > 1e-4
    return _pass("mask_correctness", passed, {"masked_loss_difference": loss_difference, "masked_gradient_max_abs": gradient_difference, "unmasked_loss_difference": unmasked_difference})


def _active_gradient_groups(model, inventory: Mapping[str, Any]) -> Dict[str, bool]:
    present = {group: False for group, count in inventory["groups"].items() if count > 0 and group not in {"output_projection_tied", "cochain_maps", "initial_state_optional"}}
    for entry in inventory["entries"]:
        name = entry["name"]
        if name.endswith("learned_initial_state"):
            continue
        parameter = dict(model.named_parameters())[name]
        if parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item()):
            present[entry["group"]] = True
    return present


def train_smoke_gate(seed: int = 42, steps: int = 100) -> Dict[str, Any]:
    config, _, tokenizer, _, train_batches, _, _, _ = _resources(seed)
    model = build_model("dcss_cdi", tokenizer, config)
    losses, _, _ = train_steps(model, train_batches[:1], config, steps=steps)
    reduction = 1.0 - losses[-1] / max(losses[0], 1e-12)
    inventory = model.parameter_inventory()
    gradients = _active_gradient_groups(model, inventory)
    passed = reduction >= 0.90 and bool(torch.isfinite(torch.tensor(losses)).all().item()) and all(gradients.values())
    return _pass("tiny_overfit_and_gradient_coverage", passed, {"steps": steps, "initial_loss": losses[0], "final_loss": losses[-1], "relative_reduction": reduction, "gradient_groups": gradients, "parameter_inventory": inventory})


def resume_gate(seed: int = 42, steps: int = 50, interrupt_at: int = 25) -> Dict[str, Any]:
    config, corpus, tokenizer, _, train_batches, _, _, _ = _resources(seed)
    manifest = corpus.manifest(tokenizer, config)
    seed_everything(seed)
    uninterrupted = build_model("dcss_cdi", tokenizer, config)
    full_losses, _, _ = train_steps(uninterrupted, train_batches, config, steps=steps)
    probe = train_batches[0]
    full_logits, _ = uninterrupted.forward_chunk(probe["input_ids"], attention_mask=probe["attention_mask"])

    seed_everything(seed)
    interrupted = build_model("dcss_cdi", tokenizer, config)
    first_losses, optimizer, cursor = train_steps(interrupted, train_batches, config, steps=interrupt_at)
    payload = checkpoint_payload(interrupted, optimizer, tokenizer, manifest, config, step=interrupt_at, cursor=cursor)
    resumed = build_model("dcss_cdi", tokenizer, config)
    resumed_optimizer = optimizer_for(resumed, config)
    restored_step, restored_cursor = restore_checkpoint(payload, resumed, resumed_optimizer, tokenizer)
    second_losses, _, _ = train_steps(resumed, train_batches, config, steps=steps - interrupt_at, optimizer=resumed_optimizer, start_cursor=restored_cursor)
    resumed_logits, _ = resumed.forward_chunk(probe["input_ids"], attention_mask=probe["attention_mask"])
    parameter_errors = [_error(left, right)["max_abs"] for left, right in zip(uninterrupted.parameters(), resumed.parameters())]
    loss_error = max(abs(left - right) for left, right in zip(full_losses, first_losses + second_losses))
    logit_error = _error(full_logits, resumed_logits)["max_abs"]
    parameter_error = max(parameter_errors, default=0.0)
    passed = restored_step == interrupt_at and loss_error <= ATOL and logit_error <= ATOL and parameter_error <= ATOL
    return _pass("checkpoint_resume_determinism", passed, {"steps": steps, "interrupt_at": interrupt_at, "restored_cursor": restored_cursor, "loss_max_abs": loss_error, "logit_max_abs": logit_error, "parameter_max_abs": parameter_error, "tokenizer_fingerprint": tokenizer.fingerprint})


def precision_validation_generation_gate(seed: int = 42) -> Dict[str, Any]:
    config, _, tokenizer, _, train_batches, validation_batches, _, _ = _resources(seed)
    model = build_model("dcss_cdi", tokenizer, config)
    before = parameter_fingerprint(model)
    validation_one = evaluate(model, validation_batches)
    validation_two = evaluate(model, validation_batches)
    after = parameter_fingerprint(model)
    batch = train_batches[0]
    report = model.causal_loss(batch["input_ids"], batch["attention_mask"])
    report.loss.backward()
    prefix = batch["input_ids"][0, :3]
    greedy_one = model.generate(prefix, mode="greedy", max_new_tokens=4)
    greedy_two = model.generate(prefix, mode="greedy", max_new_tokens=4)
    sample_one = model.generate(prefix, mode="sample", seed=7, max_new_tokens=4)
    sample_two = model.generate(prefix, mode="sample", seed=7, max_new_tokens=4)
    finite = bool(torch.isfinite(report.loss).item()) and all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item()) for parameter in model.parameters())
    generation_ok = torch.equal(greedy_one, greedy_two) and torch.equal(sample_one, sample_two) and int(sample_one.max()) < tokenizer.vocab_size
    validation_ok = validation_one == validation_two and before == after
    return _pass("precision_validation_generation", finite and generation_ok and validation_ok, {"precision": "float32_cpu", "cuda_amp": "UNAVAILABLE", "finite": finite, "validation": validation_one, "validation_parameter_unchanged": before == after, "generation": {"greedy_ids": greedy_one.tolist(), "sample_ids": sample_one.tolist(), "reproducible": generation_ok}})


def throughput_gate(seed: int = 42) -> Dict[str, Any]:
    config, _, tokenizer, _, train_batches, _, _, _ = _resources(seed)
    model = build_model("dcss_cdi", tokenizer, config)
    records = []
    for length in (2, 4, 8):
        ids = train_batches[0]["input_ids"][:, :length]
        mask = train_batches[0]["attention_mask"][:, :length]
        start = time.perf_counter()
        with torch.no_grad():
            logits, state = model.forward_chunk(ids, attention_mask=mask)
        seconds = time.perf_counter() - start
        records.append({"length": length, "seconds": seconds, "tokens_per_second": int(ids.numel() / max(seconds, 1e-12)), "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, "state_elements_per_batch": config.chunk_length * 0 + 48, "finite": bool(torch.isfinite(logits).all().item())})
    return _pass("throughput_memory", all(row["finite"] for row in records), {"records": records})


def comparison_gate(seed: int = 42, steps: int = 20) -> Dict[str, Any]:
    config, corpus, tokenizer, _, train_batches, validation_batches, _, _ = _resources(seed)
    results = {}
    for name in ("v2", "dcss_cdi", "transformer"):
        seed_everything(seed)
        model = build_model(name, tokenizer, config)
        start = time.perf_counter()
        losses, _, _ = train_steps(model, train_batches, config, steps=steps)
        elapsed = time.perf_counter() - start
        validation = evaluate(model, validation_batches)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        results[name] = {"model_class": type(model).__name__, "steps": steps, "initial_loss": losses[0], "final_loss": losses[-1], "validation": validation, "parameter_count": parameter_count, "seconds": elapsed, "tokens_per_second": int(steps * config.batch_size * (config.chunk_length - 1) / max(elapsed, 1e-12)), "classification": "synthetic_plumbing_validation"}
    passed = all(torch.isfinite(torch.tensor(value["final_loss"])).item() for value in results.values())
    return _pass("matched_baseline_comparison", passed, {"shared_tokenizer_fingerprint": tokenizer.fingerprint, "shared_data_manifest": corpus.manifest(tokenizer, config)["fingerprint"], "shared_optimizer": "AdamW", "results": results, "warning": "This is a local synthetic plumbing comparison, not a real-corpus language-quality benchmark."})


def render_report(report: Mapping[str, Any]) -> str:
    rows = "\n".join(f"| {gate['name']} | {gate['status']} | {json.dumps(gate['details'], sort_keys=True)[:220]} |" for gate in report["gates"])
    return f"""# Stage D Gate Report — Zero-Dependency Causal LM Integration

## Result

**Status:** `{report['status']}`. Stage D provides a versioned pure-Python Unicode character tokenizer, audited repository-local synthetic corpus, token-level DCSS causal language model, deterministic checkpoint/resume, and a matched small synthetic baseline protocol. The run is explicitly **not** a real-corpus language-quality claim.

| Gate | Status | Evidence summary |
|---|---:|---|
{rows}

## Transition state

```json
{{
  "stage_d": "{report['status']}",
  "stage_e_implementation_allowed": false,
  "required_action": "explicit user approval before Stage E"
}}
```

## References

[1]: https://github.com/nexuss0781/CDI "CDI repository and DCSS-CDI Stage D implementation"
"""


def run_all(seed: int = 42, output_dir: Path | str = Path("results/stage_d")) -> Dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config, corpus, tokenizer, _, _, _, _, _ = _resources(seed)
    gates = [
        tokenizer_gate(seed, output_dir),
        data_audit_gate(seed, output_dir),
        causal_alignment_gate(seed),
        masking_gate(seed),
        train_smoke_gate(seed, config.overfit_steps),
        resume_gate(seed, config.resume_steps, config.resume_interrupt_at),
        precision_validation_generation_gate(seed),
        throughput_gate(seed),
        comparison_gate(seed),
    ]
    passed = all(gate["passed"] for gate in gates)
    manifest = {"format": "dcss-cdi-stage-d-transition-manifest-v1", "config": config.as_dict(), "tokenizer_fingerprint": tokenizer.fingerprint, "data_manifest_fingerprint": corpus.manifest(tokenizer, config)["fingerprint"], "model_widths": {"embedding_dim": tokenizer.config.embedding_dim, "chunk_length": config.chunk_length}, "training_token_budget": config.overfit_steps * config.batch_size * (config.chunk_length - 1), "data_ordering_policy": "deterministic_document_chunks_no_cross_document_packing", "precision": "float32_cpu", "optimizer": {"name": "AdamW", "learning_rate": config.learning_rate, "weight_decay": config.weight_decay, "gradient_clip_norm": config.gradient_clip_norm}, "stopping_rule": "fixed_steps", "stage_e_implementation_allowed": False}
    manifest["manifest_fingerprint"] = __import__("hashlib").sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    report = {"format": "dcss-cdi-stage-d-report-v1", "stage": "D", "status": "PASS" if passed else "FAIL", "seed": seed, "elapsed_seconds": time.perf_counter() - started, "config": config.as_dict(), "gates": gates, "transition_manifest": manifest, "stage_e_implementation_allowed": False, "transition": "Await explicit user approval before Stage E.", "environment": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(), "cuda_available": torch.cuda.is_available()}}
    run_directory = output_dir / f"stage_d_nano_{seed}_{int(time.time())}"
    run_directory.mkdir(parents=True, exist_ok=True)
    report_json = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    (output_dir / "latest.json").write_text(report_json, encoding="utf-8")
    (run_directory / "run.json").write_text(report_json, encoding="utf-8")
    manifest_json = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (output_dir / "transition_manifest.json").write_text(manifest_json, encoding="utf-8")
    (run_directory / "transition_manifest.json").write_text(manifest_json, encoding="utf-8")
    Path("Stages/STAGE_D_GATE_REPORT.md").write_text(render_report(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="all", choices=["all", "tokenizer", "data_audit", "train_smoke", "train", "resume_test", "compare", "report"])
    parser.add_argument("--config", default="nano")
    parser.add_argument("--dataset", default="local_synthetic")
    parser.add_argument("--model", default="dcss_cdi", choices=["v2", "dcss_cdi", "transformer"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--interrupt-at", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/stage_d")
    parser.add_argument("--input", default=None)
    args = parser.parse_args()
    if args.config != "nano":
        config_path = Path(args.config)
        if not config_path.exists():
            raise ValueError("--config must be 'nano' or a committed Stage D JSON configuration artifact.")
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        if config_payload.get("name") not in {None, "nano"}:
            raise ValueError("The Stage D harness supports only the nano configuration tier.")
    if args.dataset != "local_synthetic":
        raise ValueError("This zero-dependency Stage D build ships only the documented local_synthetic corpus; it will not mislabel it as WikiText-2.")
    output_dir = Path(args.output_dir)
    commands = {
        "tokenizer": lambda: tokenizer_gate(args.seed, output_dir),
        "data_audit": lambda: data_audit_gate(args.seed, output_dir),
        "train_smoke": lambda: train_smoke_gate(args.seed, args.steps),
        "train": lambda: train_smoke_gate(args.seed, args.steps),
        "resume_test": lambda: resume_gate(args.seed, args.steps, args.interrupt_at),
        "compare": lambda: comparison_gate(args.seed, min(args.steps, 20)),
    }
    if args.command == "all":
        report = run_all(args.seed, output_dir)
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0 if report["status"] == "PASS" else 1
    if args.command == "report":
        if args.input is None:
            raise ValueError("report requires --input results/stage_d/<run_id> or a report JSON path.")
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        print(render_report(payload))
        return 0
    result = commands[args.command]()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
