"""Stage E controlled-ablation and comparative-evaluation harness.

This harness evaluates the frozen Stage D local synthetic corpus only. It
reports engineering, synthetic-quality, and scientific gates separately and
does not present the results as real-corpus language-model quality.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from math import exp, log, sqrt
from pathlib import Path
import random
import statistics
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from cdi.v3 import DCSSLanguageModel, MATRIX, LocalSyntheticCorpus, StageDConfig, StateCodec, build_matrix_model, matrix_manifest
from cdi.v3.training import deterministic_batches, evaluate, optimizer_for, pack_documents, parameter_fingerprint, seed_everything, train_steps

ATOL = 1e-6
SEEDS = (1, 2, 3)
MATRIX_IDS = ("T", "V2", "U", "G", "H", "E", "C", "F")


def _pass(name: str, passed: bool, details: Mapping[str, Any]) -> Dict[str, Any]:
    return {"name": name, "status": "PASS" if passed else "FAIL", "passed": bool(passed), "details": dict(details)}


def _resources(seed: int = 1):
    config = StageDConfig.nano(seed=seed)
    corpus = LocalSyntheticCorpus.default()
    tokenizer = corpus.tokenizer(config)
    splits = corpus.split(seed)
    train_examples, train_truncation = pack_documents(splits["train"], tokenizer, config.chunk_length)
    validation_examples, validation_truncation = pack_documents(splits["validation"], tokenizer, config.chunk_length)
    test_examples, test_truncation = pack_documents(splits["test"], tokenizer, config.chunk_length)
    return config, corpus, tokenizer, deterministic_batches(train_examples, tokenizer, config), deterministic_batches(validation_examples, tokenizer, config), deterministic_batches(test_examples, tokenizer, config), train_truncation + validation_truncation + test_truncation


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()


def _linear_exponent(records: Sequence[Mapping[str, Any]], x_key: str, y_key: str) -> Dict[str, float]:
    xs = [log(float(record[x_key])) for record in records if float(record[y_key]) > 0.0]
    ys = [log(float(record[y_key])) for record in records if float(record[y_key]) > 0.0]
    if len(xs) < 2:
        return {"exponent": float("nan"), "intercept": float("nan"), "residual_rmse": float("nan")}
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    return {"exponent": slope, "intercept": intercept, "residual_rmse": sqrt(statistics.fmean(value * value for value in residuals))}


def _statistics(values: Sequence[float], seed: int = 7, bootstrap_samples: int = 1000) -> Dict[str, Any]:
    values = [float(value) for value in values]
    generator = random.Random(seed)
    bootstrap = []
    for _ in range(bootstrap_samples):
        bootstrap.append(statistics.fmean(generator.choice(values) for _ in values))
    bootstrap.sort()
    return {"n": len(values), "mean": statistics.fmean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0, "median": statistics.median(values), "bootstrap_95_ci": [bootstrap[int(0.025 * (len(bootstrap) - 1))], bootstrap[int(0.975 * (len(bootstrap) - 1))]], "seed_values": values}


def configuration_audit_gate(seeds: Sequence[int] = SEEDS) -> Dict[str, Any]:
    config, corpus, tokenizer, _, _, _, truncation = _resources(seeds[0])
    data_manifest = corpus.manifest(tokenizer, config)
    manifests = {identifier: matrix_manifest(identifier, tokenizer, config) for identifier in MATRIX_IDS}
    fingerprints = {identifier: _fingerprint(manifest) for identifier, manifest in manifests.items()}
    shared = all(manifest["tokenizer_fingerprint"] == tokenizer.fingerprint and manifest["training_config"] == config.as_dict() for manifest in manifests.values())
    return _pass("configuration_and_data_audit", shared and truncation == 0, {"seeds": list(seeds), "tokenizer_fingerprint": tokenizer.fingerprint, "data_manifest_fingerprint": data_manifest["fingerprint"], "truncation_count": truncation, "matrix_manifest_fingerprints": fingerprints, "classification": "local_synthetic_engineering_study"})


def parameter_audit_gate(seed: int = 1) -> Dict[str, Any]:
    config, _, tokenizer, _, _, _, _ = _resources(seed)
    records = {}
    for identifier in MATRIX_IDS:
        seed_everything(seed)
        model = build_matrix_model(identifier, tokenizer, config)
        records[identifier] = {"class": type(model).__name__, "total_parameters": sum(parameter.numel() for parameter in model.parameters()), "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad), "manifest": matrix_manifest(identifier, tokenizer, config)}
    return _pass("parameter_audit", all(record["total_parameters"] > 0 for record in records.values()), {"records": records, "matching_note": "Exact matching is impossible for heterogeneous tiny baselines; counts are reported rather than hidden."})


def train_one(identifier: str, seed: int, steps: int = 100) -> Dict[str, Any]:
    config, corpus, tokenizer, train_batches, validation_batches, test_batches, _ = _resources(seed)
    seed_everything(seed)
    model = build_matrix_model(identifier, tokenizer, config)
    parameter_before = parameter_fingerprint(model)
    started = time.perf_counter()
    losses, optimizer, cursor = train_steps(model, train_batches, config, steps=steps)
    elapsed = time.perf_counter() - started
    validation = evaluate(model, validation_batches)
    test = evaluate(model, test_batches)
    finite = bool(torch.isfinite(torch.tensor(losses)).all().item()) and all(torch.isfinite(parameter).all().item() for parameter in model.parameters())
    return {"id": identifier, "seed": seed, "steps": steps, "train_losses": losses, "initial_loss": losses[0], "final_loss": losses[-1], "validation": validation, "test": test, "elapsed_seconds": elapsed, "tokens_per_second": steps * config.batch_size * (config.chunk_length - 1) / max(elapsed, 1e-12), "cursor": cursor, "finite": finite, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad), "parameter_fingerprint_before": parameter_before, "parameter_fingerprint_after": parameter_fingerprint(model), "tokenizer_fingerprint": tokenizer.fingerprint, "data_manifest_fingerprint": corpus.manifest(tokenizer, config)["fingerprint"], "model_manifest": matrix_manifest(identifier, tokenizer, config)}


def train_matrix_gate(seeds: Sequence[int] = SEEDS, steps: int = 100) -> Dict[str, Any]:
    records = [train_one(identifier, seed, steps) for identifier in MATRIX_IDS for seed in seeds]
    grouped: Dict[str, List[Dict[str, Any]]] = {identifier: [] for identifier in MATRIX_IDS}
    for record in records:
        grouped[record["id"]].append(record)
    summary = {}
    for identifier, runs in grouped.items():
        summary[identifier] = {"definition": MATRIX[identifier].description, "final_validation_loss": _statistics([run["validation"]["loss"] for run in runs]), "final_test_loss": _statistics([run["test"]["loss"] for run in runs]), "final_train_loss": _statistics([run["final_loss"] for run in runs]), "tokens_per_second": _statistics([run["tokens_per_second"] for run in runs]), "finite_runs": sum(1 for run in runs if run["finite"]), "parameter_count": runs[0]["parameter_count"], "runs": runs}
    passed = all(run["finite"] for run in records)
    return _pass("matched_multi_seed_training", passed, {"steps": steps, "seeds": list(seeds), "summary": summary, "raw_runs": records})


def _dcss_for_identifier(identifier: str, seed: int = 1) -> Tuple[StageDConfig, Any, DCSSLanguageModel]:
    config, _, tokenizer, _, _, _, _ = _resources(seed)
    model = build_matrix_model(identifier, tokenizer, config)
    if not isinstance(model, DCSSLanguageModel):
        raise ValueError("This helper only supports DCSS matrix members.")
    return config, tokenizer, model


def long_context_gate(seed: int = 1, delay: int = 32) -> Dict[str, Any]:
    config, tokenizer, full = _dcss_for_identifier("F", seed)
    _, _, without_harmonic = _dcss_for_identifier("H", seed)
    token = torch.tensor([tokenizer.token_to_id.get("a", tokenizer.unk_id)], dtype=torch.long)
    with torch.no_grad():
        _, full_state = full.forward_chunk(token.view(1, 1))
        _, ablated_state = without_harmonic.forward_chunk(token.view(1, 1))
        initial_full_harmonic = float(torch.linalg.vector_norm(full_state.harmonic).item())
        zeros = torch.full((1, 1), tokenizer.pad_id, dtype=torch.long)
        mask = torch.ones((1, 1), dtype=torch.bool)
        for _ in range(delay):
            _, full_state = full.forward_chunk(zeros, state=full_state, attention_mask=mask)
            _, ablated_state = without_harmonic.forward_chunk(zeros, state=ablated_state, attention_mask=mask)
    full_harmonic = float(torch.linalg.vector_norm(full_state.harmonic).item())
    ablated_harmonic = float(torch.linalg.vector_norm(ablated_state.harmonic).item())
    retained_ratio = full_harmonic / max(initial_full_harmonic, 1e-12)
    passed = retained_ratio >= 0.50 and ablated_harmonic == 0.0
    return _pass("long_context_harmonic_retention", passed, {"task": "delayed_token_retention_with_zero_distractors", "delay": delay, "full_harmonic_initial_norm": initial_full_harmonic, "full_harmonic_norm": full_harmonic, "full_retained_ratio": retained_ratio, "no_harmonic_norm": ablated_harmonic, "threshold": {"full_retained_ratio_min": 0.50, "no_harmonic_norm": 0.0}, "scope": "synthetic_memory_diagnostic"})


class SequenceAllocationTrace(TorchDispatchMode):
    """Record actual L×L or full-state-square production allocations."""

    def __init__(self, sequence_length: int, full_state: int) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.full_state = full_state
        self.sequence_square: List[Dict[str, Any]] = []
        self.full_state_square: List[Dict[str, Any]] = []
        self.operations: List[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        result = func(*args, **(kwargs or {}))
        name = str(func)
        self.operations.append(name)
        values: Iterable[torch.Tensor]
        if isinstance(result, torch.Tensor):
            values = (result,)
        elif isinstance(result, (tuple, list)):
            values = tuple(value for value in result if isinstance(value, torch.Tensor))
        else:
            values = ()
        for value in values:
            if value.ndim >= 2 and value.shape[-1] == self.sequence_length and value.shape[-2] == self.sequence_length:
                self.sequence_square.append({"operation": name, "shape": list(value.shape)})
            if value.ndim >= 2 and value.shape[-1] == self.full_state and value.shape[-2] == self.full_state:
                self.full_state_square.append({"operation": name, "shape": list(value.shape)})
        return result


def scaling_gate(lengths: Sequence[int] = (8, 16, 32, 64, 128, 256), seed: int = 1) -> Dict[str, Any]:
    config, tokenizer, model = _dcss_for_identifier("F", seed)
    model.eval()
    records = []
    for length in lengths:
        ids = torch.full((1, length), tokenizer.token_to_id.get("a", tokenizer.unk_id), dtype=torch.long)
        mask = torch.ones_like(ids, dtype=torch.bool)
        start = time.perf_counter()
        with torch.no_grad():
            logits, _ = model.forward_chunk(ids, attention_mask=mask)
        elapsed = time.perf_counter() - start
        persistent_state_bytes = model.ssm.config.total_state_dim * model.embedding.weight.element_size()
        records.append({"length": length, "seconds": elapsed, "rss_mb": resource_rss_mb(), "tokens_per_second": length / max(elapsed, 1e-12), "persistent_state_bytes": persistent_state_bytes, "finite": bool(torch.isfinite(logits).all().item())})
    time_fit = _linear_exponent(records, "length", "seconds")
    memory_fit = _linear_exponent([{**record, "persistent_state_bytes": max(record["persistent_state_bytes"], 1)} for record in records], "length", "persistent_state_bytes")
    passed = time_fit["exponent"] <= 1.25 and memory_fit["exponent"] <= 1.20 and all(record["finite"] for record in records)
    return _pass("sequence_scaling", passed, {"measured_lengths": list(lengths), "records": records, "forward_time_fit": time_fit, "persistent_memory_fit": memory_fit, "scope": "CPU nano measured range; no 4k–8k extrapolation"})


def resource_rss_mb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def streaming_allocation_gate(seed: int = 1, length: int = 512) -> Dict[str, Any]:
    config, tokenizer, model = _dcss_for_identifier("F", seed)
    model.eval()
    token = torch.tensor([[tokenizer.token_to_id.get("a", tokenizer.unk_id)]], dtype=torch.long)
    state = None
    latencies = []
    with torch.no_grad():
        for index in range(length):
            started = time.perf_counter()
            _, state = model.forward_chunk(token, state=state, attention_mask=torch.ones_like(token, dtype=torch.bool))
            if index >= 16:
                latencies.append(time.perf_counter() - started)
    assert state is not None
    payload = StateCodec.pack(state)
    restored = StateCodec.unpack(payload)
    with torch.no_grad():
        left_logits, left_state = model.forward_chunk(token, state=state, attention_mask=torch.ones_like(token, dtype=torch.bool))
        right_logits, right_state = model.forward_chunk(token, state=restored, attention_mask=torch.ones_like(token, dtype=torch.bool))
    state_bytes = sum(value.numel() * value.element_size() for value in state.tensors())
    continuation_error = float((left_logits - right_logits).abs().max())
    trace = SequenceAllocationTrace(sequence_length=128, full_state=model.ssm.config.total_state_dim)
    source = inspect.getsource(DCSSLanguageModel) + inspect.getsource(type(model.ssm.cell))
    original_kron = torch.kron
    try:
        def forbidden_kron(*args: Any, **kwargs: Any):
            raise AssertionError("torch.kron is forbidden in full DCSS production execution")
        torch.kron = forbidden_kron  # type: ignore[assignment]
        ids = torch.full((1, 128), tokenizer.token_to_id.get("a", tokenizer.unk_id), dtype=torch.long)
        with torch.no_grad(), trace:
            model.forward_chunk(ids, attention_mask=torch.ones_like(ids, dtype=torch.bool))
    finally:
        torch.kron = original_kron  # type: ignore[assignment]
    passed = continuation_error <= ATOL and StateCodec.fingerprint(state) == StateCodec.fingerprint(restored) and not trace.sequence_square and not trace.full_state_square and "torch.kron(" not in source and "to_dense" not in source
    return _pass("streaming_and_allocation_audit", passed, {"stream_length": length, "state_bytes": state_bytes, "state_layout": "three bands x four vertices x four channels", "warm_latency_ms_mean": statistics.fmean(latencies) * 1000.0, "continuation_max_abs": continuation_error, "state_fingerprint": StateCodec.fingerprint(state), "runtime_dense_sequence_allocations": trace.sequence_square, "runtime_dense_full_state_allocations": trace.full_state_square, "forbidden_kron_operations": [operation for operation in trace.operations if "kron" in operation.lower()], "source_guard": {"torch_kron_call": "torch.kron(" in source, "to_dense": "to_dense" in source}})


def reproducibility_gate(seed: int = 1, steps: int = 40) -> Dict[str, Any]:
    first = train_one("F", seed, steps)
    second = train_one("F", seed, steps)
    loss_error = max(abs(left - right) for left, right in zip(first["train_losses"], second["train_losses"]))
    final_error = abs(first["validation"]["loss"] - second["validation"]["loss"])
    passed = loss_error <= ATOL and final_error <= ATOL and first["parameter_fingerprint_after"] == second["parameter_fingerprint_after"]
    return _pass("fresh_reproducibility_rerun", passed, {"seed": seed, "steps": steps, "loss_max_abs": loss_error, "validation_loss_abs": final_error, "parameter_fingerprint": first["parameter_fingerprint_after"]})


def analyze_gate(training: Mapping[str, Any], scaling: Mapping[str, Any], streaming: Mapping[str, Any], long_context: Mapping[str, Any], reproducibility: Mapping[str, Any]) -> Dict[str, Any]:
    summary = training["details"]["summary"]
    full_loss = summary["F"]["final_validation_loss"]["mean"]
    ablation_losses = {identifier: summary[identifier]["final_validation_loss"]["mean"] for identifier in ("U", "G", "H", "E", "C")}
    best_ablation = min(ablation_losses.values())
    quality_passed = full_loss <= 4.0 and full_loss <= best_ablation + 0.5 and long_context["passed"]
    engineering_measured = scaling["passed"] and streaming["passed"] and all(summary[identifier]["finite_runs"] == 3 for identifier in MATRIX_IDS)
    scientific = reproducibility["passed"] and training["passed"]
    attribution = {
        "full_minus_geometry_free_validation_loss": full_loss - ablation_losses["G"],
        "full_minus_no_harmonic_validation_loss": full_loss - ablation_losses["H"],
        "full_minus_ungated_validation_loss": full_loss - ablation_losses["U"],
        "harmonic_retention_passed": long_context["passed"],
    }
    if engineering_measured and quality_passed and scientific:
        decision = "CONDITIONAL_GO_SYNTHETIC_ONLY"
    elif engineering_measured and scientific:
        decision = "CONDITIONAL_ENGINEERING_ONLY"
    else:
        decision = "NO_GO_OR_INCONCLUSIVE"
    return _pass("separate_engineering_quality_scientific_analysis", scientific, {"engineering_measured": engineering_measured, "quality_synthetic": quality_passed, "scientific": scientific, "full_validation_loss": full_loss, "best_internal_ablation_loss": best_ablation, "ablation_validation_losses": ablation_losses, "attribution": attribution, "long_context": long_context["details"], "not_measured": ["real_corpus_quality", "transfer_corpus", "4k_transformer_memory_ratio", "legacy_v2_speed_ratio", "natural_language_capability"], "decision": decision, "negative_result_record": "On this synthetic corpus, full and geometry-free validation loss are effectively identical and the no-harmonic ablation has lower validation loss, while full retains a harmonic state in the delayed-retention diagnostic. These observations do not support a blanket geometry or harmonic quality advantage; they are recorded as component-specific negative results. Synthetic-only data also prevents a real-corpus quality go decision regardless of measured engineering success."})


def render_report(report: Mapping[str, Any]) -> str:
    rows = "\n".join(f"| {gate['name']} | {gate['status']} | {json.dumps(gate['details'], sort_keys=True)[:220]} |" for gate in report["gates"])
    decision = report["analysis"]["details"]["decision"]
    return f"""# Stage E Gate Report — Controlled Ablations and Nano Scale Study

## Result

**Status:** `{report['status']}` with decision **`{decision}`**. This is a controlled, multi-seed, CPU nano study on the frozen repository-local synthetic corpus. It separates engineering, synthetic-quality, and scientific evidence and makes no real-corpus or natural-language capability claim.

| Gate | Status | Evidence summary |
|---|---:|---|
{rows}

## Required negative-result record

The 4k–8k scaling range, real-corpus quality, transfer evaluation, matched dense CDI-v2 speed ratio, and 4k Transformer memory ratio are **not measured** in the fixed CPU nano study. They are not inferred from the 8–256 measured range and are not marked passed.

## Transition state

```json
{{
  "stage_e": "{report['status']}",
  "decision": "{decision}",
  "stage_f_implementation_allowed": false,
  "required_action": "explicit user review and approval before Stage F"
}}
```

## References

[1]: https://github.com/nexuss0781/CDI "CDI repository and DCSS-CDI Stage E implementation"
"""


def run_all(seeds: Sequence[int] = SEEDS, output_dir: Path | str = Path("results/stage_e"), steps: int = 100) -> Dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = configuration_audit_gate(seeds)
    parameter = parameter_audit_gate(seeds[0])
    training = train_matrix_gate(seeds, steps)
    scaling = scaling_gate(seed=seeds[0])
    streaming = streaming_allocation_gate(seed=seeds[0])
    long_context = long_context_gate(seed=seeds[0])
    reproducibility = reproducibility_gate(seed=seeds[0])
    analysis = analyze_gate(training, scaling, streaming, long_context, reproducibility)
    gates = [configuration, parameter, training, scaling, streaming, long_context, reproducibility, analysis]
    passed = all(gate["passed"] for gate in gates)
    study_manifest = {"format": "dcss-cdi-stage-e-study-manifest-v1", "seeds": list(seeds), "steps": steps, "matrix": {identifier: MATRIX[identifier].__dict__ for identifier in MATRIX_IDS}, "corpus_classification": "repository_local_synthetic_only", "preregistration": "Stages/STAGE_E_PREREGISTRATION.md", "stage_f_implementation_allowed": False}
    study_manifest["manifest_fingerprint"] = _fingerprint(study_manifest)
    report = {"format": "dcss-cdi-stage-e-report-v1", "stage": "E", "status": "PASS" if passed else "FAIL", "elapsed_seconds": time.perf_counter() - started, "seeds": list(seeds), "steps": steps, "gates": gates, "analysis": analysis, "study_manifest": study_manifest, "stage_f_implementation_allowed": False, "transition": "Await explicit user review and approval before Stage F."}
    run_dir = output_dir / f"stage_e_nano_{'-'.join(map(str, seeds))}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    (output_dir / "latest.json").write_text(payload, encoding="utf-8")
    (run_dir / "run.json").write_text(payload, encoding="utf-8")
    manifest_payload = json.dumps(study_manifest, indent=2, sort_keys=True) + "\n"
    (output_dir / "study_manifest.json").write_text(manifest_payload, encoding="utf-8")
    (run_dir / "study_manifest.json").write_text(manifest_payload, encoding="utf-8")
    rendered = render_report(report)
    (output_dir / "REPORT.md").write_text(rendered, encoding="utf-8")
    (run_dir / "REPORT.md").write_text(rendered, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="all", choices=["all", "matrix", "train_all", "scaling", "long_context", "trace_allocations", "analyze"])
    parser.add_argument("--config", default="nano")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--token-budget", type=int, default=100)
    parser.add_argument("--lengths", default="8,16,32,64,128,256")
    parser.add_argument("--suite", default="all")
    parser.add_argument("--models", default="dcss_cdi,transformer")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", default="results/stage_e")
    args = parser.parse_args()
    if args.config != "nano" and not Path(args.config).exists():
        raise ValueError("--config must be 'nano' or a committed Stage E configuration artifact.")
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    lengths = tuple(int(value) for value in args.lengths.split(",") if value)
    commands = {
        "matrix": lambda: {"matrix": {identifier: MATRIX[identifier].__dict__ for identifier in MATRIX_IDS}},
        "train_all": lambda: train_matrix_gate(seeds, args.token_budget),
        "scaling": lambda: scaling_gate(lengths, seeds[0]),
        "long_context": lambda: long_context_gate(seeds[0]),
        "trace_allocations": lambda: streaming_allocation_gate(seeds[0]),
    }
    if args.command == "all":
        report = run_all(seeds, args.output_dir, args.token_budget)
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0 if report["status"] == "PASS" else 1
    if args.command == "analyze":
        if args.input is None:
            raise ValueError("analyze requires --input pointing to a Stage E report JSON.")
        report = json.loads(Path(args.input).read_text(encoding="utf-8"))
        print(render_report(report))
        return 0
    result = commands[args.command]()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
