"""CCT-G3.3: controlled harmonic-memory-band contribution ablation.

This harness freezes the CCT-G3.2 real-data contract. It compares full CDI with
an exact parameter-preserving harmonic-disabled CDI control, retains the
geometry-free reference, and includes matched GRU and Transformer baselines.
It records state and gradient diagnostics but cannot authorize scaling.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Mapping, Sequence

import torch

from benchmarks.ethiobbpe_synaxarium_pilot import (
    MODEL_NAMES,
    PilotConfig,
    _canonical_digest,
    _git_revision,
    run,
)
from cdi.v3.language_model import DCSSLanguageModel
from cdi.v3.ssm import DynamicsDiagnostics


G3_3_MODEL_NAMES = (
    "dcss_cdi",
    "dcss_harmonic_disabled",
    "dcss_geometry_free",
    *MODEL_NAMES[1:],
)
DEFAULT_PARAMETER_TOLERANCE = 0.01
DEFAULT_MAX_HOST_MEMORY_GB = 11.0


def _by_seed(records: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, Mapping[str, Any]]]:
    grouped: Dict[int, Dict[str, Mapping[str, Any]]] = {}
    for record in records:
        seed = record["seed"]
        model = record["model"]
        if not isinstance(seed, int) or not isinstance(model, str):
            raise ValueError("G3.3 records require integer seed and string model fields.")
        grouped.setdefault(seed, {})[model] = record
    return grouped


def _gradient_group(name: str) -> str | None:
    if name.startswith("ssm.cell.bands.fast."):
        return "fast"
    if name.startswith("ssm.cell.bands.middle."):
        return "middle"
    if name.startswith("ssm.cell.bands.harmonic."):
        return "harmonic"
    if name.startswith("ssm.cell.geometry."):
        return "geometry"
    if name.startswith("ssm.cell.readout"):
        return "readout"
    if name.startswith("embedding.") or name == "output_bias":
        return "embedding_output"
    return None


def _gradient_l2_by_group(model: DCSSLanguageModel) -> Dict[str, float]:
    squared = {name: 0.0 for name in ("fast", "middle", "harmonic", "geometry", "readout", "embedding_output")}
    for name, parameter in model.named_parameters():
        group = _gradient_group(name)
        if group is None or parameter.grad is None:
            continue
        squared[group] += float(parameter.grad.detach().square().sum().cpu())
    return {name: math.sqrt(value) for name, value in squared.items()}


def collect_diagnostics(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    memory_check: Callable[[str], int] | None = None,
) -> Dict[str, Any]:
    """Collect a fixed-held-out-batch trace without changing fitted parameters."""

    if not isinstance(model, DCSSLanguageModel):
        return {"applicable": False, "reason": "state trace applies only to DCSS/CDI variants"}
    if memory_check is not None:
        memory_check("g3_3_diagnostics_before")
    previous_mode = model.training
    model.eval()
    input_ids = batch["input_ids"].to(device=model.embedding.weight.device, dtype=torch.long)
    attention_mask = batch["attention_mask"].to(device=input_ids.device, dtype=torch.bool)
    state = model.ssm.initial_state(batch_shape=(input_ids.shape[0],), mode="zero")
    state_norm_trace = []
    with torch.no_grad():
        for index in range(input_ids.shape[1] - 1):
            _, candidate = model.ssm.step(model.embedding(input_ids[:, index]), state)
            state = model._select_state(state, candidate, attention_mask[:, index])
            norms = DynamicsDiagnostics.norms(state)
            state_norm_trace.append({"step": index, **norms})
        final_energy = DynamicsDiagnostics.energy(state)
    model.zero_grad(set_to_none=True)
    gradient_loss = model.causal_loss(input_ids, attention_mask).loss
    gradient_loss.backward()
    gradients = _gradient_l2_by_group(model)
    model.zero_grad(set_to_none=True)
    if previous_mode:
        model.train()
    if memory_check is not None:
        memory_check("g3_3_diagnostics_after")
    return {
        "applicable": True,
        "trace_batch_size": int(input_ids.shape[0]),
        "trace_steps": len(state_norm_trace),
        "state_norm_trace": state_norm_trace,
        "final_band_energy": final_energy,
        "gradient_loss": float(gradient_loss.detach().cpu()),
        "gradient_l2_by_group": gradients,
    }


def _diagnostics_gate(report: Mapping[str, Any], grouped: Mapping[int, Mapping[str, Mapping[str, Any]]], seeds: Sequence[int]) -> bool:
    for seed in seeds:
        for name in ("dcss_cdi", "dcss_harmonic_disabled", "dcss_geometry_free"):
            diagnostics = grouped[seed][name].get("post_training_diagnostics")
            if not isinstance(diagnostics, Mapping) or diagnostics.get("applicable") is not True:
                return False
            trace = diagnostics.get("state_norm_trace")
            energies = diagnostics.get("final_band_energy")
            gradients = diagnostics.get("gradient_l2_by_group")
            if not isinstance(trace, list) or not trace or not isinstance(energies, Mapping) or not isinstance(gradients, Mapping):
                return False
            numeric_values = [
                *[value for row in trace if isinstance(row, Mapping) for key, value in row.items() if key != "step"],
                *energies.values(),
                *gradients.values(),
            ]
            if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in numeric_values):
                return False
        control = grouped[seed]["dcss_harmonic_disabled"]["post_training_diagnostics"]
        control_trace = control["state_norm_trace"]
        if any(float(row["harmonic"]) != 0.0 for row in control_trace):
            return False
        if float(control["final_band_energy"]["harmonic"]) != 0.0:
            return False
        if float(control["gradient_l2_by_group"]["harmonic"]) != 0.0:
            return False
    return True


def harmonic_decision(report: Mapping[str, Any], *, parameter_tolerance: float) -> Dict[str, Any]:
    """Evaluate the pre-registered G3.3 mechanism gate without authorizing scale."""

    records = report["records"]
    config = report["config"]
    seeds = tuple(int(seed) for seed in config["seeds"])
    grouped = _by_seed(records)
    if set(grouped) != set(seeds) or any(set(rows) != set(G3_3_MODEL_NAMES) for rows in grouped.values()):
        raise ValueError("G3.3 report lacks the complete per-seed model matrix.")
    parameter_counts = {name: int(report["summary"][name]["parameter_count"]) for name in G3_3_MODEL_NAMES}
    parameter_relative_spread = (max(parameter_counts.values()) / min(parameter_counts.values())) - 1.0
    full_vs_harmonic_disabled = {
        str(seed): float(grouped[seed]["dcss_cdi"]["validation"]["loss"])
        < float(grouped[seed]["dcss_harmonic_disabled"]["validation"]["loss"])
        for seed in seeds
    }
    harmonic_disabled_vs_full = {
        str(seed): float(grouped[seed]["dcss_harmonic_disabled"]["validation"]["loss"])
        < float(grouped[seed]["dcss_cdi"]["validation"]["loss"])
        for seed in seeds
    }
    full_vs_geometry_free = {
        str(seed): float(grouped[seed]["dcss_cdi"]["validation"]["loss"])
        < float(grouped[seed]["dcss_geometry_free"]["validation"]["loss"])
        for seed in seeds
    }
    harmonic_improvement = mean(
        float(grouped[seed]["dcss_harmonic_disabled"]["validation"]["loss"])
        - float(grouped[seed]["dcss_cdi"]["validation"]["loss"])
        for seed in seeds
    )
    geometry_improvement = mean(
        float(grouped[seed]["dcss_geometry_free"]["validation"]["loss"])
        - float(grouped[seed]["dcss_cdi"]["validation"]["loss"])
        for seed in seeds
    )
    learning = all(bool(grouped[seed]["dcss_cdi"]["train_loss_decreased"]) for seed in seeds)
    parameter_match = parameter_relative_spread <= parameter_tolerance
    diagnostics = _diagnostics_gate(report, grouped, seeds)
    harmonic_value = learning and diagnostics and all(full_vs_harmonic_disabled.values())
    harmonic_negative = learning and diagnostics and all(harmonic_disabled_vs_full.values())
    if parameter_match and harmonic_value:
        verdict = "EARNED_HARMONIC_EVIDENCE"
    elif parameter_match and harmonic_negative:
        verdict = "HARMONIC_NEGATIVE_EVIDENCE"
    else:
        verdict = "NO_HARMONIC_EVIDENCE"
    return {
        "finite_and_baseline_contract": report["decision"]["finite_values_gate"],
        "full_cdi_learning": learning,
        "parameter_match_gate": parameter_match,
        "parameter_counts": parameter_counts,
        "parameter_relative_spread": parameter_relative_spread,
        "parameter_tolerance": parameter_tolerance,
        "state_gradient_diagnostics_gate": diagnostics,
        "full_beats_harmonic_disabled_per_seed": full_vs_harmonic_disabled,
        "harmonic_disabled_beats_full_per_seed": harmonic_disabled_vs_full,
        "mean_harmonic_validation_loss_improvement": harmonic_improvement,
        "harmonic_value_gate": harmonic_value,
        "harmonic_negative_gate": harmonic_negative,
        "full_beats_geometry_free_per_seed": full_vs_geometry_free,
        "mean_geometry_validation_loss_improvement": geometry_improvement,
        "geometry_reconfirmation_gate": all(full_vs_geometry_free.values()),
        "g3_3_verdict": verdict,
        "scale_authorized": False,
        "next_action": "Review this harmonic-band result before any architecture-selection or quality-rerun decision.",
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    rows = "\n".join(
        "| {name} | {parameters:,.0f} | {validation:.4f} | {test:.4f} | {throughput:.1f} |".format(
            name=name,
            parameters=values["parameter_count"],
            validation=values["mean_validation_loss"],
            test=values["mean_test_loss"],
            throughput=values["mean_tokens_per_second"],
        )
        for name, values in report["summary"].items()
    )
    decision = report["g3_3_decision"]
    return f"""# CCT-G3.3 Harmonic-Memory-Band Contribution Ablation

**Harmonic verdict:** `{decision['g3_3_verdict']}`. **Scale authorization:** `False`.

| Model | Parameters | Mean validation loss | Mean test loss | Mean tokens/sec |
|---|---:|---:|---:|---:|
{rows}

## Pre-registered gates

| Gate | Result | Evidence |
|---|---|---|
| Complete finite evidence | {'PASS' if decision['finite_and_baseline_contract'] else 'FAIL'} | Inherited strict CCT record validation. |
| Full CDI learning | {'PASS' if decision['full_cdi_learning'] else 'FAIL'} | CDI train loss decreased in every seed. |
| Parameter matching | {'PASS' if decision['parameter_match_gate'] else 'FAIL'} | Relative spread `{decision['parameter_relative_spread']:.2%}`; tolerance `{decision['parameter_tolerance']:.2%}`. |
| State and gradient diagnostics | {'PASS' if decision['state_gradient_diagnostics_gate'] else 'FAIL'} | Finite fixed-held-out trace; harmonic-disabled band remains zero. |
| Full versus harmonic-disabled control | {'PASS' if decision['harmonic_value_gate'] else 'FAIL'} | Full CDI lower validation loss in every seed: `{decision['full_beats_harmonic_disabled_per_seed']}`. |
| Harmonic-negative pattern | {'PASS' if decision['harmonic_negative_gate'] else 'FAIL'} | Harmonic-disabled CDI lower validation loss in every seed: `{decision['harmonic_disabled_beats_full_per_seed']}`. |
| Geometry reconfirmation | {'PASS' if decision['geometry_reconfirmation_gate'] else 'FAIL'} | Full CDI lower validation loss than geometry-free CDI in every seed: `{decision['full_beats_geometry_free_per_seed']}`. |

The mean full-CDI validation-loss improvement over the harmonic-disabled control is `{decision['mean_harmonic_validation_loss_improvement']:.6f}`. The geometry-reconfirmation improvement is `{decision['mean_geometry_validation_loss_improvement']:.6f}`. The result does not authorize increased data, steps, context, capacity, or memory.

## Reproducibility

| Field | Value |
|---|---|
| Code revision | `{report['code_revision']}` |
| Data manifest | `{report['data_manifest_fingerprint']}` |
| Tokenizer fingerprint | `{report['tokenizer_fingerprint']}` |
| Seeds | `{report['config']['seeds']}` |
| Steps per model/seed | `{report['config']['steps']}` |
| Training batch order | `{report['records'][0]['training_batch_order']}` |
| Evaluation scope | `{report['records'][0]['evaluation_scope']}` |
| Host-memory guard | `{report['host_memory']}` |

## References

[1]: [CCT-G3.3 pre-registration](../docs/CCT_G3_3_PREREGISTRATION.md)
[2]: [CCT-G3.2 decision](../docs/CCT_G3_2_DECISION.md)
[3]: [Active matched pilot contract](ethiobbpe_synaxarium_pilot.py)
"""


def run_g3_3(config: PilotConfig, *, parameter_tolerance: float = DEFAULT_PARAMETER_TOLERANCE) -> Dict[str, Any]:
    if not 0.0 < parameter_tolerance < 1.0:
        raise ValueError("parameter_tolerance must lie in (0, 1).")
    report = run(config, model_names=G3_3_MODEL_NAMES, post_training_diagnostics=collect_diagnostics)
    report["format"] = "dcss-cdi-cct-g3-3-harmonic-ablation-v1"
    report["g3_3_preregistration"] = "docs/CCT_G3_3_PREREGISTRATION.md"
    report["g3_3_decision"] = harmonic_decision(report, parameter_tolerance=parameter_tolerance)
    report["code_revision"] = _git_revision()
    report["fingerprint"] = _canonical_digest({key: value for key, value in report.items() if key != "fingerprint"})
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    parser.add_argument("--document-limit", type=int, default=321)
    parser.add_argument("--chunks-per-document", type=int, default=32)
    parser.add_argument("--chunk-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batches", type=int, default=0)
    parser.add_argument("--shuffle-training-batches", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--relative-loss-tolerance", type=float, default=0.05)
    parser.add_argument("--parameter-relative-tolerance", type=float, default=DEFAULT_PARAMETER_TOLERANCE)
    parser.add_argument("--max-host-memory-gb", type=float, default=DEFAULT_MAX_HOST_MEMORY_GB)
    parser.add_argument("--output-dir", default="results/colab_cct_g3_3_harmonic")
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
    report = run_g3_3(config, parameter_tolerance=args.parameter_relative_tolerance)
    print(f"CCT-G3.3 {report['g3_3_decision']['g3_3_verdict']}; report={Path(config.output_dir) / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
