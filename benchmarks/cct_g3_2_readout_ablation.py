"""CCT-G3.2: controlled fixed-contrast readout contribution ablation.

This harness freezes the CCT-G3.1 real-data contract. It compares full CDI with
an exact mean-only feature-value control that retains the same 48-to-4 readout
layer and sparse geometry computation, alongside the geometry-free CDI, GRU,
and Transformer controls. It cannot authorize scaling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Mapping, Sequence

from benchmarks.ethiobbpe_synaxarium_pilot import (
    MODEL_NAMES,
    PilotConfig,
    _canonical_digest,
    _git_revision,
    run,
)


G3_2_MODEL_NAMES = (
    "dcss_cdi",
    "dcss_mean_readout_control",
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
            raise ValueError("G3.2 records require integer seed and string model fields.")
        grouped.setdefault(seed, {})[model] = record
    return grouped


def readout_decision(report: Mapping[str, Any], *, parameter_tolerance: float) -> Dict[str, Any]:
    """Evaluate the pre-registered G3.2 readout gate without authorizing scale."""

    records = report["records"]
    config = report["config"]
    seeds = tuple(int(seed) for seed in config["seeds"])
    grouped = _by_seed(records)
    if set(grouped) != set(seeds) or any(set(rows) != set(G3_2_MODEL_NAMES) for rows in grouped.values()):
        raise ValueError("G3.2 report lacks the complete per-seed model matrix.")
    parameter_counts = {name: int(report["summary"][name]["parameter_count"]) for name in G3_2_MODEL_NAMES}
    parameter_relative_spread = (max(parameter_counts.values()) / min(parameter_counts.values())) - 1.0
    full_vs_mean_control = {
        str(seed): float(grouped[seed]["dcss_cdi"]["validation"]["loss"])
        < float(grouped[seed]["dcss_mean_readout_control"]["validation"]["loss"])
        for seed in seeds
    }
    full_vs_geometry_free = {
        str(seed): float(grouped[seed]["dcss_cdi"]["validation"]["loss"])
        < float(grouped[seed]["dcss_geometry_free"]["validation"]["loss"])
        for seed in seeds
    }
    readout_improvement = mean(
        float(grouped[seed]["dcss_mean_readout_control"]["validation"]["loss"])
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
    readout_value = learning and all(full_vs_mean_control.values())
    geometry_reconfirmed = all(full_vs_geometry_free.values())
    return {
        "finite_and_baseline_contract": report["decision"]["finite_values_gate"],
        "full_cdi_learning": learning,
        "parameter_match_gate": parameter_match,
        "parameter_counts": parameter_counts,
        "parameter_relative_spread": parameter_relative_spread,
        "parameter_tolerance": parameter_tolerance,
        "full_beats_mean_readout_control_per_seed": full_vs_mean_control,
        "mean_readout_validation_loss_improvement": readout_improvement,
        "readout_value_gate": readout_value,
        "full_beats_geometry_free_per_seed": full_vs_geometry_free,
        "mean_geometry_validation_loss_improvement": geometry_improvement,
        "geometry_reconfirmation_gate": geometry_reconfirmed,
        "g3_2_verdict": "EARNED_READOUT_EVIDENCE" if parameter_match and readout_value else "NO_READOUT_EVIDENCE",
        "scale_authorized": False,
        "next_action": "Review this readout result before any corrected G2.1 quality rerun or scale decision.",
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
    decision = report["g3_2_decision"]
    return f"""# CCT-G3.2 Contrast-Readout Contribution Ablation

**Readout verdict:** `{decision['g3_2_verdict']}`. **Scale authorization:** `False`.

| Model | Parameters | Mean validation loss | Mean test loss | Mean tokens/sec |
|---|---:|---:|---:|---:|
{rows}

## Pre-registered gates

| Gate | Result | Evidence |
|---|---|---|
| Complete finite evidence | {'PASS' if decision['finite_and_baseline_contract'] else 'FAIL'} | Inherited strict CCT record validation. |
| Full CDI learning | {'PASS' if decision['full_cdi_learning'] else 'FAIL'} | CDI train loss decreased in every seed. |
| Parameter matching | {'PASS' if decision['parameter_match_gate'] else 'FAIL'} | Relative spread `{decision['parameter_relative_spread']:.2%}`; tolerance `{decision['parameter_tolerance']:.2%}`. |
| Full versus mean-only control | {'PASS' if decision['readout_value_gate'] else 'FAIL'} | Full CDI lower validation loss in every seed: `{decision['full_beats_mean_readout_control_per_seed']}`. |
| Geometry reconfirmation | {'PASS' if decision['geometry_reconfirmation_gate'] else 'FAIL'} | Full CDI lower validation loss than geometry-free CDI in every seed: `{decision['full_beats_geometry_free_per_seed']}`. |

The mean full-CDI validation-loss improvement over the mean-only control is `{decision['mean_readout_validation_loss_improvement']:.6f}`. The geometry-reconfirmation improvement is `{decision['mean_geometry_validation_loss_improvement']:.6f}`. This mechanism result is not authorization to increase data, steps, context, capacity, or the memory ceiling.

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

[1]: [CCT-G3.2 pre-registration](../docs/CCT_G3_2_PREREGISTRATION.md)
[2]: [CCT-G3.1 decision](../docs/CCT_G3_1_DECISION.md)
[3]: [Active matched pilot contract](ethiobbpe_synaxarium_pilot.py)
"""


def run_g3_2(config: PilotConfig, *, parameter_tolerance: float = DEFAULT_PARAMETER_TOLERANCE) -> Dict[str, Any]:
    if not 0.0 < parameter_tolerance < 1.0:
        raise ValueError("parameter_tolerance must lie in (0, 1).")
    report = run(config, model_names=G3_2_MODEL_NAMES)
    report["format"] = "dcss-cdi-cct-g3-2-readout-ablation-v1"
    report["g3_2_preregistration"] = "docs/CCT_G3_2_PREREGISTRATION.md"
    report["g3_2_decision"] = readout_decision(report, parameter_tolerance=parameter_tolerance)
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
    parser.add_argument("--output-dir", default="results/colab_cct_g3_2_readout")
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
    report = run_g3_2(config, parameter_tolerance=args.parameter_relative_tolerance)
    print(f"CCT-G3.2 {report['g3_2_decision']['g3_2_verdict']}; report={Path(config.output_dir) / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
