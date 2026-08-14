"""CCT-G3.4: bounded selective token-residual quality-recovery comparison.

This harness freezes the CCT-G3.3 real-data contract. It compares a retained-CDI
candidate with a bounded source-token residual against an exact zero-residual
control, the CCT-G3.3 predecessor, GRU, and Transformer. It distinguishes
mechanism evidence from a 2% material-quality gate and cannot scale directly.
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


G3_4_MODEL_NAMES = (
    "dcss_residual_cdi",
    "dcss_residual_control",
    "dcss_cdi",
    *MODEL_NAMES[1:],
)
DEFAULT_PARAMETER_TOLERANCE = 0.01
DEFAULT_MAX_HOST_MEMORY_GB = 11.0
# Two percent below the submitted CCT-G3.3 matched GRU mean validation loss.
MATERIAL_GRU_TARGET_LOSS = 6.664364393744617


def _by_seed(records: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, Mapping[str, Any]]]:
    grouped: Dict[int, Dict[str, Mapping[str, Any]]] = {}
    for record in records:
        seed = record["seed"]
        model = record["model"]
        if not isinstance(seed, int) or not isinstance(model, str):
            raise ValueError("G3.4 records require integer seed and string model fields.")
        grouped.setdefault(seed, {})[model] = record
    return grouped


def _per_seed_lower(
    grouped: Mapping[int, Mapping[str, Mapping[str, Any]]],
    seeds: Sequence[int],
    lower_name: str,
    higher_name: str,
) -> Dict[str, bool]:
    return {
        str(seed): float(grouped[seed][lower_name]["validation"]["loss"])
        < float(grouped[seed][higher_name]["validation"]["loss"])
        for seed in seeds
    }


def token_residual_decision(report: Mapping[str, Any], *, parameter_tolerance: float) -> Dict[str, Any]:
    """Apply CCT-G3.4 mechanism and material-quality gates without scale authorization."""

    records = report["records"]
    config = report["config"]
    seeds = tuple(int(seed) for seed in config["seeds"])
    grouped = _by_seed(records)
    if set(grouped) != set(seeds) or any(set(rows) != set(G3_4_MODEL_NAMES) for rows in grouped.values()):
        raise ValueError("G3.4 report lacks the complete per-seed model matrix.")
    parameter_counts = {name: int(report["summary"][name]["parameter_count"]) for name in G3_4_MODEL_NAMES}
    parameter_relative_spread = (max(parameter_counts.values()) / min(parameter_counts.values())) - 1.0
    candidate_learning = all(bool(grouped[seed]["dcss_residual_cdi"]["train_loss_decreased"]) for seed in seeds)
    candidate_vs_control = _per_seed_lower(grouped, seeds, "dcss_residual_cdi", "dcss_residual_control")
    candidate_vs_predecessor = _per_seed_lower(grouped, seeds, "dcss_residual_cdi", "dcss_cdi")
    candidate_vs_gru = {
        str(seed): float(grouped[seed]["dcss_residual_cdi"]["validation"]["loss"])
        <= float(grouped[seed]["gru_baseline"]["validation"]["loss"])
        for seed in seeds
    }
    mean_candidate_loss = float(report["summary"]["dcss_residual_cdi"]["mean_validation_loss"])
    control_improvement = mean(
        float(grouped[seed]["dcss_residual_control"]["validation"]["loss"])
        - float(grouped[seed]["dcss_residual_cdi"]["validation"]["loss"])
        for seed in seeds
    )
    predecessor_improvement = mean(
        float(grouped[seed]["dcss_cdi"]["validation"]["loss"])
        - float(grouped[seed]["dcss_residual_cdi"]["validation"]["loss"])
        for seed in seeds
    )
    parameter_match = parameter_relative_spread <= parameter_tolerance
    mechanism_value = candidate_learning and all(candidate_vs_control.values()) and all(candidate_vs_predecessor.values())
    gru_per_seed = all(candidate_vs_gru.values())
    material_margin = mean_candidate_loss <= MATERIAL_GRU_TARGET_LOSS
    if parameter_match and mechanism_value:
        mechanism_verdict = "EARNED_TOKEN_RESIDUAL_EVIDENCE"
    else:
        mechanism_verdict = "NO_TOKEN_RESIDUAL_EVIDENCE"
    if parameter_match and mechanism_value and gru_per_seed and material_margin:
        quality_verdict = "MATERIAL_QUALITY_ADVANTAGE_EARNED"
    elif parameter_match and mechanism_value and gru_per_seed:
        quality_verdict = "QUALITY_RECOVERY_PARTIAL"
    else:
        quality_verdict = "REDESIGN_BEFORE_SCALE"
    return {
        "finite_and_baseline_contract": report["decision"]["finite_values_gate"],
        "candidate_learning": candidate_learning,
        "parameter_match_gate": parameter_match,
        "parameter_counts": parameter_counts,
        "parameter_relative_spread": parameter_relative_spread,
        "parameter_tolerance": parameter_tolerance,
        "candidate_beats_residual_control_per_seed": candidate_vs_control,
        "mean_residual_control_validation_loss_improvement": control_improvement,
        "candidate_beats_predecessor_per_seed": candidate_vs_predecessor,
        "mean_predecessor_validation_loss_improvement": predecessor_improvement,
        "token_residual_value_gate": mechanism_value,
        "candidate_matches_or_beats_gru_per_seed": candidate_vs_gru,
        "gru_per_seed_gate": gru_per_seed,
        "mean_candidate_validation_loss": mean_candidate_loss,
        "material_gru_target_loss": MATERIAL_GRU_TARGET_LOSS,
        "material_gru_margin_gate": material_margin,
        "g3_4_mechanism_verdict": mechanism_verdict,
        "g3_4_quality_verdict": quality_verdict,
        "scale_authorized": False,
        "next_action": "Review this quality-recovery result before any CCT-G2.2 scale-rung proposal.",
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
    decision = report["g3_4_decision"]
    return f"""# CCT-G3.4 Selective Token-Residual Quality Recovery

**Mechanism verdict:** `{decision['g3_4_mechanism_verdict']}`. **Quality verdict:** `{decision['g3_4_quality_verdict']}`. **Scale authorization:** `False`.

| Model | Parameters | Mean validation loss | Mean test loss | Mean tokens/sec |
|---|---:|---:|---:|---:|
{rows}

## Pre-registered gates

| Gate | Result | Evidence |
|---|---|---|
| Complete finite evidence | {'PASS' if decision['finite_and_baseline_contract'] else 'FAIL'} | Inherited strict CCT record validation. |
| Candidate learning | {'PASS' if decision['candidate_learning'] else 'FAIL'} | Candidate train loss decreased in every seed. |
| Parameter matching | {'PASS' if decision['parameter_match_gate'] else 'FAIL'} | Relative spread `{decision['parameter_relative_spread']:.2%}`; tolerance `{decision['parameter_tolerance']:.2%}`. |
| Candidate versus exact residual control | {'PASS' if decision['token_residual_value_gate'] else 'FAIL'} | Candidate lower in every seed: `{decision['candidate_beats_residual_control_per_seed']}`. |
| Candidate versus CCT-G3.3 predecessor | {'PASS' if all(decision['candidate_beats_predecessor_per_seed'].values()) else 'FAIL'} | Candidate lower in every seed: `{decision['candidate_beats_predecessor_per_seed']}`. |
| Candidate versus GRU | {'PASS' if decision['gru_per_seed_gate'] else 'FAIL'} | Candidate matches or beats GRU in every seed: `{decision['candidate_matches_or_beats_gru_per_seed']}`. |
| Material GRU margin | {'PASS' if decision['material_gru_margin_gate'] else 'FAIL'} | Candidate mean `{decision['mean_candidate_validation_loss']:.6f}`; required at or below `{decision['material_gru_target_loss']:.6f}`. |

Mean candidate improvement over the exact residual control is `{decision['mean_residual_control_validation_loss_improvement']:.6f}`. Mean candidate improvement over the CCT-G3.3 predecessor is `{decision['mean_predecessor_validation_loss_improvement']:.6f}`. This result cannot scale directly.

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

[1]: [CCT-G3.4 pre-registration](../docs/CCT_G3_4_PREREGISTRATION.md)
[2]: [CCT-G3.3 decision](../docs/CCT_G3_3_DECISION.md)
[3]: [Active matched pilot contract](ethiobbpe_synaxarium_pilot.py)
"""


def run_g3_4(config: PilotConfig, *, parameter_tolerance: float = DEFAULT_PARAMETER_TOLERANCE) -> Dict[str, Any]:
    if not 0.0 < parameter_tolerance < 1.0:
        raise ValueError("parameter_tolerance must lie in (0, 1).")
    report = run(config, model_names=G3_4_MODEL_NAMES)
    report["format"] = "dcss-cdi-cct-g3-4-token-residual-v1"
    report["g3_4_preregistration"] = "docs/CCT_G3_4_PREREGISTRATION.md"
    report["g3_4_decision"] = token_residual_decision(report, parameter_tolerance=parameter_tolerance)
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
    parser.add_argument("--output-dir", default="results/colab_cct_g3_4_token_residual")
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
    report = run_g3_4(config, parameter_tolerance=args.parameter_relative_tolerance)
    print(f"CCT-G3.4 {report['g3_4_decision']['g3_4_quality_verdict']}; report={Path(config.output_dir) / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
