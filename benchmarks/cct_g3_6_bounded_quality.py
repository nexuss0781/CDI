"""CCT-G3.6: bounded 1,500-step continuation of the retained residual CDI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from benchmarks.cct_g3_4_token_residual import (
    MATERIAL_GRU_TARGET_LOSS,
    _canonical_digest,
    _git_revision,
)
from benchmarks.ethiobbpe_synaxarium_pilot import PilotConfig, run


MODEL_NAMES = ("dcss_residual_cdi", "gru_baseline", "transformer")
REFERENCE_VALIDATION_LOSS = 6.743546


def continuation_decision(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    base = report["decision"]
    candidate_mean = float(summary["dcss_residual_cdi"]["mean_validation_loss"])
    progress_gate = candidate_mean <= REFERENCE_VALIDATION_LOSS
    material_gate = candidate_mean <= MATERIAL_GRU_TARGET_LOSS
    if base["finite_values_gate"] and base["learning_gate"] and base["gru_per_seed_gate"] and progress_gate:
        verdict = "EARNED_BOUNDED_CONTINUATION"
    else:
        verdict = "NO_CONTINUATION_EVIDENCE"
    return {
        "finite_values_gate": bool(base["finite_values_gate"]),
        "learning_gate": bool(base["learning_gate"]),
        "gru_per_seed_gate": bool(base["gru_per_seed_gate"]),
        "progress_gate": progress_gate,
        "reference_validation_loss": REFERENCE_VALIDATION_LOSS,
        "candidate_mean_validation_loss": candidate_mean,
        "material_target_gate": material_gate,
        "material_target_loss": MATERIAL_GRU_TARGET_LOSS,
        "verdict": verdict,
        "scale_authorized": False,
        "next_action": "Review continuation result before any larger step, corpus, context, or capacity proposal.",
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    decision = report["g3_6_decision"]
    rows = "\n".join(
        "| {name} | {params:,.0f} | {loss:.4f} | {test:.4f} | {speed:.1f} |".format(
            name=name,
            params=values["parameter_count"],
            loss=values["mean_validation_loss"],
            test=values["mean_test_loss"],
            speed=values["mean_tokens_per_second"],
        )
        for name, values in report["summary"].items()
    )
    return f"""# CCT-G3.6 Bounded Quality Continuation

**Decision:** `{decision['verdict']}`. **Scale authorization:** `False`.

| Model | Parameters | Mean validation loss | Mean test loss | Mean tokens/sec |
|---|---:|---:|---:|---:|
{rows}

## Gates

| Gate | Result | Evidence |
|---|---|---|
| Complete finite evidence | {'PASS' if decision['finite_values_gate'] else 'FAIL'} | Nine model/seed records are complete and finite. |
| CDI learning | {'PASS' if decision['learning_gate'] else 'FAIL'} | Residual CDI training loss decreased in every seed. |
| CDI versus GRU | {'PASS' if decision['gru_per_seed_gate'] else 'FAIL'} | Residual CDI matches or beats GRU in every seed. |
| Progress over 1,000-step reference | {'PASS' if decision['progress_gate'] else 'FAIL'} | Candidate mean `{decision['candidate_mean_validation_loss']:.6f}`; reference `{decision['reference_validation_loss']:.6f}`. |
| 2% material target | {'PASS' if decision['material_target_gate'] else 'FAIL'} | Candidate mean `{decision['candidate_mean_validation_loss']:.6f}`; target `{decision['material_target_loss']:.6f}`. |

This is a bounded continuation result, not a scale or fluency claim. No outcome authorizes 3,000 steps, larger data, context expansion, capacity changes, or English scaling automatically.

## Reproducibility

| Field | Value |
|---|---|
| Code revision | `{report['code_revision']}` |
| Manifest | `{report['data_manifest_fingerprint']}` |
| Tokenizer | `{report['tokenizer_fingerprint']}` |
| Seeds | `{report['config']['seeds']}` |
| Steps per model/seed | `{report['config']['steps']}` |
| Evaluation | `{report['records'][0]['evaluation_scope']}` |
| Host memory | `{report['host_memory']}` |

## References

[1]: [CCT-G3.6 pre-registration](../docs/CCT_G3_6_PREREGISTRATION.md)
[2]: [Performance readiness](../docs/PERFORMANCE_READINESS.md)
"""


def run_g3_6(config: PilotConfig) -> dict[str, Any]:
    report = run(config, model_names=MODEL_NAMES, primary_model_name="dcss_residual_cdi")
    report["format"] = "dcss-cdi-cct-g3-6-bounded-quality-v1"
    report["g3_6_preregistration"] = "docs/CCT_G3_6_PREREGISTRATION.md"
    report["g3_6_decision"] = continuation_decision(report)
    report["code_revision"] = _git_revision()
    report["fingerprint"] = _canonical_digest({key: value for key, value in report.items() if key != "fingerprint"})
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    parser.add_argument("--document-limit", type=int, default=321)
    parser.add_argument("--chunks-per-document", type=int, default=32)
    parser.add_argument("--chunk-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batches", type=int, default=0)
    parser.add_argument("--shuffle-training-batches", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--relative-loss-tolerance", type=float, default=0.05)
    parser.add_argument("--max-host-memory-gb", type=float, default=11.0)
    parser.add_argument("--output-dir", default="results/colab_cct_g3_6_bounded_quality")
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
    report = run_g3_6(config)
    print(f"CCT-G3.6 {report['g3_6_decision']['verdict']}; report={Path(config.output_dir) / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
