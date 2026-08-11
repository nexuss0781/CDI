# CDI v2 Stage A Baseline Contract

## Purpose

This document defines the reproducible reference contract for CDI v2 before the sparse Stage B implementation begins. The reference is intended for correctness and regression comparison; it is not a claim that CDI v2 is production-efficient.

## Reference configurations

| Configuration | Intended use | Precision | Resource policy |
|---|---|---|---|
| `micro` | Default Stage A end-to-end harness | `float32` | Must run on a development CPU within a bounded memory envelope. |
| `tiny` | Historical CDI v2 test configuration | `float64` | May materialize large dense operators; run only when machine resources are sufficient. |
| `small` | Historical production-baseline configuration | `float64` | Not part of the default Stage A gate; run only as an explicitly provisioned benchmark. |

The `micro` configuration preserves CDI v2 semantics while reducing manifold, belief, spinor, and sequence dimensions so the full reproducibility harness can execute safely. It is not a replacement for the historical `tiny` or `small` results. All reports must name the configuration.

## Stage A objective

The ultimate Stage A objective is:

> A clean environment can construct CDI v2, execute its documented forward and loss path, backpropagate through all intended trainable parameters, save and restore a checkpoint, reproduce outputs under a fixed seed, overfit a tiny deterministic corpus, and produce complete scaling and environment records.

The objective is **not** to prove that v2 is efficient. The current implementation contains dense global operators and rebuilds them after optimizer updates. Those limitations are measured and preserved as the motivation for Stage B.

## Harness command

The canonical safe command is:

```bash
python -m benchmarks.stage_a \
  --config micro \
  --seed 42 \
  --scale-lengths 1,2,4,8,16 \
  --output-dir results/stage_a
```

The command writes a unique run directory containing `environment.json`, `config.json`, `checkpoints/`, `scaling.json`, and `run.json`. It also writes `results/stage_a/latest.json` as a pointer to the latest report.

## Mandatory gates

| Gate | Criterion |
|---|---|
| Forward/loss | Construction, forward sequence, composite LM loss, and finiteness succeed. |
| Gradient flow | All intended v2 parameter groups receive finite nonzero gradients. |
| Optimizer/rebuild | An optimizer update and mandatory v2 operator rebuild complete. |
| Checkpoint parameters | Save/restore produces identical parameter fingerprint. |
| Checkpoint outputs | Restored engine produces outputs within the configured tolerance. |
| Determinism | Two same-seed runs produce identical fingerprints, outputs, and losses within tolerance. |
| Tiny overfit | The fixed synthetic corpus achieves at least 90% relative loss reduction in the configured step budget. |
| Scaling forward | All requested sequence lengths produce finite outputs and timing/memory records. |

Every gate is represented as `PASS` or `FAIL` in `run.json`. An exception is an error and cannot be treated as a skip.

## Transition to Stage B

Stage A is ready for review only when `results/stage_a/latest.json` reports `status: PASS`, all mandatory gates are `PASS`, and the run directory contains the raw checkpoint and scaling artifacts. Stage A does not authorize Stage B automatically.

The agent must stop and request explicit user approval after presenting:

1. the Stage A report;
2. the exact commit or source revision;
3. the resource and timing measurements;
4. any known v2 compatibility defects; and
5. the proposed Stage B starting point.

No sparse operator, matrix-free Laplacian, or Stage B code may be implemented before that approval.

## Failure policy

A failed gate must be classified as environment, numerical, serialization, gradient, resource, or algorithmic. The original raw output must be retained. Thresholds may not be weakened after observing a failure without a versioned specification change and a rerun.
