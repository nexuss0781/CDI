# Stage A Gate Report — CDI v2 Reproducible Baseline

**Stage:** A — Freeze CDI v2 and establish reproducibility

**Run:** `stage_a_micro_42_1786474345`

**Source revision:** `a2595bfe1ae9b915edb7c8114f318591c5741400`

**Decision:** **READY FOR USER REVIEW — Stage B is not authorized**

## Objective

The Stage A objective was to establish a reproducible CDI v2 baseline that can be constructed, executed, differentiated, checkpointed, restored, overfit on a deterministic micro-corpus, and measured with complete environment and scaling records.

The objective was evaluated using the resource-safe `micro` configuration. The historical `tiny` and `small` configurations remain available, but they materialize substantially larger float64 dense operators and were not used as the default gate in this constrained development environment.

## Implemented Stage A artifacts

| Artifact | Purpose |
|---|---|
| `benchmarks/stage_a.py` | End-to-end deterministic harness and machine-readable gate runner. |
| `benchmarks/configs/stage_a_micro.json` | Frozen micro configuration and thresholds. |
| `tests/stage_a/test_stage_a_contract.py` | Low-memory contract tests for the new harness. |
| `run_stage_a.py` | One-command repository-root entry point. |
| `docs/BASELINE_V2.md` | Baseline contract and transition policy. |
| `results/stage_a/latest.json` | Machine-generated latest gate report. |
| `results/stage_a/<run_id>/` | Raw environment, config, scaling, checkpoint, and run artifacts. |
| `cdi/engine.py` | Minimal compatibility correction: weak but nonzero float32 point gradients are not classified as disconnected. |

No Stage B sparse operator, matrix-free Laplacian, selective state-space recurrence, or v3 code was implemented.

## Harness command

```bash
python run_stage_a.py \
  --config micro \
  --seed 42 \
  --scale-lengths 1,2,4,8,16 \
  --output-dir results/stage_a
```

## Gate results

| Gate | Result | Measured evidence |
|---|---|---|
| Forward and composite loss | PASS | Finite total loss `12.671173`; CE `3.465854`; PPL `32.0038`. |
| Determinism | PASS | Two same-seed runs had identical parameter fingerprints, loss `12.671173`, and output max error `0.0`. |
| Critical gradient flow | PASS | Manifold, metric, theta, injection, readout, connection, and belief paths active and finite. |
| Optimizer and v2 rebuild | PASS | Optimizer update and `rebuild_operators()` completed; recorded global step `1`. |
| Checkpoint parameters | PASS | Saved/restored SHA-256 parameter fingerprint identical. |
| Checkpoint outputs | PASS | Restored output max absolute error `7.9479e-06`, below the `1e-5` float32 gate. |
| Tiny deterministic overfit | PASS | Loss decreased from `11.978094` to `0.895664`; relative reduction `92.52%` in 60 steps. |
| Scaling forward finiteness | PASS | Lengths `1,2,4,8,16` all produced finite outputs and complete timing/memory records. |
| Harness contract tests | PASS | `3 passed`. |
| Repository syntax validation | PASS | `41 passed, 0 failed`. |

The run status is `PASS_WITH_KNOWN_V2_DEFECTS`, not an unqualified pass, because the historical v2 test suite contains a known Clifford relation defect and the sheaf tensors are inactive in the v2 recurrent LM path.

## Measured scaling sample

| Length | Seconds | Tokens/s | RSS MB |
|---:|---:|---:|---:|
| 1 | 0.000366 | 2,733.86 | 676.75 |
| 2 | 0.000399 | 5,009.17 | 676.75 |
| 4 | 0.000699 | 5,721.71 | 676.75 |
| 8 | 0.001290 | 6,201.17 | 676.75 |
| 16 | 0.002253 | 7,100.42 | 676.75 |

These are micro-configuration development measurements only. They are not evidence that CDI v2 is efficient at production scale. Stage B exists specifically to remove the dense-operator bottleneck.

## Known v2 issues preserved in the report

### Clifford relation defect

The historical test `tests/test_core.py::TestCliffordAlgebra::test_clifford_relations_flat` fails with relation error `4.0` for the current real 4×4 dimension-four representation. The low-memory core result was `25 passed, 1 failed`. This defect is outside the micro recurrent LM gate and was not silently changed in Stage A. It must remain a named issue for future mathematical repair.

### Inactive sheaf parameters

`sheaf.embedding_matrix` and `sheaf.output_matrix` receive no gradient in the v2 recurrent LM path because token embeddings and vocabulary projection are supplied externally. They remain visible in diagnostics but are excluded from the critical LM-gradient gate. This is a known v2 limitation and a relevant design input for later stages.

### Resource envelope

The historical v2 path builds dense global operators and can exceed the safe memory envelope even for its documented tiny configuration. Stage A therefore gates reproducibility on `micro` and records the historical configurations separately rather than pretending that a small development run proves scalable behavior.

## Transition decision

The objective gate is **ready for review** because the Stage A harness is reproducible, all mandatory micro gates pass, raw artifacts are written, the compatibility correction is explicit, and known v2 defects are surfaced rather than hidden.

The transition status is:

```json
{
  "stage_a": "READY_FOR_REVIEW",
  "stage_b_implementation_allowed": false,
  "required_action": "user_approval"
}
```

Stage B must not begin until the user explicitly approves it. Approval should authorize only the sparse/matrix-free operator substrate described in `Stages/STAGE_B_SPARSE_OPERATOR_SUBSTRATE.md`.

## References

[1]: https://github.com/nexuss0781/CDI "CDI repository and v2 source revision"
