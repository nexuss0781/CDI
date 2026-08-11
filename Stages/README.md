# DCSS-CDI Staged Implementation Specifications

This directory contains the independent implementation and evaluation specifications for evolving CDI v2 into DCSS-CDI. These documents are **plans and test contracts**, not implementation commits. Each stage defines its own scope, required modules, evaluation harness, pass/fail gates, transition tests, failure handling, and exit artifacts.

## Stage order

| Stage | Document | Primary purpose | May start after |
|---|---|---|---|
| A | [`STAGE_A_FREEZE_AND_BASELINE.md`](./STAGE_A_FREEZE_AND_BASELINE.md) | Freeze CDI v2 and make installation, testing, measurement, and checkpointing reproducible. | Initial approval. |
| B | [`STAGE_B_SPARSE_OPERATOR_SUBSTRATE.md`](./STAGE_B_SPARSE_OPERATOR_SUBSTRATE.md) | Replace dense geometric/cochain construction with sparse matrix-free operators. | Stage A pass. |
| C | [`STAGE_C_STABLE_SELECTIVE_RECURRENCE.md`](./STAGE_C_STABLE_SELECTIVE_RECURRENCE.md) | Implement stable selective state dynamics and fast/middle/harmonic memory bands. | Stage B pass. |
| D | [`STAGE_D_NLP_TRAINING_INTEGRATION.md`](./STAGE_D_NLP_TRAINING_INTEGRATION.md) | Connect the verified recurrence to reproducible causal language-model training. | Stage C pass. |
| E | [`STAGE_E_ABLATION_AND_SCALE_STUDY.md`](./STAGE_E_ABLATION_AND_SCALE_STUDY.md) | Determine whether DCSS-CDI provides measured efficiency or quality benefits under matched comparisons. | Stage D pass. |
| F | [`STAGE_F_CAPABILITY_MODULES_AND_VERIFICATION.md`](./STAGE_F_CAPABILITY_MODULES_AND_VERIFICATION.md) | Add optional memory, retrieval, tools, planning, and verification with capability isolation and containment. | Stage E signed go decision. |

## Global transition rule

A stage may not advance because implementation is complete. It may advance only when its **mandatory tests pass, raw results are archived, failures are classified, and a transition manifest is generated**. A later stage must not silently modify the assumptions or configurations that a prior stage certified.

Each stage manifest must contain:

```text
stage_id
stage_version
source_revision
parent_manifest_hash
configuration_hashes
environment_metadata
seed_policy
test_commands
pass_fail_results
failure_records
artifacts
reviewer_decision
```

A failed gate must produce a reproducible failure record. The project may revise a stage specification, but it must then increment the stage version and rerun the affected tests. It must not loosen a threshold after observing results without recording a preregistered rationale.

## Cross-stage acceptance matrix

| Property | A | B | C | D | E | F |
|---|---:|---:|---:|---:|---:|---:|
| Clean installation | Required | Inherited | Inherited | Required | Inherited | Inherited |
| Deterministic configuration | Required | Required | Required | Required | Required | Required |
| Dense-reference equivalence | Baseline only | Required | Inherited | Inherited | Audited | Inherited |
| No dense production operator | Measured | Required | Required | Required | Required | Required |
| Causality | Not applicable | Not applicable | Required | Required | Required | Required |
| Stability envelope | Baseline | Operator-level | Required | Required | Required | Required |
| Checkpoint/resume | Required | Operator state | Required | Required | Required | Required |
| Data leakage audit | Legacy record | Not applicable | Not applicable | Required | Required | Required |
| Matched quality comparison | Baseline | Not applicable | Synthetic only | Initial | Required | Capability-specific |
| Safety/containment | Not applicable | Not applicable | Not applicable | Data integrity | Experimental | Required |

## Common harness rules

All benchmark runners must write raw machine-readable outputs before producing summaries. Human-readable tables are derived artifacts, not the source of truth. Every metric must be traceable to a command, configuration, source revision, seed, and environment.

The harness must distinguish four statuses:

| Status | Meaning |
|---|---|
| `PASS` | Test ran and met its preregistered criterion. |
| `FAIL` | Test ran and did not meet its criterion. |
| `ERROR` | Test could not complete due to an implementation or environment error. |
| `NOT_RUN` | Test was intentionally not applicable and has a recorded reason. |

`ERROR` and unexplained `NOT_RUN` are not passes. A stage cannot transition with a mandatory `FAIL` or `ERROR`.

## Common reporting requirements

Every stage report must separate:

1. **Correctness**, which asks whether the implementation computes the specified function.
2. **Numerical stability**, which asks whether the function remains finite and controlled under declared conditions.
3. **Systems performance**, which asks how time and memory scale.
4. **Language or task quality**, which asks what behavior is learned.
5. **Scientific attribution**, which asks whether an observed result is caused by the claimed component.
6. **Safety and containment**, where external actions or untrusted content are introduced.

A mathematical diagnostic such as a spectral gap, harmonic dimension, or cochain residual must not be reported as a general intelligence score. A benchmark improvement must include its baseline, parameter budget, token budget, seed policy, and uncertainty.

## Required repository convention

The implementation should keep v2 and v3 code paths separate until Stage E is complete. Suggested locations are:

```text
cdi/legacy_v2/
cdi/v3/
benchmarks/
tests/stage_a/
tests/stage_b/
tests/stage_c/
tests/stage_d/
tests/stage_e/
tests/stage_f/
results/stage_a/
results/stage_b/
results/stage_c/
results/stage_d/
results/stage_e/
results/stage_f/
```

These paths are recommendations. If the repository uses different names, the stage manifest must record the mapping.

## Final decision rule

The project may proceed toward larger models or broader capabilities only if the evidence supports the specific next step. Efficiency without trainability is insufficient. Quality without reproducibility is insufficient. Capability without provenance and containment is insufficient. The final objective is a measured, extensible, and scientifically honest sequence substrate—not an unsupported claim of superintelligence.

## References

[1]: https://github.com/nexuss0781/CDI "CDI repository and current architecture"

[2]: https://arxiv.org/abs/2312.00752 "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

[3]: https://proceedings.mlr.press/v202/poli23a.html "Hyena Hierarchy: Towards Larger Convolutional Language Models"

[4]: https://aclanthology.org/2023.findings-emnlp.936/ "RWKV: Reinventing RNNs for the Transformer Era"

[5]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/84a7fc24ed52e8eff514c33e8ac76ea3-Abstract-Conference.html "Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling"
