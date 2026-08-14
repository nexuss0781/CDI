# CCT-G3.3 Pre-Registration: Harmonic-Memory-Band Contribution

> **Status:** pre-registered implementation plan. This document authorizes one exact harmonic-band control, its local gates, and one bounded empirical comparison. It does **not** authorize a quality-scale, context, capacity, corpus, throughput, or fluency claim.

## Question

CCT-G3.1 established repeated sparse-geometry value, and CCT-G3.2 established repeated fixed-contrast-readout value. Full CDI nevertheless remained above GRU validation loss in every frozen-protocol seed. The remaining unmeasured distinctive component is the **harmonic memory band**, whose declared base time-constant range is 16–64 while the current training chunks have length 16 and reset state between chunks. [1] [2]

CCT-G3.3 asks whether this harmonic band adds held-out predictive value, subtracts value under the current short-context protocol, or has no repeated detectable contribution. It does not test long-range retention; the protocol is intentionally held fixed.

## One Changed Mechanism

The control retains the complete full-CDI parameter inventory and state layout, including the harmonic band module, its gate, generator, integrator, learned-initial-state slot, readout columns, sparse topology, Laplacian correction, tied embedding/output projection, and 48-to-4 readout. It changes only the harmonic band’s causal state contribution:

\[
z_{t+1}^{\mathrm{harmonic}} = 0,
\qquad
\phi_{\mathrm{harmonic}}(z_{t+1}) = \phi_{\mathrm{harmonic}}(0).
\]

The harmonic-disabled control uses a first-class `StageCConfig.harmonic_ablation=True` setting. At every recurrent step it deterministically emits an all-zero harmonic state instead of evaluating its harmonic gate, generator, exact integrator, or geometry correction. The harmonic feature slots remain in the unchanged readout input and receive zeros. No parameter is removed, resized, frozen by optimizer mutation, or added; the expected total parameter count remains **80,510**.

Because no causal path reaches the harmonic module in this control, every trainable parameter whose name begins `ssm.cell.bands.harmonic.` is intentionally inactive and must be explicitly declared to the strict trainer. The geometry edge parameter remains active through the fast and middle bands and must **not** be declared inactive.

## Frozen Controls

| Field | Locked value |
|---|---|
| Dataset and governed manifest | `Nexuss0781/synaxarium`; 321 documents; document/content-hash isolated splits |
| Tokenizer | EthioBBPE artifact fingerprint `d78996f0aca122d74054b927902aa9bf80c2b5cf00747a7cf4327ff0f7d1a88c` |
| Seeds | `[11, 29, 47]` |
| Steps | 1,000 per model/seed; 30,000 causal positions per model/seed |
| Context and batches | Chunk length 16; batch size 2; 32 chunks/document |
| Optimizer and precision | AdamW; learning rate 0.01; CPU float32 |
| Batch/evaluation policy | Deterministic per-epoch shuffle; all held-out validation and test batches |
| Memory guard | Process/container RSS limit of 11 GiB |
| Parameter tolerance | Maximum total parameter spread of 1% |
| Decision metric | Token-weighted held-out validation cross-entropy; lower is better |

## Model Matrix

| Model | Purpose |
|---|---|
| `dcss_cdi` | Full three-band CDI with mean-plus-contrast readout and sparse geometry. |
| `dcss_harmonic_disabled` | Exact parameter-preserving control with only the harmonic state path deterministically zeroed. |
| `dcss_geometry_free` | Mean-plus-contrast CDI with only Laplacian correction disabled; preserves the G3.1 geometry reference. |
| `gru_baseline` | Matched recurrent quality reference. |
| `transformer` | Matched causal-attention quality reference. |

## Local Gates Before Colab

| Gate | Requirement |
|---|---|
| First-class configuration | The control is created only by `StageCConfig.harmonic_ablation=True`; the serialized configuration records that setting. |
| Exact harmonic state control | The harmonic-disabled state is all zeros at initialization and after every causal step; fast and middle state shape is unchanged. |
| Causal behavior | Forward logits and causal loss are finite and preserve the full model’s target, mask, and output shapes. |
| Distinguishability | Identically initialized full and harmonic-disabled CDI produce a nonzero causal logit or loss difference on a fixed token fixture. |
| Gradient contract | Full CDI has finite, nonzero causal-loss gradient in at least one harmonic parameter. The control declares exactly all `ssm.cell.bands.harmonic.*` trainable parameters inactive; those gradients are zero or absent, while every other non-exempt active parameter is finite-gradient checked by the trainer. |
| Parameter fairness | Full, harmonic-disabled, and geometry-free CDI have identical total parameter counts; all five models remain within 1%. |
| State safety | A short deterministic control training step passes the existing runtime norm/energy guards and produces finite state diagnostics. |

## Recorded Diagnostics

The CCT-G3.3 result must retain ordinary held-out loss, test loss, token accuracy, throughput, host-memory, and parameter-count records. In addition, it must emit a fixed-held-out-batch diagnostic trace after training for each CDI variant: per-step fast, middle, harmonic, and total state norms; final band energies; and the causal-loss gradient L2 norms grouped by `fast`, `middle`, `harmonic`, `geometry`, `readout`, and `embedding/output` parameters. The trace is descriptive evidence only and does not replace the pre-registered validation-loss gate.

## Empirical Decision Rules

| Result | Decision |
|---|---|
| Full CDI has lower validation loss than harmonic-disabled CDI in every seed; all finite, learning, local, and parameter gates pass | `EARNED_HARMONIC_EVIDENCE`; retain the harmonic band in the current bounded configuration. |
| Harmonic-disabled CDI has lower validation loss than full CDI in every seed; all finite, learning, local, and parameter gates pass | `HARMONIC_NEGATIVE_EVIDENCE`; do not remove the band yet—require a separately pre-registered architecture-selection review. |
| Neither model wins in every seed, or a contract gate fails | `NO_HARMONIC_EVIDENCE`; make no configuration-selection or scaling claim. |
| Full CDI also beats geometry-free CDI in every seed | G3.1 geometry evidence is independently re-confirmed under the five-model matrix. |
| Full CDI still fails the three-seed GRU relation | Global status remains `REDESIGN_BEFORE_SCALE`; no CCT-G2.2, G4, or G5 work is permitted. |

No CCT-G3.3 outcome authorizes scaling. Any later quality rerun, architecture selection, context increase, capacity change, corpus change, throughput work, or memory-limit change requires a separate pre-registration and an explicit `Todo.md` transition.

## References

[1]: [CCT-G3.2 decision](CCT_G3_2_DECISION.md)  
[2]: [Active Stage C implementation](../cdi/v3/ssm.py)  
[3]: [Authoritative CCT checklist](../Todo.md)
