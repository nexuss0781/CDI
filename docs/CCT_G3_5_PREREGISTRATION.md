# CCT-G3.5 Pre-Registration: State-Conditioned Token-Residual Fusion

> **Status:** pre-registered implementation plan. This document authorizes one bounded quality-recovery mechanism, one exact parameter-aware control, local gates, and one frozen real-data comparison. It does **not** authorize changes to data scale, corpus, context, model width, steps, optimizer, precision, throughput work, English-fluency claims, or production deployment.

## Question

CCT-G3.4 established that a bounded source-token residual materially improves the retained CDI configuration and beats GRU in every seed. The residual is currently added unconditionally to the DCSS state readout:

\[
h_t = R\!\left(\phi(z_t)\right) + r(e_t).
\]

The residual module receives only the current source-token embedding \(e_t\); it cannot adapt its contribution to the current recurrent representation \(R(\phi(z_t))\). The remaining gap to the pre-registered 2% material-quality target is 0.079182 validation-loss units. [1]

CCT-G3.5 asks whether a bounded state-conditioned fusion gate can choose when each residual coordinate should complement the recurrent state rather than contribute with a fixed additive coefficient.

## One Changed Mechanism

The candidate preserves the full CCT-G3.4 configuration: the 48-element DCSS recurrence, fixed contrast readout, sparse geometry, harmonic memory band, selective token residual, tokenizer, tied output projection, data manifest, context, width, and training contract. It changes only the combination of the existing four-dimensional state readout \(h_t^{\mathrm{state}}\) and existing four-dimensional token residual \(r_t\):

\[
g_t=\sigma\!\left(W_f\,[h_t^{\mathrm{state}};r_t]+b_f\right),
\qquad
h_t=h_t^{\mathrm{state}}+g_t\odot r_t.
\]

The fusion gate is a four-output affine map of the concatenated eight-dimensional causal features. Its 36 trainable parameters are 32 weights and four biases. The gate weights use Xavier-uniform initialization with gain 0.25 and the gate bias is initialized to 2.0, yielding a bounded initial preference toward the retained CCT-G3.4 additive path without making the candidate identical to the control. The sigmoid gate bounds every coordinate in \((0,1)\); it cannot create an unbounded residual. Both inputs at position \(t\) are causal, so the fusion gate cannot access target token \(x_{t+1}\) or later sequence positions.

The candidate has **80,586** trainable parameters. The exact fusion-control has the same 80,586 parameters and same execution path, but deterministically sets \(g_t=\mathbf{1}\), producing the exact CCT-G3.4 unconditionally added residual representation. Its `residual_fusion.*` parameters are declared intentionally inactive to the strict trainer; no parameter is removed, resized, optimizer-frozen, or replaced.

## Frozen Contract

| Field | Locked value |
|---|---|
| Dataset and governed manifest | `Nexuss0781/synaxarium`; 321 documents; document/content-hash isolated splits |
| Tokenizer | EthioBBPE artifact fingerprint `d78996f0aca122d74054b927902aa9bf80c2b5cf00747a7cf4327ff0f7d1a88c` |
| Seeds | `[11, 29, 47]` |
| Training budget | 1,000 steps/model/seed; 30,000 causal positions/model/seed |
| Context and batches | Chunk length 16; batch size 2; 32 chunks/document |
| Optimizer and precision | AdamW; learning rate 0.01; CPU float32 |
| Batch/evaluation policy | Deterministic per-epoch shuffle; all held-out validation and test batches |
| Memory guard | Process/container RSS maximum 11 GiB |
| Parameter tolerance | Maximum total parameter spread 1% |
| Primary metric | Token-weighted held-out validation cross-entropy; lower is better |
| Material target | Candidate mean validation loss at or below **6.664364**, 2% below recorded matched-GRU mean 6.800372 |

## Model Matrix

| Model | Purpose |
|---|---|
| `dcss_fused_residual_cdi` | Full retained CCT-G3.4 residual CDI plus bounded state-conditioned residual fusion. |
| `dcss_fusion_control` | Exact capacity-matched candidate with fusion gate deterministically one, reproducing unconditionally added residual behavior. |
| `dcss_residual_cdi` | CCT-G3.4 selected residual CDI predecessor. |
| `gru_baseline` | Matched recurrent quality and material-target reference. |
| `transformer` | Matched causal-attention reference. |

## Local Gates Before Colab

| Gate | Requirement |
|---|---|
| Exact control | Candidate and fusion control have equal 80,586 parameter counts, equal shapes, and fusion-control output exactly equals state readout plus token residual. |
| Retained mechanisms | Candidate and control retain the CCT-G3.4 residual, geometry, contrast, harmonic, tokenizer, and recurrence configurations. |
| Causality | Fusion at position \(t\) depends only on state/readout and source-token residual at \(t\); changing a later input must not change earlier logits. |
| Recurrent invariance | Identically initialized candidate and control have identical DCSS state trajectories and token-residual values before fusion. |
| Distinguishability | Identically initialized candidate and control produce a nonzero causal logit or loss difference on a fixed token fixture. |
| Gradient contract | Candidate has finite, nonzero causal-loss gradient on `residual_fusion.*`. Control declares exactly those parameters inactive and receives only zero or absent gradients there; all other non-exempt parameters pass strict checks. |
| Parameter and stability contract | Full five-model spread is at or below 1%; a deterministic training step has finite loss and passes existing state/memory safeguards. |

## Empirical Decision Rules

| Result | Decision |
|---|---|
| Candidate beats fusion control and CCT-G3.4 predecessor in every seed; all contract gates pass | `EARNED_FUSION_EVIDENCE`; retain the fusion mechanism. |
| Candidate does not beat fusion control and predecessor in every seed | `NO_FUSION_EVIDENCE`; do not retain fusion complexity. |
| Candidate matches or beats GRU in every seed but mean loss is above 6.664364 | `QUALITY_RECOVERY_PARTIAL`; no material-quality or scale claim. |
| Candidate beats GRU in every seed and mean loss is at or below 6.664364 | `MATERIAL_QUALITY_ADVANTAGE_EARNED`; only a separately reviewed CCT-G2.2 scale-rung proposal may then be considered. |
| Candidate fails the GRU relation in any seed | `REDESIGN_BEFORE_SCALE`. |

No outcome automatically authorizes scaling. A result applies only to this compact CPU float32 model and frozen short-context corpus contract.

## References

[1]: [CCT-G3.4 decision](CCT_G3_4_DECISION.md)  
[2]: [CCT-G3.4 pre-registration](CCT_G3_4_PREREGISTRATION.md)  
[3]: [Authoritative CCT checklist](../Todo.md)
