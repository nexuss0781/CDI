# CCT-G3.4 Pre-Registration: Selective Token-Residual Readout Quality Recovery

> **Status:** pre-registered implementation plan. This document authorizes one isolated quality-recovery mechanism, an exact parameter-preserving control, local gates, and one bounded real-data comparison. It does **not** authorize a data-scale, context, capacity, corpus, throughput, English-fluency, or production claim.

## Question

CCT-G3.1, CCT-G3.2, and CCT-G3.3 established repeated held-out value for sparse geometry, fixed contrast readout, and the harmonic memory band. Full CDI nevertheless remained above the matched GRU validation loss in every seed. [1] [2] [3]

The current language-model interface reduces the complete 48-element recurrent state through one static 48-to-4 readout before the tied vocabulary projection. The current source-token embedding affects next-token logits only after passing through the recurrent update; the model has no bounded, input-conditioned residual readout path. This is a concrete representational bottleneck at the state-to-logit boundary, not a reason to alter data scale or model width.

CCT-G3.4 asks whether a bounded selective token-residual can improve the retained full CDI configuration and recover a **material** matched quality advantage over GRU.

## One Changed Mechanism

The candidate preserves the complete CCT-G3.3 full CDI recurrence, three memory bands, harmonic band, sparse topology, Laplacian correction, fixed contrast readout, tokenizer, tied embedding/output projection, state width, and training contract. It adds only a bounded input-conditioned residual to the existing four-dimensional state readout:

\[
h_t = R\!\left(\phi(z_t)\right) + r(e_t),
\qquad
r(e_t)=\sigma(W_g e_t+b_g)\odot\tanh(W_v e_t+b_v),
\]

where \(e_t\) is the current causal source-token embedding, \(R\) is the existing 48-to-4 readout, and \(r(e_t)\in\mathbb{R}^4\). The value branch and gate branch are each four-by-four affine maps. The residual is bounded coordinatewise in \((-1,1)\), is evaluated only from the current source token, and is therefore causal.

The candidate adds 40 trainable parameters, yielding **80,550** total parameters. This remains within the frozen 1% five-model parameter tolerance. The exact control has the same 80,550 parameters and uses the same code path, but supplies \(r(e_t)=0\) deterministically. Its `token_residual.*` parameters are intentionally disconnected and must be explicitly declared inactive to the strict trainer. No control parameter is removed, resized, optimizer-frozen, or replaced.

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
| Primary metric | Token-weighted held-out validation cross-entropy; lower is better |
| Material-advantage threshold | Candidate must be at or below **6.664364** mean validation loss: 2% below the recorded matched GRU mean of 6.800372 |

## Model Matrix

| Model | Purpose |
|---|---|
| `dcss_residual_cdi` | Full retained CDI plus the bounded selective token-residual readout. |
| `dcss_residual_control` | Exact capacity-matched candidate with only residual feature values zeroed. |
| `dcss_cdi` | CCT-G3.3 full CDI predecessor, retained as the architecture-change reference. |
| `gru_baseline` | Matched recurrent quality reference and material-advantage threshold reference. |
| `transformer` | Matched causal-attention quality reference. |

## Local Gates Before Colab

| Gate | Requirement |
|---|---|
| Exact control | Candidate and residual control have identical total parameter counts and state/readout shapes; the control residual is all zeros. |
| Causality | Residual features for position \(t\) depend only on source token \(x_t\), not target \(x_{t+1}\) or later input. Target alignment and masks remain unchanged. |
| Retained recurrence | Identically initialized candidate and residual control have equal DCSS recurrent state trajectories before residual addition. |
| Distinguishability | Identically initialized candidate and residual control have a nonzero causal logit or loss difference on a fixed token fixture. |
| Gradient contract | Candidate has finite, nonzero residual-parameter causal-loss gradient. The control declares exactly `token_residual.*` parameters inactive and receives only zero or absent gradients there; all other non-exempt parameters remain finite-gradient checked. |
| Parameter fairness | Candidate/control are equal at 80,550 parameters and the full five-model spread remains at or below 1%. |
| Stability | One deterministic control training step passes the existing state-norm/energy and host-memory guards with finite loss. |

## Empirical Decision Rules

| Result | Decision |
|---|---|
| Candidate beats residual control and CCT-G3.3 full CDI in every seed; all contract gates pass | `EARNED_TOKEN_RESIDUAL_EVIDENCE`; retain the residual for the bounded candidate. |
| Candidate does not beat residual control and predecessor in every seed | `NO_TOKEN_RESIDUAL_EVIDENCE`; do not retain the residual as justified complexity. |
| Candidate matches or beats GRU in every seed but does not reach the 2% mean margin | `QUALITY_RECOVERY_PARTIAL`; no scale authorization and no dominance claim. |
| Candidate beats GRU in every seed and mean validation loss is at or below 6.664364 | `MATERIAL_QUALITY_ADVANTAGE_EARNED`; only a separate CCT-G2.2 scale-rung pre-registration may then be reviewed. |
| Candidate fails the GRU relation in any seed | Global status remains `REDESIGN_BEFORE_SCALE`. |

No CCT-G3.4 outcome automatically authorizes data-scale, context, capacity, corpus, throughput, English-fluency, or production work. The decision remains constrained to this compact CPU float32 configuration and frozen corpus contract.

## References

[1]: [CCT-G3.1 decision](CCT_G3_1_DECISION.md)  
[2]: [CCT-G3.2 decision](CCT_G3_2_DECISION.md)  
[3]: [CCT-G3.3 decision](CCT_G3_3_DECISION.md)  
[4]: [Authoritative CCT checklist](../Todo.md)
