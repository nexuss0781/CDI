# CCT-G3.2 Pre-Registration: Contrast-Readout Contribution

> **Status:** pre-registered implementation plan. This document authorizes one isolated readout-control implementation and one bounded experiment. It does not authorize a quality-scale, context, capacity, corpus, throughput, or fluency claim.

## Question

CCT-G3.1 established that, with the fixed zero-sum contrast readout enabled, full CDI outperformed its exact geometry-free counterpart in all three seeds. The remaining ambiguity is whether the **readout pathway itself** improves language prediction or whether the observed G3.1 relation is specific to the geometry correction acting through that pathway.

CCT-G3.2 isolates the readout contribution before any corrected G2.1 quality rerun.

## One Changed Mechanism

The control uses the identical DCSS recurrence, sparse topology, Laplacian correction, geometry gates, state dimensions, tied embedding/output projection, and 48-to-4 readout layer as full CDI. It changes only the feature values supplied by the readout:

\[
\phi_{\mathrm{full}}(z_b)=\left[\operatorname{mean}_V(z_b), Q^\top z_b\right],
\qquad
\phi_{\mathrm{mean\ control}}(z_b)=\left[\operatorname{mean}_V(z_b), 0\right].
\]

The control retains the contrast slots and their readout weights, so it has the same **80,510 total parameters** as full CDI. The contrast feature values are deterministically zero; no parameter is removed, resized, or added. Because the remaining mean is invariant to the Laplacian correction, the geometry edge parameters are intentionally disconnected from causal loss in this control and must be declared inactive by the strict trainer.

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

## Model Matrix

| Model | Purpose |
|---|---|
| `dcss_cdi` | Full mean-plus-contrast readout with sparse geometry. |
| `dcss_mean_readout_control` | Mean-only feature values with unchanged recurrence, geometry computation, 48-to-4 readout layer, and parameter count. |
| `dcss_geometry_free` | Mean-plus-contrast readout with only Laplacian correction disabled. |
| `gru_baseline` | Matched recurrent quality reference. |
| `transformer` | Matched causal-attention quality reference. |

## Local Gates Before Colab

| Gate | Requirement |
|---|---|
| Exact feature control | Mean-control feature tensor has the same shape as full CDI but zero contrast entries. |
| Causal behavior | Forward logits and causal loss are finite and preserve target/mask shape. |
| Gradient contract | Full CDI has a finite, nonzero geometry-edge causal-loss gradient; mean-control declares only the geometry edge as intentionally inactive; every other active parameter remains finite-gradient checked. |
| Parameter fairness | Full, mean-control, and geometry-free CDI have identical total parameter counts; all five models remain within 1%. |
| Distinguishability | Identically initialized full and mean-control CDI produce a nonzero causal logit/loss difference on a fixed token fixture. |

## Empirical Decision Rules

| Result | Decision |
|---|---|
| Full CDI has lower validation loss than mean-control CDI in every seed; all finite, learning, and parameter gates pass | `EARNED_READOUT_EVIDENCE`. |
| Full CDI does not beat mean-control CDI in every seed | `NO_READOUT_EVIDENCE`; do not retain the contrast readout as justified complexity without a bounded redesign review. |
| Full CDI also beats geometry-free CDI in every seed | G3.1 geometry evidence is independently re-confirmed under the five-model matrix. |
| Full CDI still fails the three-seed GRU relation | Global status remains `REDESIGN_BEFORE_SCALE`; no G2.2 or G4/G5 work is permitted. |

No G3.2 outcome directly authorizes scaling. A successful readout result can only authorize a **corrected CCT-G2.1 quality rerun** at the same 1,000-step budget after its separate pre-registration.

## References

[1]: [CCT-G3.1 decision](CCT_G3_1_DECISION.md)  
[2]: [CCT-G3.1 pre-registration](CCT_G3_1_PREREGISTRATION.md)  
[3]: [Authoritative CCT checklist](../Todo.md)
