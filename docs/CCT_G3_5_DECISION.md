# CCT-G3.5 Decision: State-Conditioned Token-Residual Fusion

> **Mechanism decision:** `NO_FUSION_EVIDENCE`. **Quality decision:** `REDESIGN_BEFORE_SCALE`. **Scale authorization:** `False`.
>
> The state-conditioned fusion gate did not add value to the retained CCT-G3.4 residual CDI under the frozen three-seed contract. The selected CCT-G3.4 residual CDI remains the valid compact baseline: it beats GRU in every seed, while the G3.5 fusion candidate does not beat its exact control. This decision does not authorize larger training, data expansion, context expansion, capacity changes, performance claims, or production claims.

## Submitted Result

CCT-G3.5 changed only the combination of the existing DCSS state readout and CCT-G3.4 source-token residual. The candidate used a bounded state-conditioned sigmoid gate. The exact control retained the same 36 fusion parameters but used a deterministic all-one gate, reproducing the CCT-G3.4 additive residual path. The recurrent state, residual values, tokenizer, corpus, context, steps, optimizer, precision, seeds, and held-out evaluation were unchanged. [1]

The formal submission is complete: it contains all 15 finite model/seed records, the complete frozen configuration, the governed manifest and tokenizer fingerprints, the resource record, and a normally closed JSON result. [2]

| Gate | Result | Evidence |
|---|---|---|
| Complete finite evidence | **PASS** | All 15 records are finite and present. |
| Candidate learning | **PASS** | Candidate training loss decreased in all three seeds. |
| Parameter fairness | **PASS** | Candidate/control have 80,586 parameters; five-model spread is 0.58%, below 1%. |
| Candidate versus exact fusion control | **FAIL** | Candidate lower in only seed 47; it lost in seeds 11 and 29. |
| Candidate versus CCT-G3.4 predecessor | **FAIL** | Same per-seed comparison: lower only in seed 47. |
| Candidate versus GRU | **PASS** | Candidate beat GRU in all three seeds. |
| Material 2% GRU margin | **FAIL** | Candidate mean loss 6.744694; target is at or below 6.664364. |
| Host-memory guard | **PASS** | Peak recorded host memory was 2.59923 GiB, below 11 GiB. |

## Aggregate Results

| Model | Parameters | Mean validation loss | Mean test loss | Mean tokens/sec |
|---|---:|---:|---:|---:|
| `dcss_fused_residual_cdi` | 80,586 | 6.744694 | 6.7755 | 229.0 |
| `dcss_fusion_control` | 80,586 | 6.7435 | 6.7740 | 232.4 |
| `dcss_residual_cdi` | 80,550 | 6.7435 | 6.7740 | 233.2 |
| `gru_baseline` | 80,120 | 6.8004 | 6.8283 | 2,112.2 |
| `transformer` | 80,172 | 6.8561 | 6.8807 | 3,735.1 |

The candidate was **0.001148 worse** than both the exact fusion control and the CCT-G3.4 predecessor on mean validation loss. It nevertheless remained **0.0557 loss units better than GRU**, or approximately 0.82% better on the submitted rounded aggregate. This is a useful quality result for the retained residual CDI, but it is not evidence that the new fusion mechanism helps.

The control and predecessor values agree to the displayed precision, as expected: the fusion-one control was designed to reproduce the unconditionally added residual path. The formal mechanism gate therefore correctly fails rather than treating a small mean difference as evidence.

## Decision and Training Boundary

CCT-G3.5 does not justify retaining the new fusion gate. The **CCT-G3.4 residual CDI** remains the selected compact candidate because it beat GRU in all three seeds and had already earned `EARNED_TOKEN_RESIDUAL_EVIDENCE`, while the fusion extension earned `NO_FUSION_EVIDENCE`.

The user-requested 2% material-quality target remains unmet. Before any new training ladder, the execution path must first receive a performance-readiness audit. That audit must measure where time is spent, verify that any optimization is semantics-preserving, and rerun the full regression suite before a training command is considered eligible. This is a performance gate, not a license to change the quality protocol.

No automatic next architecture modification is authorized by this decision. The next bounded work item is **performance-first profiling and, only if justified, semantics-preserving runtime repair**. Larger corpora, English scaling, 3,000-step training, longer context, capacity expansion, and fluency claims remain blocked until a separately reviewed transition changes the gate.

## Reproducibility Record

| Field | Submitted value |
|---|---|
| Result format | `dcss-cdi-cct-g3-5-residual-fusion-v1` |
| Code revision | `d9b3802b285d8bc8c64a10ffc5b363c2403cfebf` |
| Result fingerprint | No `fingerprint` JSON field was present in the submitted artifact; reproducibility is bound by code revision, manifest, tokenizer fingerprint, full configuration, and complete JSON closure |
| Dataset | `Nexuss0781/synaxarium`; 321 documents; MIT asserted by source card |
| Data manifest | `947d152f129f2fd91433fa9b64574c674f9aea8e472ef3cb4fd7dafa5a2bd0d9` |
| Tokenizer | EthioBBPE fingerprint `d78996f0aca122d74054b927902aa9bf80c2b5cf00747a7cf4327ff0f7d1a88c` |
| Seeds | `[11, 29, 47]` |
| Steps | 1,000 per model/seed |
| Context | Chunk length 16; 32 chunks/document; batch size 2 |
| Optimizer and precision | AdamW; learning rate 0.01; CPU float32 |
| Evaluation | Deterministic per-epoch shuffle; all held-out batches |
| Memory | 11 GiB guard; 2.59923 GiB peak |

## References

[1]: [CCT-G3.5 pre-registration](CCT_G3_5_PREREGISTRATION.md)  
[2]: Submitted CCT-G3.5 Colab artifact reviewed in the session; the complete report and JSON were not checked into the repository.  
[3]: [CCT-G3.4 decision](CCT_G3_4_DECISION.md)  
[4]: [Authoritative CCT checklist](../Todo.md)
