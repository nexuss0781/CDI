# CCT-G3.4 Decision: Selective Token-Residual Quality Recovery

> **Mechanism decision:** `EARNED_TOKEN_RESIDUAL_EVIDENCE`. **Quality decision:** `QUALITY_RECOVERY_PARTIAL`. **Scale authorization:** `False`.
>
> The candidate establishes a repeated, matched advantage over the prior full CDI configuration and over GRU, but it does **not** meet the pre-registered 2% material-quality margin. It is not a dominance result, not a scale authorization, and not an English-fluency, long-context, speed, or production claim.

## Decision Summary

CCT-G3.4 tested a single bounded change at the CDI state-to-logit interface: a four-dimensional input-conditioned token residual added to the existing 48-to-4 state readout. The recurrent state-space, fixed contrast readout, sparse geometry, harmonic memory band, tokenization, corpus, context, steps, optimizer, seeds, and evaluation scope were held fixed. The exact control retained the same 40 residual parameters but supplied zero residual feature values. [1]

All pre-registered mechanism gates passed. The candidate had lower held-out validation loss than both the exact zero-residual control and CCT-G3.3 predecessor in every seed. It also matched or beat GRU in every seed, but its 0.8356% mean advantage over GRU was below the pre-registered 2% material threshold. The correct quality decision is therefore `QUALITY_RECOVERY_PARTIAL`, with scale still blocked. [1]

| Gate | Result | Evidence |
|---|---|---|
| Complete finite five-model, three-seed matrix | **PASS** | All 15 records are complete and finite. |
| Candidate learning | **PASS** | Candidate training loss decreased in seeds 11, 29, and 47. |
| Parameter fairness | **PASS** | Candidate/control have 80,550 parameters; five-model spread is 0.5367%, below 1%. |
| Exact residual control | **PASS** | Control retains all residual parameters but supplies zero residual values; local state and inactive-gradient gates passed before the formal run. |
| Candidate versus exact residual control | **PASS** | Candidate lower held-out validation loss in all three seeds. |
| Candidate versus CCT-G3.3 predecessor | **PASS** | Candidate lower held-out validation loss in all three seeds. |
| Candidate versus GRU | **PASS** | Candidate lower held-out validation loss in all three seeds. |
| 2% material-GRU margin | **FAIL** | Candidate mean is 6.743546; required threshold is at or below 6.664364. |
| Host-memory guard | **PASS** | Peak recorded host memory is 1.87055 GiB, below 11 GiB. |

## Seed-Level Quality Result

| Seed | Residual CDI | Zero-residual control | CCT-G3.3 CDI | GRU | Residual CDI improvement versus GRU |
|---:|---:|---:|---:|---:|---:|
| 11 | 6.713290 | 6.818117 | 6.818117 | 6.790431 | 0.077141 |
| 29 | 6.765666 | 6.885216 | 6.885216 | 6.819336 | 0.053670 |
| 47 | 6.751683 | 6.839128 | 6.839128 | 6.791349 | 0.039666 |
| **Mean** | **6.743546** | **6.847487** | **6.847487** | **6.800372** | **0.056826** |

The selective token residual improved held-out validation loss by **0.103941** against both the exact control and CCT-G3.3 predecessor. It improved mean loss by **0.056826** versus GRU, equal to a **0.8356%** matched mean advantage. The remaining distance to the pre-registered material target is **0.079182** validation-loss units.

## Aggregate Matched Results

| Model | Parameters | Mean validation loss | Mean test loss | Mean validation accuracy | Mean tokens/sec |
|---|---:|---:|---:|---:|---:|
| `dcss_residual_cdi` | 80,550 | 6.743546 | 6.774044 | 0.097815 | 213.5 |
| `dcss_residual_control` | 80,550 | 6.847487 | 6.877744 | 0.088383 | 217.7 |
| `dcss_cdi` | 80,510 | 6.847487 | 6.877744 | 0.088383 | 219.0 |
| `gru_baseline` | 80,120 | 6.800372 | 6.828304 | 0.088686 | 1,847.0 |
| `transformer` | 80,172 | 6.856126 | 6.880668 | 0.091536 | 2,936.2 |

The current candidate has earned retention as the selected compact CDI configuration under this frozen protocol. Its token loop remains substantially slower than GRU and Transformer; no runtime conclusion or optimization work is authorized from this result.

## Reproducibility Record

| Field | Submitted value |
|---|---|
| Run status | `COMPLETE` |
| Result format | `dcss-cdi-cct-g3-4-token-residual-v1` |
| Result fingerprint | `ffd6237a6ff05b48ad51c1225044d51ad2e42d9c453ec92af4874b1bb22ef67b` |
| Code revision | `49f8b0caec82623e26f1c087421e24e8ba848bfb` |
| Dataset | `Nexuss0781/synaxarium`; 321 documents; MIT asserted by source card |
| Data manifest | `947d152f129f2fd91433fa9b64574c674f9aea8e472ef3cb4fd7dafa5a2bd0d9` |
| Tokenizer artifact | EthioBBPE fingerprint `d78996f0aca122d74054b927902aa9bf80c2b5cf00747a7cf4327ff0f7d1a88c` |
| Seeds | `[11, 29, 47]` |
| Training budget | 1,000 steps/model/seed; 30,000 causal positions/model/seed |
| Context and batches | Chunk length 16; batch size 2; 32 chunks/document |
| Optimizer and precision | AdamW; learning rate 0.01; CPU float32 |
| Training/evaluation | Deterministic per-epoch shuffle; all held-out validation and test batches |
| Environment | Python 3.12.13; PyTorch 2.11.0+cpu |
| Host-memory guard | 11 GiB maximum; 1.87055 GiB peak |

## Boundary and Next Action

CCT-G3 has now established repeated held-out contribution evidence for the fixed contrast readout, sparse geometry, harmonic memory band, and bounded selective token residual. These mechanisms are retained for later reviewed work.

The current candidate clears the original per-seed GRU relation but does not clear the newer 2% material-quality margin that defines the requested substantial advantage. The next action is **not** CCT-G2.2, larger data, longer context, capacity expansion, performance work, or generation claims. Any further quality-recovery attempt must be user-approved and separately pre-registered, retain the four earned mechanisms, and target the remaining 0.079182 validation-loss distance without altering the frozen comparison contract.

## References

[1]: [CCT-G3.4 pre-registration](CCT_G3_4_PREREGISTRATION.md)  
[2]: [CCT-G3.3 decision](CCT_G3_3_DECISION.md)  
[3]: [Authoritative CCT checklist](../Todo.md)  
[4]: [CCT evidence index](CCT_EVIDENCE_INDEX.md)
