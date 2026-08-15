# CCT-G3.6 Decision: Bounded Quality Continuation

> **Decision:** `EARNED_BOUNDED_CONTINUATION`. **Training eligibility:** `ELIGIBLE_FOR_REVIEWED_NEXT_RUNG`. **Scale authorization:** `False`.

CCT-G3.6 continued the retained CCT-G3.4 residual CDI from the 1,000-step evidence rung to 1,500 total steps without changing the architecture, tokenizer, corpus, context, optimizer, seeds, precision, or evaluation scope. The autonomous CPU run completed under the 11 GiB guard. All nine model/seed records were finite, CDI learning passed in every seed, CDI beat GRU in every seed, and validation loss improved over the 1,000-step retained-CDI reference.

This is the first bounded continuation result that passes the pre-registered progress and quality gates. It **does not authorize the 3,000-step scale ladder automatically**.

## Results

| Model | Parameters | Mean validation loss | Mean test loss | Mean tokens/sec |
|---|---:|---:|---:|---:|
| `dcss_residual_cdi` | 80,550 | **6.576891** | 6.600720 | 277.4 |
| `gru_baseline` | 80,120 | 6.642577 | 6.667879 | 2,546.6 |
| `transformer` | 80,172 | 6.694665 | 6.718255 | 4,394.8 |

The retained CDI improved by **0.166655** validation-loss units over its 1,000-step reference of 6.743546. It beat GRU by **0.065686** mean validation-loss units, an approximately **0.989%** lower loss. It also passed the separately reported 2% material target of 6.664364 by reaching 6.576891, approximately **1.313% below the target threshold**.

| Seed | CDI validation loss | GRU validation loss | CDI advantage |
|---:|---:|---:|---:|
| 11 | 6.584620 | 6.633786 | 0.049166 |
| 29 | 6.584950 | 6.656888 | 0.071938 |
| 47 | 6.561102 | 6.637055 | 0.075953 |

The CDI-versus-GRU relation is therefore repeated at seed level, not only in the mean. CDI training loss decreased in all three seeds, and the full held-out evaluation scope was used.

## Gate Results

| Gate | Result | Evidence |
|---|---|---|
| Complete finite evidence | **PASS** | Nine model/seed records are present and finite. |
| CDI learning | **PASS** | Training loss decreased in seeds 11, 29, and 47. |
| CDI versus GRU | **PASS** | CDI beat GRU in all three seeds. |
| Progress over 1,000-step CDI reference | **PASS** | 6.576891 is below 6.743546. |
| 2% material target | **PASS** | 6.576891 is below 6.664364. |
| Parameter fairness | **PASS** | CDI 80,550; GRU 80,120; Transformer 80,172. |
| Memory | **PASS** | Peak host memory 0.83606 GiB, below 11 GiB. |
| Regression | **PASS** | 300 tests passed after the G3.6 harness and final source state. |

## Eligibility Boundary

The retained CDI is now **eligible for a separately reviewed next quality rung**. This means the training path is technically and empirically ready for one controlled continuation. It does not mean that the model is fluent, perfect, English-ready, faster than the baselines, or authorized for unrestricted scaling.

The next rung must preserve the retained CCT-G3.4 architecture, the EthioBBPE tokenizer, the governed corpus, the document-isolated split, the fair baseline matrix, and the 11 GiB guard. Any increase in steps, corpus size, context length, capacity, or language domain requires its own pre-registration. In particular, CCT-G2.2's 3,000-step ladder remains blocked until a separate transition record reviews this result.

## Reproducibility

| Field | Submitted value |
|---|---|
| Result format | `dcss-cdi-cct-g3-6-bounded-quality-v1` |
| Code revision | `39f068974cf8297bdb2f54c0921931c0442adbcd` |
| Result fingerprint | `807d4ea6d599c768571ebb7ff533bfad3684a7de9b144f6ee5ada22f3c35af8b` |
| Dataset manifest | `947d152f129f2fd91433fa9b64574c674f9aea8e472ef3cb4fd7dafa5a2bd0d9` |
| Tokenizer | EthioBBPE fingerprint `d78996f0aca122d74054b927902aa9bf80c2b5cf00747a7cf4327ff0f7d1a88c` |
| Seeds | `[11, 29, 47]` |
| Total steps | 1,500 per model/seed |
| Chunks and context | 32 chunks/document; chunk length 16; batch size 2 |
| Optimizer and precision | AdamW; learning rate 0.01; CPU float32 |
| Evaluation | Deterministic per-epoch shuffle; all held-out batches |
| Peak memory | 0.83606 GiB under 11 GiB limit |
| Regression | 300 passing tests |

## References

[1]: [CCT-G3.6 pre-registration](CCT_G3_6_PREREGISTRATION.md)  
[2]: [Performance readiness](PERFORMANCE_READINESS.md)  
[3]: [CCT-G3.4 decision](CCT_G3_4_DECISION.md)  
[4]: [Authoritative CCT checklist](../Todo.md)
