# Stage E Preregistration — Controlled DCSS-CDI Ablations

## Study scope and data limitation

This Stage E study uses the frozen Stage D **repository-local synthetic corpus** and the same zero-dependency tokenizer. It is therefore an **engineering, reproducibility, and controlled synthetic-sequence study**, not a real-corpus language-quality evaluation. Any conclusion about natural-language quality, transfer, or general language modeling is explicitly out of scope. The frozen corpus, tokenizer fingerprint, data ordering policy, float32 CPU precision, AdamW family, and fixed-token stopping rule are recorded in every run manifest.

## Frozen matrix

| ID | Name | Single intended difference from full DCSS-CDI |
|---|---|---|
| `T` | Transformer | Small causal Transformer baseline. |
| `V2` | Legacy adapter | Historical-style compact recurrent reference adapter. |
| `U` | Ungated recurrence | Selective gates replaced by fixed non-content-dependent controls. |
| `G` | Geometry-free | Stage B matrix-free geometric field disabled. |
| `H` | No harmonic band | Harmonic/slow state update and readout contribution disabled. |
| `E` | Explicit Euler | Exact pairwise block exponential replaced by an explicit Euler update. |
| `C` | Unconstrained cochain | Structured correction supplemented with a learned unconstrained vertex mixing map. |
| `F` | Full DCSS-CDI | Frozen Stage D selective, geometric, exact, three-band engine. |

All DCSS variants retain the same tokenizer, state width, chunk length, batch size, optimizer, learning rate, token budget, seed list, train examples, validation examples, and stopping rule. The `C` ablation is intentionally exempt from the full-engine no-dense allocation claim because it is a named diagnostic with a small learned vertex mixing map; the allocation gate applies to `F`.

## Evaluation protocol

Three headline seeds (`1, 2, 3`) train each matrix member for 100 deterministic optimizer steps. Validation loss, token count, finite status, parameter count, wall time, and a loss curve are recorded per run. The primary synthetic quality measure is final held-out masked cross-entropy. All statistics report mean, standard deviation, median, and a seeded bootstrap 95% interval.

Sequence scaling uses lengths `8, 16, 32, 64, 128, 256`, generated from the fixed tokenizer’s valid IDs. These small lengths are appropriate to the CPU nano tier. The report explicitly does not extrapolate the measured exponent to the 4k–8k scale target in the Stage E specification. Streaming tests use 512 tokens and record serialised-state fingerprint/continuation equality, persistent state bytes, and warm-up-excluded latency. Runtime allocation tracing rejects dense sequence-square tensors and full-state-square tensors in `F`.

## Frozen thresholds

| Category | Gate | Threshold |
|---|---|---:|
| Engineering | Full DCSS forward-memory exponent | `≤ 1.20` across the measured range. |
| Engineering | Full DCSS forward-time exponent | `≤ 1.25` across the measured range. |
| Engineering | Runtime dense sequence/full-state allocation | None. |
| Engineering | Streaming state growth | Exactly constant structured-state byte count. |
| Engineering | Finite status | Zero non-finite headline runs. |
| Quality | Full synthetic validation floor | Mean final loss `≤ 4.0` nats. |
| Quality | Non-inferiority | Full mean loss no more than `0.50` nats above the best internal DCSS ablation mean. |
| Long context | Harmonic retention diagnostic | Full `F` harmonic retained-norm ratio after the fixed 32-token delay must be at least `0.50`; `H` must have zero harmonic state by construction. |
| Scientific | Matching | Identical declared tokenizer/data/precision/optimizer/token budget for every matrix member. |
| Scientific | Reproducibility | Deterministic rerun of `F`, seed 1, matches final loss within `1e-6`. |

The speed-versus-legacy and 4k Transformer memory targets are **not measured** under this fixed nano CPU configuration. They are retained as `NOT_MEASURED` rather than inferred or silently marked passed.

## Decision rule

The study may be an engineering **conditional result** if its measured scaling, allocation, stability, and reproducibility gates pass but the synthetic-only quality evidence cannot justify the real-corpus claims required for a full Stage F capability expansion. The resulting report will include all negative findings and will retain `stage_f_implementation_allowed: false` unless the user explicitly approves the result and decides the permitted Stage F scope.
