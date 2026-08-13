# CCT-G2.1 Full-Corpus Diagnostic Decision

> **Scope:** This decision applies only to CCT-G2.1: the 1,000-step, full-corpus, three-seed diagnostic on the recorded CPU Colab runtime. It is not a fluency, scale, speed, or general-language claim.

## Verdict

**`REDESIGN_BEFORE_SCALE`**

The submitted harness emitted `EARNED_NEXT_PILOT` because CDI met its learning and best-baseline tolerance checks. The governing CCT-G2.1 gate in `Todo.md` is stricter: CDI must also **consistently match or beat the GRU**. It did not. CDI’s validation loss was higher than the GRU in each of the three seeds, so CCT-G2.2, the 3,000-step scale rung, is **not unlocked**.

| Decision basis | Result |
|---|---|
| Learning and numerical stability | Pass |
| Document/content-hash split isolation | Pass |
| Transformer tolerance: CDI within 5% | Pass |
| CDI matches or beats GRU in every seed | **Fail** |
| CCT-G2.1 transition decision | **`REDESIGN_BEFORE_SCALE`** |

## Reproducibility Contract

| Field | Recorded value |
|---|---|
| Submitted code revision | `d5a2180e6e61494140b8ff221703cef7c317ecd3` |
| Runtime | CPU Colab; Python 3.12; PyTorch 2.11.0+cpu |
| Tokenizer | EthioBBPE 2.0.0; artifact fingerprint `d78996f0aca122d74054b927902aa9bf80c2b5cf00747a7cf4327ff0f7d1a88c` |
| Corpus | 321 deduplicated documents from the governed Synaxarium manifest |
| Manifest fingerprint | `2b868a661d628ec0e4507f65ee99e79abfbed12910241f95e7660a99e97e39c8` |
| Seeds | `[11, 29, 47]` |
| Training steps / model / seed | 1,000 |
| Causal token positions / model / seed | 30,000 |
| Context / batch size | 16 / 2 |
| Optimizer learning rate | 0.01 |
| Training order | `deterministic_per_epoch_shuffle` |
| Evaluation scope | `all_held_out_batches` |

## Aggregate Results

Lower loss is better. All models used the same stated data, causal-token budget, context length, batch size, precision, optimizer family, seed list, and held-out evaluation scope.

| Model | Parameters | Mean validation loss | Mean test loss | Mean training tokens/sec |
|---|---:|---:|---:|---:|
| CDI | 80,366 | 6.8984 | 6.9267 | 366.2 |
| GRU baseline | 80,120 | **6.8004** | **6.8283** | 2,427.5 |
| Transformer | 80,172 | 6.8587 | 6.8816 | 3,739.5 |

CDI was **0.58%** above the Transformer’s mean validation loss and therefore within the declared 5% tolerance. CDI was **1.44%** above the GRU’s mean validation loss, making the GRU the best baseline in this diagnostic.

## Per-Seed Gate Check

Every recorded scalar in the submitted `latest.json` was finite. CDI training loss decreased in all three seeds. The failure is therefore a repeatable quality relation, not a NaN/Inf or loss-divergence event.

| Seed | CDI validation loss | GRU validation loss | CDI relative gap to GRU | Transformer validation loss | CDI relative gap to Transformer | CDI matches/beats GRU? |
|---:|---:|---:|---:|---:|---:|---|
| 11 | 6.8803 | 6.7904 | +1.32% | 6.8082 | +1.06% | No |
| 29 | 6.9252 | 6.8193 | +1.55% | 6.9015 | +0.34% | No |
| 47 | 6.8897 | 6.7913 | +1.45% | 6.8665 | +0.34% | No |

The 300-step bounded result had CDI slightly ahead of the GRU on mean validation loss. At the larger full-corpus diagnostic budget, the direction reversed in every seed. That is enough evidence to stop scale expansion under the predeclared rule.

## Required Next Action

Do **not** add training steps, context, model capacity, corpus size, or performance optimization to compensate for this result. The only permitted next experiment is **one controlled CCT-G3.1 architecture ablation**.

The first ablation must pre-register one mechanism and change only that mechanism while holding the tokenizer, governed split, model budget, causal token budget, optimizer, context, precision, batch size, and seeds fixed. The initial candidate is the existing geometry-free CDI variant against full CDI, GRU, and Transformer. Its purpose is diagnostic: determine whether the current geometry contribution helps, harms, or is indistinguishable at the failed G2.1 setting.

## Evidence Provenance

The decision was derived from the submitted `REPORT.md` output and submitted `latest.json` payload. The canonical raw result directory remains `results/colab_stage2a_full_corpus/` in the Colab execution environment; its final JSON includes the configuration, manifest, per-seed records, and harness decision used here.
