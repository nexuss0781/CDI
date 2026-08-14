# CDI Step-by-Step Development Roadmap

**Prepared for:** CDI project review  
**Author:** Manus AI  
**Operating rule:** We complete and review one stage at a time. We do **not** advance because of hope, a pretty sample, or a single loss curve. We advance only when the stage’s saved evidence passes its declared gate.

> The goal is not to promise that CDI will replace Transformers. The goal is to find out, rigorously and quickly, whether CDI can become a competitive language model with a clear quality, long-context, or efficiency advantage.

## 1. Current starting point

The EthioBBPE migration is complete on `master`, and the real-data pilot established an important but narrow result. The 48-state CDI model learned held-out Amharic Synaxarium text after the tokenizer mismatch was fixed. At 300 matched CPU steps and three seeds, its validation cross-entropy was 7.1700, compared with 7.1923 for the GRU baseline and 7.0507 for the Transformer. CDI therefore learned and slightly exceeded the simple GRU on this task, but it did **not** beat the Transformer and was much slower in the current Python-oriented implementation.

| Current fact | Meaning | Decision |
|---|---|---|
| EthioBBPE, training, checkpoint, and inference now use one saved tokenizer artifact | The prior silent token-ID collapse has been removed | The next training result is meaningful |
| CDI is 1.69% behind the small Transformer after the 300-step real-text pilot | The architecture is viable enough for a larger controlled test | Do not discard CDI |
| CDI runs about 15× slower than the Transformer in the CPU pilot | The implementation is not ready to claim speed | Do not scale blindly |
| Context was 16 tokens and state size was 48 | The test did not assess long-context capability | Build a larger controlled experiment |

## 2. Non-negotiable experiment rules

Every experiment must use the same rules. These rules stop us from accidentally “winning” through different data, more tokens, or a favorable seed.

| Rule | What we record | Why it matters |
|---|---|---|
| One frozen tokenizer artifact | `EthioBBPE==2.0.0`, artifact file, tokenizer fingerprint, vocabulary size | A checkpoint must decode with exactly the same tokenizer used in training |
| Document-first splitting | Document IDs and SHA-256 content hashes for train/validation/test | Prevents the same text appearing in both training and evaluation |
| Equal comparison budget | Processed tokens, parameter count, optimizer, precision, context length, batch size, seeds | Makes CDI-vs-baseline results interpretable |
| Three seeds | Seed-level loss, mean, and standard deviation | Rejects one lucky initialization |
| Blind test set | Validation chooses settings; test is read once for the final stage result | Prevents tuning to the test data |
| Numerical evaluation first | Cross-entropy, perplexity, accuracy, gradient statistics, throughput, memory | A short generated sample is not evidence of language learning |
| Immutable run evidence | Config, manifest, Git commit, environment, checkpoints, result JSON | Lets us reproduce or audit any claim |

## 3. Stage sequence

The stages below are in order. We will stop after every stage so you can inspect the report before you decide to proceed.

| Stage | Main question | Primary deliverable | Gate to move on |
|---:|---|---|---|
| 0 | Is the codebase reproducible? | Clean test result and environment manifest | All tests pass; tokenizer artifact round-trips |
| 1 | Does the existing result reproduce on Colab? | Three-seed 300-step pilot report | CDI learns in all seeds; no data/tokenizer leakage |
| 2 | Does CDI retain its quality at a larger real-data budget? | Full-corpus matched baseline run | CDI stays within 5% of Transformer validation loss and consistently matches or beats GRU |
| 3 | Does the unique CDI geometry help? | Full CDI vs geometry-free ablation report | A repeatable quality, stability, or retention difference exists |
| 4 | Can CDI handle longer context? | Context ladder and retention benchmark | No material degradation against matched baselines at the intended context |
| 5 | Can CDI become computationally competitive? | Profile plus optimized-kernel benchmark | Measured improvement on the same hardware and workload |
| 6 | Is a larger corpus justified? | Scale-readiness review | Stages 2–5 pass, including quality and speed/long-context value |
| 7 | Does scaled CDI generate coherent text? | Fixed-prompt blind generation evaluation | Numerical metrics improve and continuations are coherent in-domain |

## 4. Stage 0 — clean, reproducible installation

Run this first in a fresh Colab notebook. This stage does not train a model. It confirms that everyone is running the same repository, branch, package versions, and test suite.

```bash
# Safe to rerun in a fresh or partially failed CPU Colab runtime.
%cd /content
!rm -rf CDI
!git clone --branch master --single-branch https://github.com/nexuss0781/CDI.git CDI
%cd /content/CDI
!python -m pip install --upgrade pip
!python -m pip install -r requirements.txt
!python -c "import ethiobbpe; print('EthioBBPE installed:', ethiobbpe.__file__)"
!git branch --show-current
!git rev-parse HEAD
!python -m pytest -q
```

**Expected evidence:** `288 passed`, the branch `master`, the current commit, and a printed EthioBBPE installation path. If this setup cell fails, stop immediately; send the full error output rather than changing model hyperparameters.

## 5. Stage 1 — reproduce the completed 300-step real-data pilot

This stage verifies that the existing result is not specific to one machine. It uses the already committed harness and public in-domain Synaxarium dataset. Do not change the code, seed list, tokenizer, context length, learning rate, or comparison models during this reproduction.

```bash
%cd /content/CDI
!PYTHONPATH=. python benchmarks/ethiobbpe_synaxarium_pilot.py \
  --steps 300 \
  --document-limit 60 \
  --chunks-per-document 8 \
  --chunk-length 16 \
  --batch-size 2 \
  --eval-batches 12 \
  --learning-rate 0.01 \
  --relative-loss-tolerance 0.10 \
  --output-dir results/colab_reproduction_300steps

!cat results/colab_reproduction_300steps/REPORT.md
```

| Stage 1 check | Pass condition | If it fails |
|---|---|---|
| Data gate | No duplicate content, no split leakage, and the EthioBBPE fingerprint matches the recorded artifact | Repair source ingestion or tokenizer loading; do not judge CDI |
| Stability gate | No NaN/Inf; CDI loss falls in all three seeds | Inspect learning rate, precision, and state update stability |
| Reproduction gate | CDI remains within 5% validation loss of the Transformer and close to the recorded loss range | Compare Colab hardware/version; investigate nondeterminism or dependency drift |

**Stop here.** You check the resulting report. We do not alter the architecture until the reproduction is understood.

## 6. Stage 2 — full-corpus, matched real-data pilot

Use the complete deduplicated Synaxarium corpus only after Stage 1 passes. This tests whether CDI retains its quality signal when it sees substantially more document variation. It is still an **in-domain Amharic pilot**, not evidence of English general language modeling.

The Stage 2 harness now uses all **321 unique** documents available after removing 45 exact duplicate-content records from the 366-row source. It supports deterministic per-epoch batch shuffling and complete held-out evaluation, so it no longer repeatedly trains on the first small set of batches or evaluates a short held-out prefix. The Transformer and GRU receive the same shuffled data order, total causal token positions, tokenizer, context length, precision, and optimizer family for each seed.

### Stage 2A — CPU-safe full-corpus diagnostic

Run this exact command first. It is a bounded three-seed, full-corpus diagnostic at 1,000 steps; it is **not** the final scaling claim. It will take longer than Stage 1 because it evaluates every validation and test batch.

```bash
%cd /content/CDI
# CCT-G0 validated the master checkout and installed requirements.
!PYTHONPATH=. python benchmarks/ethiobbpe_synaxarium_pilot.py \
  --steps 1000 \
  --document-limit 321 \
  --chunks-per-document 32 \
  --chunk-length 16 \
  --batch-size 2 \
  --eval-batches 0 \
  --shuffle-training-batches \
  --learning-rate 0.01 \
  --relative-loss-tolerance 0.05 \
  --output-dir results/colab_stage2a_full_corpus
!cat results/colab_stage2a_full_corpus/REPORT.md
```

Do not begin Stage 2B (3,000 steps) until Stage 2A is reviewed. The completed Stage 2A execution met the harness-level tolerance but failed the stricter CCT gate because CDI was above GRU validation loss in all three seeds. Its recorded decision is `REDESIGN_BEFORE_SCALE` in `docs/CCT_G2_1_DECISION.md`; Stage 2B is not authorized. The required next activity is one controlled CCT-G3.1 mechanism ablation under the same comparison contract.

| Stage 2 measurement | Required report |
|---|---|
| Learning quality | Train/validation/test cross-entropy, perplexity, and token accuracy per seed |
| Generalization | Training-to-validation gap and one final held-out test score |
| Stability | Gradient norm percentiles, NaN/Inf count, DCSS state-norm percentiles |
| Fairness | Parameter count, token budget, batch size, sequence length, precision, hardware |
| Efficiency | Training tokens/sec, generation tokens/sec, peak memory |

**Stage 2 pass rule:** CDI must be within **5%** of the Transformer’s mean validation cross-entropy across three seeds, and it must consistently match or beat the GRU. A result that gets worse as the token budget grows means the current CDI capacity, optimization, or readout needs redesign before larger pretraining.

## 7. Stage 3 — Architecture-Value Evidence

### Stage 3.1 — Completed Geometry-Observability Ablation

CCT-G3.1 is complete. Under the frozen 321-document, three-seed, 1,000-step contract, full CDI had lower held-out validation loss than its exact geometry-free counterpart in all three seeds. The mean improvement was `0.014489`, the parameter spread was 0.49%, and the 11 GiB host-memory guard recorded 1.8596 GiB peak memory. The mechanism decision is `EARNED_GEOMETRY_EVIDENCE`.

This does **not** authorize scaling. Full CDI remained above GRU validation loss in all three seeds, so the global quality decision remains `REDESIGN_BEFORE_SCALE`. See `docs/CCT_G3_1_DECISION.md`. Do not rerun G3.1 or begin the 3,000-step, context, capacity, corpus, or performance ladders.

### Stage 3.2 — Completed Readout-Contribution Ablation

CCT-G3.2 is complete. Full CDI beat the capacity-matched mean-readout control in all three seeds, with a 0.057296 mean validation-loss improvement. It also re-confirmed the sparse-geometry improvement against the geometry-free control. The five-model matrix remained parameter-matched and peak guarded memory was 1.8634 GiB. The mechanism verdict is `EARNED_READOUT_EVIDENCE`.

This does **not** authorize scaling. Full CDI remained above GRU validation loss in all three seeds, so the global quality decision remains `REDESIGN_BEFORE_SCALE`. See `docs/CCT_G3_2_DECISION.md`.

### Stage 3.3 — Completed Harmonic-Memory-Band Ablation

CCT-G3.3 is complete. Full CDI had lower held-out validation loss than the exact parameter-preserving harmonic-disabled control in all three seeds, with a **0.031414** mean validation-loss improvement. The control's harmonic state, energy, and harmonic gradient were zero in the submitted fixed-held-out diagnostics, while full CDI retained finite active harmonic dynamics. Sparse geometry was independently re-confirmed against geometry-free CDI. The mechanism decision is `EARNED_HARMONIC_EVIDENCE`.

This does **not** authorize scaling. Full CDI remained above GRU validation loss in all three seeds, with a 0.6928% mean gap. The global quality decision remains `REDESIGN_BEFORE_SCALE`; do not rerun G3.3, begin the 3,000-step ladder, expand context or corpus, change capacity, or begin speed work. See `docs/CCT_G3_3_DECISION.md`.

The next action is not an automatic scale experiment. CCT-G3.4 is the separately pre-registered quality-recovery diagnostic below; it retains all CCT-G3 mechanisms and changes only the state-to-logit interface.

### Stage 3.4 — Selective Token-Residual Quality Recovery

CCT-G3.4 is pre-registered and locally validated. It adds a bounded, input-conditioned four-dimensional residual to the existing 48-to-4 CDI state readout. It does **not** alter the recurrent state-space, contrast readout, geometry, harmonic memory, tokenizer, data, context, model width, optimizer, or seed protocol. The exact capacity-matched control retains the same 40 residual parameters but supplies deterministic zero residual values. The five-model matrix is residual CDI, zero-residual control, CCT-G3.3 full CDI, GRU, and Transformer.

```bash
%cd /content/CDI
!git pull --ff-only origin master
!PYTHONPATH=. python benchmarks/cct_g3_4_token_residual.py \
  --steps 1000 \
  --document-limit 321 \
  --chunks-per-document 32 \
  --chunk-length 16 \
  --batch-size 2 \
  --eval-batches 0 \
  --shuffle-training-batches \
  --learning-rate 0.01 \
  --relative-loss-tolerance 0.05 \
  --parameter-relative-tolerance 0.01 \
  --max-host-memory-gb 11 \
  --output-dir results/colab_cct_g3_4_token_residual
!cat results/colab_cct_g3_4_token_residual/REPORT.md
!cat results/colab_cct_g3_4_token_residual/latest.json
```

**Material-quality rule:** the candidate must beat both its exact zero-residual control and the CCT-G3.3 predecessor in every seed. To earn the material-quality target, it must also match or beat GRU in every seed and achieve mean validation loss at or below **6.664364**, which is 2% below the recorded matched-GRU mean of 6.800372. No result scales automatically; send both generated files for review.

## 8. Stage 4 — context ladder and retention test

Your original speed ambition matters most at longer sequences, not at 16-token chunks. State-space research motivates testing long dependencies and hardware-aware execution, but it does not grant CDI an automatic advantage. S4, Mamba, and Mamba-2 all pair structured state models with specific efficient algorithms and benchmark them against alternatives. [1] [2] [3]

Run a context ladder with the same model family and matched token budget:

| Context length | Purpose | Required comparison |
|---:|---|---|
| 16 | Reproduce the established short-context baseline | CDI, geometry-free CDI, GRU, Transformer |
| 64 | First meaningful chunked-language test | Same four models |
| 128 | Test whether CDI state allocation remains stable | Same four models |
| 256 and beyond | Run only after the 128-token gate passes | Same four models with memory and throughput recorded |

Build a held-out document-retention task from real Synaxarium text: place a short factual span near the beginning of a document, then evaluate whether a later continuation predicts or selects the associated span. Keep the prompts and held-out documents identical for all models. This avoids treating an invented toy task as a language-quality claim.

**Stage 4 pass rule:** CDI must retain information at least as well as the GRU and remain within the Stage 2 quality tolerance against the Transformer at the intended context. If it does not, change state capacity, state-to-readout access, or the update rule before scaling data.

## 9. Stage 5 — speed engineering, before speed claims

The current CDI `forward_chunk` loop iterates token by token in Python, so its measured CPU throughput cannot support a “faster than Transformer” claim. The speed program is separate from the quality program.

| Order | Engineering work | Evidence required |
|---:|---|---|
| 5.1 | Profile with `torch.profiler` at fixed batch size, context, and precision | Time share for embedding/output softmax, recurrent loop, state update, and geometry operation |
| 5.2 | Remove avoidable Python-loop overhead without changing model mathematics | Exact-output or tolerance test against the reference implementation |
| 5.3 | Test compiled/chunked execution and avoid dense state materialization | Correctness regression plus throughput/memory benchmark |
| 5.4 | Optimize the 16,000-way tied output path only after profiling proves it dominates | End-to-end training and generation benchmark |
| 5.5 | Consider parallel scan or an SSD-like chunked execution only with a formal CDI-equivalence test | Accuracy, gradient, and state-equivalence checks |

The rule is strict: **do not call CDI faster than a Transformer because its recurrence is theoretically linear.** Claim speed only when the benchmark holds parameter count, hardware, precision, batch size, context length, warm-up, token budget, and decoding method fixed. Mamba’s reported gains depend on a hardware-aware algorithm; Mamba-2 explicitly develops a more efficient state-space execution method. [2] [3]

## 10. Stage 6 — scale ladder

Only after Stages 2–5 pass should we increase corpus size. The scale ladder makes failures interpretable.

| Scale rung | Data scope | Purpose | Go/no-go decision |
|---:|---|---|---|
| A | 60 Synaxarium documents; 300 steps | Completed wiring and learnability proof | Passed |
| B | Full deduplicated Synaxarium corpus; 3k and 10k steps | Test durable in-domain learning | Required next |
| C | Approved in-domain corpus with approximately 1–5 million EthioBBPE tokens | Test parameter and context scaling | Only after B passes |
| D | Approved corpus with 10–20 million tokens | Test learning curves and efficiency at moderate scale | Only after C passes |
| E | Larger corpus | Pretraining research | Only after all quality, retention, and efficiency gates pass |

At each rung, save checkpoints at fixed **token** intervals, never only step intervals. The model sees different numbers of tokens per step when batch or context changes.

## 11. Stage 7 — generation comes last

Only once numerical metrics pass do we inspect fixed prompts. Use prompts that match the trained language and corpus. Since EthioBBPE and Synaxarium are an Amharic/Ge'ez-domain setup, English prompts such as “Aristotle was a Greek philosopher” are not a fair first fluency test.

For each checkpoint, use a fixed decoding configuration and produce the same held-out prompt suite. Save greedy outputs and one deterministic sampled output. Then score the output with human review only after recording loss, retrieval, throughput, and tokenizer-validity metrics.

| Generation gate | Pass condition |
|---|---|
| Token validity | No invalid IDs; checkpoint tokenizer matches current tokenizer artifact |
| Basic coherence | In-domain continuation remains linguistically plausible over a fixed length |
| No memorization claim | Outputs are checked against train-document hashes and source passages |
| Metric alignment | Better samples must coincide with improved held-out numerical metrics |

## 12. What I will do at every stage

I will keep the workflow bounded and visible. I will review the code and config before a run, give you the exact Colab command, state what the output should contain, inspect the saved report, and give one of three decisions: **continue**, **redesign**, or **stop**. I will not silently move to a larger dataset after a failure. I will also not call CDI better or faster than a Transformer without the matched evidence above.

| After you share | I will return |
|---|---|
| `REPORT.md` and `latest.json` | Quality, stability, and fairness diagnosis |
| Profiler trace or summary | Exact speed bottleneck and a prioritized optimization patch |
| A failed command output | Root cause, a documented correction, and one replacement command |
| A passing stage report | The next stage only, with its pass/fail gate |

## 13. First action for you to check

CCT-G0, CCT-G1, CCT-G2.1, CCT-G3.1, CCT-G3.2, and CCT-G3.3 are complete. CCT-G3.1 is recorded as `EARNED_GEOMETRY_EVIDENCE`, CCT-G3.2 as `EARNED_READOUT_EVIDENCE`, and CCT-G3.3 as `EARNED_HARMONIC_EVIDENCE`; however, the global quality decision remains `REDESIGN_BEFORE_SCALE` because CDI lost to GRU in every seed. Do **not** rerun Stage 2A with a larger budget or begin Stage 2B. The next permitted action is the frozen CCT-G3.4 quality-recovery command above.

If Colab has a GPU, record its name with:

```bash
!nvidia-smi || true
!python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

We will review the submitted CCT-G3.4 report and JSON before any 3,000-step, context, capacity, corpus, or optimization work.

## References

[1]: https://arxiv.org/abs/2111.00396 "Gu, Goel and Ré. Efficiently Modeling Long Sequences with Structured State Spaces (S4), 2021"
[2]: https://arxiv.org/abs/2312.00752 "Gu and Dao. Mamba: Linear-Time Sequence Modeling with Selective State Spaces, 2023"
[3]: https://arxiv.org/abs/2405.21060 "Dao and Gu. Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality, 2024"
[4]: https://github.com/nexuss0781/CDI/tree/master "CDI master branch"
[5]: https://huggingface.co/datasets/Nexuss0781/synaxarium "Nexuss0781 Synaxarium dataset card"
