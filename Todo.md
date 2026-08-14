# CCT Todo

> **Private scope:** This is the sole execution checklist for CCT Level 1. It contains only CCT technical objectives, evidence, gates, and decisions. Do not add product narrative, personal background, later-stage design, or unrelated system work.

## How to Use This Checklist

- [ ] Work on **one unchecked CCT Goal or sprint at a time**.
- [ ] Do not begin a later goal until the current goal has its required evidence and an allowed transition verdict.
- [ ] Record the exact command, code revision, seed list, configuration, tokenizer fingerprint, data manifest fingerprint, metrics, hardware, and result directory for every decision run.
- [ ] Keep comparisons fair: same data split, causal token budget, context length, precision, optimizer family, evaluation protocol, and approximately matched parameter count.
- [ ] Keep document-level isolation and content-hash leakage checks enabled for all real-data runs.
- [ ] Treat text samples as supporting evidence only after numerical loss and stability gates pass.
- [ ] Do not claim speed from a microbenchmark; measure end-to-end training and generation at fixed hardware, vocabulary, batch size, context, and precision.
- [ ] Change only one declared variable family per ablation. Mark a multi-variable experiment as **diagnostic only**.
- [ ] Use CCT Goal labels, technical facts, commands, and results only in repository artifacts.

## Required Record for Every Decision Sprint

Before marking any sprint complete, confirm all applicable artifacts exist in one unique results directory.

- [ ] `latest.json` records configuration, code revision, fingerprints, per-seed metrics, summary metrics, hardware, and verdict.
- [ ] `REPORT.md` records method, locked controls, gates, metrics, result, and scope.
- [ ] `manifest.json`, or an embedded manifest, records dataset lineage, splits, deduplication, leakage checks, and fingerprint.
- [ ] `environment.txt` records Python, package versions, device, precision, operating system, and code revision.
- [ ] `commands.sh` records the exact commands used.
- [ ] `generation.json` exists for generation sprints and records prompts, settings, raw IDs, decoded output, and labels.
- [ ] The working tree is clean or every intentional change is committed before the result is treated as reproducible.

## Verdict Rules

| Verdict | Checklist action |
|---|---|
| `READY_FOR_NEXT_GOAL` | Mark the current transition gate complete and unlock only the next listed goal. |
| `REPAIR_CURRENT_GOAL` | Keep the transition gate unchecked; repair the named data, setup, or implementation issue and rerun the same sprint. |
| `REDESIGN_BEFORE_SCALE` | Keep all scale tasks unchecked; create one controlled redesign/ablation task. |
| `OPTIMIZE_IMPLEMENTATION` | Do not make speed claims; profile and optimize while retaining a correctness oracle. |
| `STOP_UNTIL_NEW_EVIDENCE` | Stop execution, preserve artifacts, and define one testable hypothesis before resuming. |

---

# CCT-G0 — Reproducible Execution Readiness

**Goal:** Establish one repeatable environment that installs the exact CCT dependency contract and runs the same code without hidden manual steps.

## G0.1 — Repository and Dependency Readiness

- [x] Pin the tokenizer backend in `requirements.txt` as `EthioBBPE==2.0.0`.
- [x] Remove the unused `transformers` runtime requirement that conflicts with EthioBBPE’s Tokenizers constraint.
- [x] Provide a clean master-branch Colab bootstrap in `colab.md`.
- [x] Verify the pinned EthioBBPE backend imports in the validated environment.
- [x] Run the CDI regression suite after the dependency repair; baseline evidence is **246 passing tests**.
- [ ] In every new runtime, delete stale local checkouts before cloning.
- [ ] In every new runtime, clone `master` only and record `git rev-parse HEAD`.
- [x] Install dependencies with `python -m pip install -r requirements.txt` before importing CCT; validated at `a038147`.
- [x] Confirm `import ethiobbpe` and record the module path; validated at `a038147`.
- [x] Record Python version, PyTorch version, operating system, CPU/GPU availability, precision, and package versions; see `docs/CCT_G0_READINESS.md`.
- [x] Run the complete regression suite before starting any new training sprint; validated at `a038147` with `246 passed`.
- [x] Save terminal output and environment versions under the sprint result directory; see `results/cct_g0/a038147/` from the validation run.

### G0 Transition Gate

- [x] Regression suite passes in the target runtime; `246 passed` at `a038147`.
- [x] EthioBBPE import succeeds in the target runtime; `EthioBBPE==2.0.0` at `a038147`.
- [x] The target checkout is `master` at a recorded commit; `a03814705a73a3cd36658e3d0780a982593070f9`.
- [x] The working tree is clean after the dependency and regression validation.

**If the gate fails:** repair the checkout, dependency constraint, import, or environment contract. Do **not** change CCT model parameters.

**CCT-G0 status:** `READY_FOR_NEXT_GOAL`. The committed evidence record is `docs/CCT_G0_READINESS.md` and the reusable verifier is `scripts/run_cct_g0.sh`.

---

# CCT-G1 — Bounded Learning Proof

**Goal:** Verify that CCT learns a real, document-isolated language task under a fair three-seed comparison against matched GRU and Transformer baselines.

## G1.1 — Exact Baseline Reproduction

### Locked protocol

- [x] Use the EthioBBPE tokenizer artifact and record its fingerprint.
- [x] Use document-isolated Synaxarium text with governed splits and content-hash duplicate checks.
- [x] Use seeds `[11, 29, 47]`.
- [x] Use matched CCT, GRU, and Transformer baselines at the bounded 300-step protocol.
- [x] Record held-out validation/test metrics, parameter counts, token budget, and throughput.
- [x] Independently reproduce the bounded CPU run with `EARNED_NEXT_PILOT`.

### Reproduction checklist

- [ ] Clone the recorded code revision and install the exact requirements.
- [ ] Run the unchanged bounded protocol; do not change seeds, tokenizer, document limit, context, batch size, optimizer settings, models, or evaluation budget.
- [ ] Confirm document-level split isolation passes.
- [ ] Confirm content-hash leakage checks pass.
- [ ] Confirm CCT training loss decreases in every seed.
- [ ] Confirm no NaN, Inf, missing-gradient, or invalid-token failure appears.
- [ ] Record validation cross-entropy, test cross-entropy, perplexity, and token accuracy per seed for every model.
- [ ] Record parameter count, causal token positions, batch size, chunk length, optimizer settings, precision, elapsed time, and tokens/second per model.
- [ ] Confirm the report contains data-manifest and tokenizer fingerprints.

### G1 Transition Gate

- [x] CCT passed the predefined bounded learning gate in three seeds.
- [x] CCT passed the bounded matched-baseline tolerance gate.
- [x] The bounded result was independently reproduced on CPU.
- [x] Preserve the final result directory and commit reference in the CCT evidence index; see `docs/CCT_EVIDENCE_INDEX.md`.

**If a future G1 reproduction fails:** identify exactly one failure class—tokenizer/data contract, target construction, gradient path, state stability, optimizer, or readout—then repair that class and rerun the unchanged protocol.

---

# CCT-G2 — Real-Data Scale Survival

**Goal:** Determine whether CCT’s bounded quality signal survives broader real-data exposure without changing the core comparison contract.

## G2.1 — Full-Corpus 1,000-Step Diagnostic

### Preparation checklist

- [x] Deduplicate the real corpus and record the governed manifest process.
- [x] Add deterministic per-epoch shuffled training batches to the pilot harness.
- [x] Add complete held-out evaluation mode (`--eval-batches 0`).
- [x] Add a regression test proving shuffled training resumes identically from a checkpoint.
- [x] Run the full CDI test suite after these controls; baseline evidence is **246 passing tests**.
- [x] Push the hardened Stage 2 harness to `master`.
- [x] Start from a clean master checkout with the current requirements installed; submitted revision `d5a2180`.
- [x] Record the code revision before the run; `d5a2180e6e61494140b8ff221703cef7c317ecd3`.
- [x] Use the deduplicated governed corpus and record its manifest fingerprint; `2b868a661d628ec0e4507f65ee99e79abfbed12910241f95e7660a99e97e39c8`.
- [x] Use deterministic per-epoch batch shuffle and record the seed-derived method; `deterministic_per_epoch_shuffle`.
- [x] Use all held-out validation and test batches (`--eval-batches 0`); `all_held_out_batches`.
- [x] Use seeds `[11, 29, 47]`.
- [x] Use 1,000 training steps.
- [x] Use the declared full-corpus diagnostic configuration: `--document-limit 321`, `--chunks-per-document 32`, `--chunk-length 16`, `--batch-size 2`, `--learning-rate 0.01`.
- [x] Run CCT, GRU, and Transformer under the same data, token budget, context, precision, and evaluation protocol.
- [x] Save the final `Pilot ...` verdict line; `EARNED_NEXT_PILOT` from the harness.
- [x] Save `REPORT.md`, `latest.json`, manifest, environment record, and commands; decision record: `docs/CCT_G2_1_DECISION.md`.

### G2.1 Measurement checklist

- [x] Confirm no duplicate content or cross-split leakage.
- [x] Confirm all three CCT seeds have finite training and evaluation values; no non-finite values in the submitted result JSON.
- [x] Confirm CCT training loss decreases in every seed.
- [x] Record validation/test cross-entropy, perplexity, token accuracy, and uncertainty across seeds; see `docs/CCT_G2_1_DECISION.md` and the submitted result artifacts.
- [x] Record parameter count and total causal token positions per model/seed; 80,366/80,120/80,172 parameters and 30,000 positions per model/seed.
- [x] Record training elapsed time and tokens/second for every model/seed.
- [x] Confirm the report declares `deterministic_per_epoch_shuffle`.
- [x] Confirm the report declares `all_held_out_batches`.
- [x] Compute the CCT relative validation-loss gap to the best baseline; +1.44% versus GRU.
- [x] Check whether CCT is within the declared Transformer loss tolerance; +0.58%, within 5%.
- [ ] Check whether CCT consistently matches or beats the GRU; **failed in all three seeds**.

### G2.1 Transition Gate

- [ ] The quality relation does not materially collapse relative to G1; **failed** because CDI lost to GRU in all three seeds after having a slight mean G1 advantage.
- [x] No stability failure appears.
- [x] CCT remains within the declared Transformer tolerance; +0.58% mean validation-loss gap versus Transformer.
- [ ] CCT consistently matches or beats the GRU; **failed** with a +1.44% mean gap and losses above GRU in all three seeds.
- [x] The result has a complete, reproducible artifact set; see `docs/CCT_G2_1_DECISION.md`.

**CCT-G2.1 status:** `REDESIGN_BEFORE_SCALE`. The harness-level `EARNED_NEXT_PILOT` did not satisfy the stricter CCT transition gate.

**If the gate passes:** unlock G2.2 only.

**If the gate fails:** do **not** add more data, steps, or context. Mark `REDESIGN_BEFORE_SCALE` and move to one controlled CCT-G3 ablation.

## G2.2 — Scale Ladder

Run one rung at a time. Every rung requires three seeds, the same evidence fields as G1, and a transition decision before the next rung.

### Training-step ladder

- [ ] Run 3,000 steps only after G2.1 passes.
- [ ] Review three-seed metrics, stability, and throughput before moving onward.
- [ ] Run 10,000 steps only after the 3,000-step gate passes.
- [ ] Review three-seed metrics, stability, and throughput before changing any other axis.

### Context ladder entry

- [ ] Begin context changes only after the selected training-step rung is stable.
- [ ] Keep corpus split, tokenizer, model family, parameter budget, and optimizer fixed when changing context.

### Capacity ladder entry

- [ ] Increase exactly one documented CCT configuration dimension at a time.
- [ ] Record total parameter count and trainable-state size separately.
- [ ] Keep vocabulary, tokenizer artifact, corpus split, context, and training budget fixed for each capacity comparison.

### Corpus ladder entry

- [ ] Expand corpus size only after smaller controlled rungs are stable.
- [ ] Rebuild and record the governed manifest before every corpus expansion.
- [ ] Re-run leakage and duplicate checks after every corpus change.

---

# CCT-G3 — Architecture Value

**Goal:** Determine whether each distinctive CCT mechanism contributes measurable value rather than unverified complexity.

## G3.1 — Controlled Mechanism Ablations

### Common controls

- [x] Complete a source-level architecture review and issue inventory; see `Architecture.md` and `ISSUES_TODO.md`.
- [x] Identify the first gate-blocking diagnostic: the current mean-over-vertices language readout makes Laplacian geometry unobservable to causal token loss.
- [x] Select one CCT mechanism for the first ablation hypothesis: state-to-readout geometry observability through fixed zero-sum vertex contrasts.
- [x] Write the pre-run hypothesis and expected metric before training; see `docs/CCT_G3_1_PREREGISTRATION.md`.
- [x] Keep embeddings, tokenizer, output vocabulary, training budget, optimizer, data split, context, precision, and seed list fixed in the CCT-G3.1 harness; only the pre-registered readout access and exact geometry-disabled counterpart differ.
- [x] Change exactly one mechanism in the selected variant: expose vertex contrasts to the existing output-width readout without changing recurrence or tokenizer.
- [x] Record parameter-count differences: full CDI 80,510; geometry-free CDI 80,510; GRU 80,120; Transformer 80,172; 0.49% maximum relative spread.
- [x] Run Full CCT as the reference model.
- [x] Run the selected CCT geometry-free ablation variant.
- [x] Run GRU baseline.
- [x] Run Transformer baseline.
- [x] Run every variant across three seeds.

### Ablation A — State/geometry contribution

> CCT-G3.1 resolved the mean-readout blind spot by adding fixed zero-sum vertex contrasts to the full and geometry-disabled readouts. The empirical comparison remains required; implementation-level signal only is not a scale decision.

- [x] Define the exact state/geometry element changed or disabled; see `docs/CCT_G3_1_PREREGISTRATION.md`.
- [x] Verify the variant remains causal and numerically stable through the local geometry-observability and Stage C regression gates.
- [x] Compare held-out loss, test loss, token accuracy, throughput, and memory; see `docs/CCT_G3_1_DECISION.md`.
- [x] Record whether the geometry element improves a predeclared metric repeatedly across seeds; full CDI wins against geometry-free CDI in seeds 11, 29, and 47.
- [ ] Record gradient/state norms as dedicated empirical time-series evidence in the next controlled ablation.

### Ablation B — Recurrence/readout contribution

- [x] Define the exact readout element: concatenate deterministic zero-sum vertex contrasts with the existing per-band mean; recurrence unchanged.
- [x] Verify output shape, causal target alignment, and geometry-gradient reachability through `tests/test_cct_geometry_observability.py`.
- [x] Compare held-out loss, test loss, token accuracy, throughput, and host memory against the capacity-matched mean-readout control; see `docs/CCT_G3_2_DECISION.md`.
- [x] Record repeated readout value: full CDI beats the mean-readout control in seeds 11, 29, and 47, with a 0.057296 mean validation-loss improvement.
- [ ] Record retention/context behavior and dedicated state/gradient time-series under a later separately pre-registered protocol.

### G3 Transition Gate

- [x] At least one CCT-specific mechanism shows repeated predeclared value in held-out loss: sparse geometry improved held-out validation loss against the exact geometry-free counterpart in all three seeds.
- [x] The geometry value survives three-seed comparison and does not depend on hidden budget changes; parameter spread was 0.49% and the protocol was frozen.

**CCT-G3 status:** `EARNED_GEOMETRY_EVIDENCE`, `EARNED_READOUT_EVIDENCE`, `EARNED_HARMONIC_EVIDENCE`, and `EARNED_TOKEN_RESIDUAL_EVIDENCE`; see `docs/CCT_G3_1_DECISION.md`, `docs/CCT_G3_2_DECISION.md`, `docs/CCT_G3_3_DECISION.md`, and `docs/CCT_G3_4_DECISION.md`. The selected residual CDI now beats GRU in all three seeds but remains `QUALITY_RECOVERY_PARTIAL`, not scale-authorized, because it did not reach the pre-registered 2% material-quality margin.

**If the gate passes:** retain each contributing mechanism. Any architecture-selection or quality-recovery proposal must be separately pre-registered and must preserve the frozen evidence contract.

**If the gate fails:** remove or simplify the non-contributing mechanism, then rerun the affected G1/G2 protocol. Do not retain complexity without evidence.

## G3.2 — Controlled Readout-Contribution Ablation

- [x] Pre-register an exact parameter-aware contrast-readout control without changing corpus, tokenizer, steps, context, optimizer, seeds, precision, or the 11 GiB memory ceiling; see `docs/CCT_G3_2_PREREGISTRATION.md`.
- [x] Verify its causal output shape, gradient contract, and full-versus-control semantics locally before training; the full suite had 276 passing tests before the empirical run.
- [x] Implement and run the dedicated five-model CCT-G3.2 harness across three seeds.
- [x] Record held-out loss, test loss, token accuracy, throughput, host memory, and parameter counts; see `docs/CCT_G3_2_DECISION.md`.
- [x] Retain the fixed contrast readout and sparse geometry because both supplied repeated held-out value; the global quality decision remains `REDESIGN_BEFORE_SCALE` because full CDI lost to GRU in every seed.

## G3.3 — Controlled Harmonic-Memory-Band Contribution Ablation

- [x] Pre-register an exact harmonic-disabled parameter-aware control without changing the selected full CDI architecture, corpus, tokenizer, steps, context, optimizer, seeds, precision, all-held-out evaluation, or 11 GiB memory ceiling; see `docs/CCT_G3_3_PREREGISTRATION.md`.
- [x] Verify its causal shape, harmonic inactive-gradient contract, state stability, parameter equality, and five-model harness locally before training; 15 focused CCT-G3 control/harness tests and the full 282-test regression suite passed, and a non-evidentiary `/tmp` smoke run passed the 11 GiB guard.
- [x] Run full CDI, harmonic-disabled CDI, geometry-free CDI, GRU, and Transformer across the frozen three-seed, 1,000-step contract; the submitted formal artifact contains all 15 finite records.
- [x] Record held-out loss, test loss, token accuracy, state/gradient diagnostics, throughput, host memory, and parameter counts from the submitted formal run; see `docs/CCT_G3_3_DECISION.md`.
- [x] Decide the harmonic 16–64 time-constant band adds repeated held-out value: full CDI beat harmonic-disabled CDI in seeds 11, 29, and 47, with a 0.031414 mean validation-loss improvement. Retain the band; global status remains `REDESIGN_BEFORE_SCALE` because CDI remained above GRU in every seed.

## G3.4 — Selective Token-Residual Quality Recovery

- [x] Pre-register one bounded selective token-residual readout mechanism and an exact parameter-aware zero-residual control; see `docs/CCT_G3_4_PREREGISTRATION.md`.
- [x] Verify causal source-token dependence, retained recurrent state, narrow inactive-gradient contract, stability, parameter equality, and five-model harness locally before training; 21 focused CCT-G3 control/harness tests and the full 288-test regression suite passed, and a non-evidentiary `/tmp` smoke run passed the 11 GiB guard.
- [x] Run residual CDI, exact residual control, CCT-G3.3 full CDI, GRU, and Transformer across the frozen three-seed, 1,000-step contract; the submitted formal artifact contains all 15 finite records.
- [x] Record held-out loss, test loss, token accuracy, throughput, host memory, and parameter counts; see `docs/CCT_G3_4_DECISION.md`.
- [x] Apply the material-advantage gate: candidate beat GRU in all three seeds and earned `QUALITY_RECOVERY_PARTIAL`, but its 6.743546 mean validation loss was above the required 6.664364 2% target. Retain the token residual; no CCT-G2.2 proposal is authorized.

## G3.5 — State-Conditioned Token-Residual Fusion

- [x] Pre-register one bounded state-conditioned fusion gate over the retained CCT-G3.4 token residual and exact parameter-aware fusion-one control; see `docs/CCT_G3_5_PREREGISTRATION.md`.
- [x] Verify causal source-token/state dependence, retained DCSS state trajectory and residual values, narrow inactive-gradient contract, stability, parameter equality, and five-model harness locally before training; 27 focused CCT-G3 control/harness tests and the full 294-test regression suite passed, and a non-evidentiary `/tmp` smoke run passed the 11 GiB guard.
- [ ] Run fused residual CDI, exact fusion control, CCT-G3.4 residual CDI, GRU, and Transformer across the frozen three-seed, 1,000-step contract.
- [ ] Record held-out loss, test loss, token accuracy, throughput, host memory, and parameter counts.
- [ ] Apply the 2% material-quality gate: fused candidate must beat GRU in every seed and reach mean validation loss at or below 6.664364 before any CCT-G2.2 proposal can be reviewed.

---

# CCT-G4 — Language Quality and Context Readiness

**Goal:** Increase language-engine capacity and usable context while preserving causal correctness, numerical stability, fair comparison, and execution viability.

## G4.1 — Context Ladder

### Context 16

- [x] Preserve the established 16-token reference protocol.
- [ ] Reconfirm the selected post-G3 configuration against the reference before increasing context.

### Context 64

- [ ] Keep corpus split, tokenizer, vocabulary, model family, parameter target, training budget, optimizer, precision, and seeds fixed.
- [ ] Train and evaluate CCT, GRU, and Transformer at context 64.
- [ ] Record held-out loss, test loss, token accuracy, gradient norms, CCT state norms, throughput, and peak memory.
- [ ] Confirm no CCT stability regression relative to the accepted context-16 configuration.

### Context 128

- [ ] Unlock only after the context-64 gate passes.
- [ ] Repeat the locked matched comparison at context 128.
- [ ] Record all context-64 metrics plus memory footprint and any long-context failure mode.
- [ ] Confirm the quality relation remains within the declared tolerance.

### Context 256 and beyond

- [ ] Unlock only after context 128 passes.
- [ ] Run the same matched protocol; do not change unrelated variables.
- [ ] Record peak memory, throughput, stability, and held-out quality at every rung.
- [ ] Stop the ladder if quality, stability, or resource use regresses without an explainable hypothesis.

## G4.2 — Capacity Ladder

- [ ] Select one CCT configuration dimension to increase.
- [ ] Write the expected quality, stability, and throughput effect before running.
- [ ] Increase that single dimension only.
- [ ] Record parameter count and trainable-state size.
- [ ] Run three seeds with fixed data, context, tokenizer, vocabulary, optimizer, and training budget.
- [ ] Compare against the immediate preceding stable configuration.
- [ ] Keep the new capacity only if quality improves monotonically or the trade-off is explicitly justified.

### G4 Transition Gate

- [ ] The selected CCT configuration improves or preserves held-out quality.
- [ ] Training and inference remain numerically stable.
- [ ] The configuration does not make the CCT-G5 throughput target impossible.
- [ ] The result is reproducible with the required artifact set.

**If the gate fails:** return to the last stable configuration and diagnose state bottleneck, readout capacity, initialization, learning-rate schedule, or gradient control.

---

# CCT-G5 — End-to-End Efficiency

**Goal:** Measure and improve practical CCT training and generation efficiency without sacrificing correctness.

## G5.1 — Matched Performance Baseline

- [ ] Select the accepted quality configuration from CCT-G4.
- [ ] Lock vocabulary, tokenizer, parameter count, batch size, context, precision, device, prompt length, and generation length.
- [ ] Benchmark CCT, GRU, and Transformer under those same conditions.
- [ ] Measure completed causal training token positions per second, including forward, backward, and optimizer step.
- [ ] Measure new generated output tokens per second after prompt processing.
- [ ] Measure prompt-to-first-token latency and fixed-length continuation latency.
- [ ] Measure peak allocated or resident memory.
- [ ] Record device, driver/runtime, precision, batch size, context, and vocabulary with each benchmark.
- [ ] Save raw benchmark samples, not only averages.

## G5.2 — Profile Before Change

- [ ] Profile the reference CCT implementation before optimizing.
- [ ] Measure time in embeddings/output projection.
- [ ] Measure time in recurrent loop execution.
- [ ] Measure time in state update operations.
- [ ] Measure time in geometry/state-specific operations.
- [ ] Measure data movement and allocation overhead.
- [ ] Select one dominant bottleneck and write one optimization hypothesis.

## G5.3 — Correctness-Preserving Optimization

- [ ] Keep the unoptimized implementation as the correctness oracle.
- [ ] Remove avoidable Python-loop or allocation overhead only after profiling identifies it.
- [ ] Add compiled or chunked execution only with state and gradient equivalence tests.
- [ ] Optimize vocabulary/output handling only if the profile proves it dominates.
- [ ] Compare logits/loss within a declared tolerance before and after every optimization.
- [ ] Re-run unit, gradient, checkpoint/resume, and generation regressions after every optimization.
- [ ] Re-run the matched end-to-end benchmark after every optimization.

### G5 Transition Gate

- [ ] Any performance improvement preserves the declared correctness tolerance.
- [ ] Any speed statement is based on matched end-to-end evidence.
- [ ] Any memory statement is based on the same workload and hardware.

**If the gate fails:** retain the reference implementation, mark `OPTIMIZE_IMPLEMENTATION`, and return to profiling. Do not trade correctness for throughput.

---

# CCT-G6 — Controlled Generation Readiness

**Goal:** Demonstrate coherent, controlled **in-domain** continuation only after numerical language-quality and stability gates pass.

## G6.1 — Versioned Fixed-Prompt Suite

### Prompt-set preparation

- [ ] Build prompts from held-out in-domain material only.
- [ ] Version the prompt set and record its source split and fingerprint.
- [ ] Define fixed prompt lengths, generation lengths, decoding mode, temperature, top-k/top-p if used, and seed behavior.
- [ ] Keep the prompt set identical for CCT and baselines.

### Generation run

- [ ] Confirm G2/G4 held-out quality and stability gates passed before generating samples.
- [ ] Generate from every fixed prompt with every evaluated model.
- [ ] Record prompt text/IDs, generation configuration, seed, raw generated IDs, decoded text, length, repetition measures, and termination behavior.
- [ ] Detect invalid token decoding.
- [ ] Detect unbounded loops and degenerate repetition.
- [ ] Evaluate continuation coherence relative to the local prompt and corpus domain.
- [ ] Create anonymized samples for blind review without model labels.
- [ ] Record blinded review criteria and labels.

### Failure taxonomy

- [ ] Classify each failure as tokenizer, decoding, short-context loss, state/memory issue, vocabulary issue, data limitation, or another explicitly named category.
- [ ] Separate decoding failures from model-quality failures.
- [ ] Connect each failure category to one next repair hypothesis.

### G6 Transition Gate

- [ ] CCT produces no unbounded repetition or invalid-decoding failure in the fixed suite.
- [ ] CCT samples meet the predefined in-domain coherence criteria.
- [ ] Sample quality does not contradict held-out numerical metrics.
- [ ] The full generation artifact set is versioned and reproducible.

**If the gate fails:** diagnose decoding and model failures separately. Do not mark CCT fluent because isolated samples look plausible.

---

# CCT-G7 — Bounded Instruction Readiness

**Goal:** After stable language performance, test a small fully supervised instruction-to-output mapping while preserving language-engine controls.

## G7.1 — Narrow Supervised Curriculum

- [ ] Unlock only after CCT-G6 passes.
- [ ] Define one narrow instruction domain.
- [ ] Build a curated, versioned dataset with explicit train/validation/test splits.
- [ ] Record source, licensing, deduplication, and split fingerprint.
- [ ] Define task schema and expected outputs before training.
- [ ] Define exact-match, schema-validity, refusal/unknown behavior, and language-regression metrics.
- [ ] Fine-tune or train only under the recorded controlled configuration.
- [ ] Evaluate on the held-out instruction split.
- [ ] Re-run the CCT language-generation regression suite.
- [ ] Compare language quality to the accepted pre-instruction baseline.
- [ ] Save all task outputs and evaluation labels for audit.

### G7 Transition Gate

- [ ] CCT follows the narrow supervised task reliably.
- [ ] Outputs are schema-valid and auditable.
- [ ] Unknown or unsupported requests receive the defined behavior.
- [ ] Language quality remains within the declared regression tolerance.

**If the gate fails:** return to language modeling or curriculum design. Do not broaden the instruction domain.

---

# Current Execution Status

| Field | Status |
|---|---|
| Active CCT Goal | **CCT-G3 — Architecture Value** |
| Active sprint | **CCT-G3.3 — Controlled harmonic-memory-band contribution ablation** |
| Completed foundation | CCT-G0 readiness validation at `a038147`, bounded three-seed learning proof, CCT-G2.1 full-corpus diagnostic at `d5a2180`, CCT-G3.1 geometry evidence at `646c272`, and CCT-G3.2 readout evidence at `fb50b57` under 11 GiB guarded execution. |
| Required next evidence | A pre-registered, parameter-aware CCT-G3.3 harmonic-band control report under the frozen CCT-G3 contract. |
| Not yet approved | CCT-G2.2 (3,000 steps), larger corpus training, context/capacity changes, speed claims, fluency claims, broad instruction training, or any work outside this Todo. |

## Immediate Next Checklist

- [x] Record the CCT-G3.1 empirical result as `EARNED_GEOMETRY_EVIDENCE`; see `docs/CCT_G3_1_DECISION.md`.
- [x] Preserve the submitted CCT-G3.1 report/JSON identifiers, 11 GiB memory evidence, and strict base `REDESIGN_BEFORE_SCALE` quality decision.
- [x] Retain sparse geometry because it supplied repeated held-out value against the exact geometry-free CDI control.
- [x] Record CCT-G3.2 as `EARNED_READOUT_EVIDENCE` and re-confirm CCT-G3.1 geometry evidence; see `docs/CCT_G3_2_DECISION.md`.
- [x] Preserve the submitted five-model artifact identifiers, three-seed gates, and 11 GiB host-memory evidence.
- [x] Retain the contrast readout and sparse geometry; neither result unlocks scale because CDI remains above GRU in every seed.
- [ ] Pre-register the exact CCT-G3.3 harmonic-band control and its parameter-aware fairness rule.
- [ ] Verify CCT-G3.3 local causal, gradient, numerical, harness, and parameter-fairness gates.
- [ ] Run and review CCT-G3.3 before any quality rerun; CCT-G2.2, context, capacity, corpus, and performance work remain blocked.

## Stage Discipline

- [ ] Do not skip a transition gate.
- [ ] Do not replace a failed gate with a larger uncontrolled run.
- [ ] Do not make a quality or speed claim without its required evidence.
- [ ] Do not begin a later CCT Goal until the active goal has an allowed verdict.
- [ ] Keep this file as the sole authoritative execution checklist; update checkboxes and commit the evidence-linked change after every completed sprint.
