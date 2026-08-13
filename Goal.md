# CCT Goal System

> **Scope:** This document defines the private technical goal system for CCT Level 1 only. It contains no product narrative, personal motivation, later-stage architecture, or external roadmap. Every stage exists to establish whether CCT can become a compact, learnable, fluent language-engine architecture.

## 1. Operating Objective

The Level 1 objective is to turn CCT from a promising recurrent/state-space experiment into a language engine with verified learning behavior, competitive quality at matched resources, reproducible execution, controlled text generation, and a justified path to scale.

A stage is complete only when its stated evidence exists. A passing run never permits skipping the next gate. A failed run is a technical result: it determines whether the next action is **repair**, **redesign**, or **stop**, rather than an excuse to add more data or parameters.

| Rule | Required behavior |
|---|---|
| Scope rule | Work only on CCT language-engine readiness. Do not add unrelated system design. |
| Evidence rule | Record command, code revision, seed, configuration, tokenizer fingerprint, data manifest, metrics, and hardware for every decision run. |
| Fairness rule | Comparisons use matched data split, causal token budget, context length, precision, optimizer family, evaluation protocol, and approximately matched parameter count. |
| Data rule | Do not scale to a larger corpus until the current controlled gate passes. Keep document-level isolation and content-hash leakage checks active. |
| Generation rule | Text samples are evidence only after held-out loss and stability gates pass. Samples do not replace quantitative evaluation. |
| Performance rule | Do not claim speed from a microbenchmark. Measure end-to-end training and generation throughput at fixed hardware, batch size, context length, precision, and vocabulary. |
| Change rule | Change one declared variable family per ablation. If architecture and training budget both change, the result is diagnostic only, not a comparison claim. |
| Privacy rule | Public or repository documents use only CCT Goal labels, technical facts, and results. |

## 2. CCT Goal Map

| CCT Goal | Primary question | Required transition evidence | If the gate fails |
|---|---|---|---|
| CCT-G0 | Can the exact code and dependency environment run deterministically? | Clean install, imports, regression suite, known code revision | Repair environment or dependency contract. |
| CCT-G1 | Can CCT learn a bounded real-language task under fair comparison? | Three-seed held-out metrics, no leakage, falling loss, matched baselines | Repair data, tokenizer, loss, optimization, or state update. |
| CCT-G2 | Does the result survive larger real-data exposure? | Full-corpus multi-seed result with shuffled training and complete held-out evaluation | Redesign capacity, readout, or optimization before further scaling. |
| CCT-G3 | Does the distinctive CCT mechanism add measurable value? | Controlled ablations with one changed mechanism at a time | Simplify or redesign the mechanism. |
| CCT-G4 | Can CCT improve quality without losing execution viability? | Context and scale ladder with stable metrics and resource logs | Fix scaling law, optimizer, or implementation bottleneck. |
| CCT-G5 | Can CCT become competitive in end-to-end efficiency? | Matched throughput and memory profile plus correctness regression | Optimize implementation before claiming architecture speed. |
| CCT-G6 | Can CCT generate coherent, controlled in-domain text? | Held-out loss gate, fixed prompts, blinded sample review, failure taxonomy | Improve model/training; do not treat samples as fluency proof. |
| CCT-G7 | Is CCT ready for a bounded instruction-readiness curriculum? | Stable language generation and supervised task-following evaluation | Continue language-engine work only. |

## 3. CCT-G0 — Reproducible Execution Readiness

### Objective

Establish one clean, repeatable environment that installs the exact CCT dependencies and runs the same code on a new machine without hidden manual steps.

### Sprint G0.1 — Clean installation

| Item | Requirement |
|---|---|
| Repository | Clone `master` only; record the commit hash. |
| Dependency backend | Install `EthioBBPE==2.0.0`; verify `import ethiobbpe`. |
| Runtime | Record Python, PyTorch, operating system, CPU/GPU availability, and precision. |
| Regression | Run the complete test suite before any training run. |
| Artifact | Save terminal summary and exact environment versions. |

**Transition gate:** The test suite passes, the EthioBBPE import succeeds, and the working tree is clean.

**Failure response:** Repair setup, package constraints, or import contracts. Do not change CCT model parameters to compensate for environment failure.

## 4. CCT-G1 — Bounded Learning Proof

### Objective

Verify that CCT learns a real, document-isolated language corpus and that the result is repeatable across seeds against matched recurrent and Transformer baselines.

### Sprint G1.1 — Exact baseline reproduction

Use the existing bounded EthioBBPE Synaxarium protocol unchanged: identical seeds, document limit, chunk length, causal token positions, tokenizer artifact, model family, and evaluation budget.

| Measurement | Required evidence |
|---|---|
| Data integrity | Document-level split and content-hash leakage checks pass. |
| Learning | Training loss decreases in every seed; no NaN/Inf. |
| Quality | Validation and test cross-entropy, perplexity, and token accuracy for every seed. |
| Fairness | Parameter count, token budget, batch size, chunk length, optimizer settings, and precision per model. |
| Performance | Training elapsed time and tokens/second, labeled by hardware. |

**Transition gate:** CCT passes the predetermined learning and baseline-tolerance gates in three seeds, and the report contains all reproducibility fingerprints.

**Failure response:** Identify one failure class—tokenizer/data contract, causal target construction, gradient path, state stability, optimizer, or readout—and repair only that class before rerunning the exact protocol.

## 5. CCT-G2 — Real-Data Scale Survival

### Objective

Test whether the bounded quality signal survives substantially broader document exposure without changing the core comparison rules.

### Sprint G2.1 — Full-corpus diagnostic

Use the deduplicated real corpus, deterministic per-epoch shuffled training batches, complete held-out validation/test evaluation, and three seeds. The existing first rung is 1,000 training steps; it is diagnostic, not a production-scale claim.

| Locked control | Requirement |
|---|---|
| Corpus | Use the governed deduplicated corpus and record its manifest fingerprint. |
| Training order | Enable deterministic per-epoch shuffle; record the seed-derived order method. |
| Evaluation | Evaluate all held-out batches, not a short prefix. |
| Baselines | Keep CCT, GRU, and Transformer at the same data and token budget. |
| Threshold | CCT must be within the declared Transformer loss tolerance and consistently match or beat the GRU. |

**Transition gate:** The quality relation does not materially collapse at the larger document and token budget, and no stability failure appears.

**Failure response:** Do not add a larger corpus. Run CCT-G3 ablations to locate a capacity, state, readout, or optimization cause.

### Sprint G2.2 — Scale ladder

Run only after G2.1 passes. Increase **one axis at a time**:

1. Training steps: 1,000 → 3,000 → 10,000.
2. Context: 16 → 64 → 128.
3. CCT capacity: one documented configuration increase at a time.
4. Corpus size: only after the smaller controlled rungs remain stable.

Each rung requires three seeds and the same report fields as G1. A rung that fails blocks the next rung.

## 6. CCT-G3 — Architecture Value

### Objective

Determine whether CCT’s distinct state/geometry mechanism contributes a measurable language-modeling benefit rather than merely adding complexity.

### Sprint G3.1 — Mechanism ablation

Create matched variants in which exactly one CCT-specific mechanism is changed or disabled while embeddings, tokenizer, training budget, optimizer, data split, and readout remain fixed.

| Variant | Question answered |
|---|---|
| Full CCT | Reference behavior of the current architecture. |
| CCT mechanism ablation A | Does the selected state/geometry element improve held-out language metrics? |
| CCT mechanism ablation B | Does the selected recurrence/readout element improve stability or retention? |
| GRU baseline | Does CCT improve on ordinary recurrence at matched resources? |
| Transformer baseline | What quality and efficiency gap remains against causal attention? |

**Transition gate:** At least one CCT-specific mechanism provides a repeated, predeclared improvement in held-out loss, stability, long-context retention, or end-to-end efficiency across seeds.

**Failure response:** Remove or simplify the non-contributing mechanism, then rerun G1/G2 rather than preserving complexity without evidence.

## 7. CCT-G4 — Language Quality and Context Readiness

### Objective

Build language-engine capacity while preserving causal correctness, state stability, and fair baseline comparison.

### Sprint G4.1 — Context ladder

| Context rung | Purpose | Required checks |
|---:|---|---|
| 16 | Preserve the established reference point | Reproduction, stability, baseline relation. |
| 64 | Verify state behavior beyond minimal chunks | Held-out loss, gradient/state norms, throughput. |
| 128 | Test extended causal context | Same metrics plus memory footprint. |
| 256+ | Run only after 128 passes | Same metrics; no quality or stability regression accepted without diagnosis. |

### Sprint G4.2 — Capacity ladder

Increase one documented CCT configuration dimension at a time, keeping output vocabulary, tokenizer artifact, context, corpus split, and training budget fixed for the comparison. Record parameter count and trainable-state size separately.

**Transition gate:** The selected CCT configuration produces monotonic or explainable quality improvement, remains numerically stable, and does not make the throughput target impossible.

**Failure response:** Return to the last stable capacity and investigate state bottlenecks, readout capacity, initialization, learning-rate schedule, or gradient control.

## 8. CCT-G5 — End-to-End Efficiency

### Objective

Determine whether CCT can approach or exceed relevant baselines in practical training and generation efficiency. Architecture speed is not assumed; it must be measured after correctness is established.

### Sprint G5.1 — Performance baseline

Profile CCT, GRU, and Transformer at matched vocabulary, parameter count, batch size, context length, precision, device, and generation length.

| Required metric | Definition |
|---|---|
| Training throughput | Completed causal token positions per second, including forward, backward, and optimizer step. |
| Generation throughput | New output tokens per second after prompt processing. |
| Peak memory | Maximum allocated or resident memory for the same workload. |
| Latency | Prompt-to-first-token and fixed-length continuation timing. |
| Correctness | Logit/loss or tolerance comparison before and after each optimization. |

### Sprint G5.2 — Optimization order

1. Profile first; locate time in embeddings/output projection, recurrent loop, state update, geometry operations, and data movement.
2. Remove avoidable Python-loop or allocation overhead without changing mathematics.
3. Add compiled or chunked execution only with a state and gradient equivalence test.
4. Optimize the vocabulary/output path only if the profile proves it dominates.
5. Re-run the matched end-to-end benchmark after every optimization.

**Transition gate:** Any claimed efficiency improvement must preserve the defined correctness tolerance and be measured end-to-end against the locked baselines.

**Failure response:** Keep the reference implementation as the correctness oracle; do not trade silently incorrect behavior for throughput.

## 9. CCT-G6 — Controlled Generation Readiness

### Objective

Show that the selected CCT configuration produces coherent, controlled **in-domain** continuations after numerical language-quality gates have passed.

### Sprint G6.1 — Fixed-prompt generation suite

Build a versioned prompt set from held-out in-domain material. For each model, generate with fixed decoding settings and record prompt, seed, generated IDs, decoded text, length, repetition measures, and termination behavior.

| Gate | Requirement |
|---|---|
| Numerical prerequisite | G2/G4 held-out quality and stability gates pass. |
| Repetition | No unbounded token loops, degenerate repetition, or invalid token decoding. |
| Continuation | Samples remain coherent relative to the local prompt and corpus domain. |
| Blind review | Compare anonymized samples without exposing model identity. |
| Error taxonomy | Classify each failure: tokenizer, decoding, short-context loss, memory/state issue, vocabulary issue, or data limitation. |

**Transition gate:** CCT samples meet the predefined in-domain coherence criteria and do not contradict held-out metrics.

**Failure response:** Diagnose decoding and model errors separately. Do not call a sample fluent because it contains a few plausible words.

## 10. CCT-G7 — Bounded Instruction-Readiness

### Objective

After language quality is stable, test whether CCT can learn a small, fully supervised instruction-to-output mapping without losing the language-engine controls established above.

### Sprint G7.1 — Supervised curriculum pilot

Use a small, curated, versioned instruction dataset with explicit train/validation/test splits. Limit the task to a narrow declared domain and evaluate exact match, schema validity, refusal/unknown behavior, and regression against the language baseline.

**Transition gate:** CCT follows the bounded task reliably, preserves baseline generation quality within the declared tolerance, and produces auditable outputs.

**Failure response:** Return to language modeling or curriculum design. Do not broaden the instruction domain until the narrow task is stable.

## 11. Required Artifact Set for Every Decision Sprint

Every decision sprint produces the following files under a unique results directory:

| Artifact | Contents |
|---|---|
| `latest.json` | Machine-readable configuration, fingerprints, seed-level metrics, summary metrics, hardware, and verdict. |
| `REPORT.md` | Human-readable method, gates, metrics, outcome, and stated scope. |
| `manifest.json` or embedded manifest | Dataset lineage, split assignments, deduplication checks, and fingerprint. |
| `environment.txt` | Python, package versions, device, precision, operating system, and code revision. |
| `commands.sh` | Exact commands used to reproduce the run. |
| `generation.json` when applicable | Fixed prompts, settings, raw IDs, decoded samples, and evaluation labels. |

## 12. Decision Vocabulary

| Verdict | Meaning | Allowed next action |
|---|---|---|
| `READY_FOR_NEXT_GOAL` | All declared gates passed within the stated scope. | Run the next CCT Goal only. |
| `REPAIR_CURRENT_GOAL` | An implementation, data, or reproducibility defect invalidated the result. | Repair the named defect; rerun the same goal. |
| `REDESIGN_BEFORE_SCALE` | The current CCT mechanism/configuration failed a fair quality or stability gate. | Run controlled ablation or redesign; do not add data. |
| `OPTIMIZE_IMPLEMENTATION` | Quality is acceptable but end-to-end throughput/memory is not. | Profile and optimize with correctness checks. |
| `STOP_UNTIL_NEW_EVIDENCE` | The evidence does not justify the next stage. | Preserve artifacts and define a new hypothesis first. |

## 13. Current Position

| Field | Current state |
|---|---|
| Active goal | `CCT-G2` — real-data scale survival. |
| Completed evidence | Bounded three-seed real-data learning proof, matched baseline comparison, reproducible CPU run, governed split checks, and EthioBBPE tokenizer contract. |
| Next approved sprint | `CCT-G2.1` — full-corpus 1,000-step diagnostic with deterministic shuffled training and complete held-out evaluation. |
| Required output before any next stage | Final verdict line, `REPORT.md`, and `latest.json` from the exact `CCT-G2.1` command. |
| Explicitly not approved yet | Production-scale corpus training, language-fluency claims, speed claims, broad instruction training, or any out-of-scope work. |

## 14. Stage Discipline

CCT advances through evidence, not urgency. The current goal is complete only when the gate says it is complete. If a result is unclear, the correct transition is to a smaller diagnostic sprint, not to a larger uncontrolled run.
