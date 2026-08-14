# CCT/CDI Engineering Issues and Remediation Backlog

> **Review basis:** fresh inspection of the reviewed master revision, two focused non-mutating geometry probes, the submitted CCT-G2.1 artifacts, the cumulative remediation regression run, the subsequent CCT-G3.1 host-memory guard regression, and the exact-ablation gradient-contract regression run of **271 passing tests**. This file records engineering facts and required work. It does not authorize a training-scale increase or change the CCT decision recorded in `Todo.md`.

## Severity and Execution Rules

| Severity | Meaning | Required handling |
|---|---|---|
| **P0 — gate-blocking** | Can invalidate an architecture conclusion, bypass CCT governance, leak held-out data, or execute an unsafe/unapproved route. | Contain before any affected route is run. |
| **P1 — high** | Can produce a misleading metric, false test confidence, irreproducible restore, or user execution of the wrong code path. | Repair before the next CCT decision that relies on it. |
| **P2 — important** | Limits stability evidence, efficiency, maintainability, or interpretability but does not by itself invalidate CCT-G2.1. | Schedule only after the P0/P1 work required by the active gate. |
| **P3 — hygiene** | Naming, metadata, or documentation debt with bounded operational impact. | Repair with the related module; do not let it displace gate work. |

The following restrictions are active until a later gate explicitly changes them.

| Locked restriction | Reason |
|---|---|
| Do not run CCT-G2.2 or increase steps, context, capacity, or corpus size. | CCT-G2.1 is `REDESIGN_BEFORE_SCALE`. |
| Do not use legacy or production commands for CCT work. | `./run.sh` is now a safe readiness/status interface; legacy v2 and external production routes are fail-closed. |
| Do not use `train.py` or the top-level `cdi` API as evidence for CCT. | They invoke the legacy v2 stack, not the active CCT v3 language engine. |
| Do not interpret Stage C geometry tests as language-model evidence. | They test recurrence-local output activity, not geometry influence on token loss. |

## Priority Summary

| ID | Severity | Area | Issue in one sentence | Required state |
|---|---|---|---|---|
| CCT-ARCH-001 | P0 | Core architecture | Mean-over-vertices readout makes the Laplacian geometry invisible to language logits and causal loss. | Diagnose and redesign under CCT-G3.1. |
| CCT-EMP-002 | P0 | Empirical gate | The harness verdict omits the stricter CCT requirement that CDI match or beat GRU. | Unify executable and Todo gate logic before next evidence run. |
| CCT-RUN-003 | P0 | Entry point | `run.sh` launches unapproved external production training after tests. | Quarantine or replace before another user follows it. |
| CCT-DATA-004 | P0 | Production route | `train_production.py` trains on all ingested text, including manifest validation/test rows. | Disable or rebuild with split-respecting data selection. |
| CCT-DATA-005 | P0 | External ingestion | Production ingestion executes remote dataset code and asserts governance metadata without source verification. | Disable by default; replace with reviewed, pinned, governed ingestion. |
| CCT-TEST-006 | P1 | Test oracle | Existing geometry tests pass when the geometry gradient is exactly zero through the actual LM loss. | Add LM-level signal and ablation tests before CCT-G3.1. |
| CCT-CKPT-007 | P1 | Checkpoint restore | Generic Stage D restore validates only tokenizer identity, not model/config/topology/data compatibility. | Bind and verify all relevant fingerprints before restore. |
| CCT-ART-008 | P1 | Reproducibility | Several benchmark runners overwrite tracked `Stages/` reports. | Route all generated reports to ignored result directories. |
| CCT-DOC-009 | P1 | User guidance | README, top-level exports, Colab legacy sections, and `run_tests.py` describe incompatible or incomplete paths. | Clearly designate CCT v3 as active and legacy paths as non-CCT. |
| CCT-TRAIN-010 | P1 | Context contract | Chunk-local training resets recurrent state and cannot establish long-context retention. | State this boundary and add a later state-carrying document protocol. |
| CCT-LEG-011 | P1 | Legacy stack | Dense v2 training/data code is externally coupled, domain-incoherent, padding-unsafe, and not feasible at advertised dense scales. | Quarantine from active entry points; retain as reference only. |
| CCT-MATH-012 | P2 | Stability | Several stated bounds are diagnostics only; the explicit geometry step has no runtime spectral safeguard. | Add enforceable stability checks after the geometry signal is repaired. |
| CCT-EVAL-013 | P2 | Metrics | Generic evaluation averages batch losses rather than causal-token-weighting. | Replace or forbid it in all evidence paths. |
| CCT-TRAIN-014 | P2 | Training contract | Warmup and gradient accumulation are declared but unused; shuffled batch order is rebuilt every step. | Either implement or remove these fields; cache epoch permutations. |
| CCT-INF-015 | P2 | Inference | Inference reconstructs dynamics from tensor shapes/defaults and strips prompt text by string prefix. | Serialize full dynamic config and slice continuations by token prefix. |
| CCT-PERF-016 | P2 | Runtime | The active recurrent loop is Python-token-serial and repeatedly materializes topology tensors. | Profile only after G3.1; then cache indices and preserve reference equivalence. |
| CCT-CFG-017 | P2 | Configuration | Production configuration retains obsolete tokenizer and mixed P1/production semantics. | Replace with a single EthioBBPE/CCT-compatible schema. |
| CCT-API-018 | P2 | Public surface | `cdi` exports legacy v2 while `cdi.v3` has stale module claims and broad experimental exports. | Make active/legacy namespace boundaries explicit. |
| CCT-DATA-019 | P3 | Manifest metadata | `DataManifest.build` always emits a P1-named format even when given P2 policy. | Version the format by policy or use a neutral manifest identifier. |

## Remediation Status

| ID | Current status | Resolution evidence or remaining boundary |
|---|---|---|
| CCT-ARCH-001 | **Implementation resolved; empirical gate pending** | Fixed zero-sum contrast readout, exact geometry-free counterpart, pre-registration, signal tests, and G3.1 harness are complete. The three-seed held-out value test is not yet run. |
| CCT-EMP-002 | **Resolved** | Executable verdict now requires finite complete records, learning, Transformer tolerance, and per-seed CDI-versus-GRU relation. |
| CCT-RUN-003 | **Contained** | `run.sh` is safe readiness/status only and rejects legacy/production commands. |
| CCT-DATA-004 | **Contained** | The split-leaking production trainer is fail-closed; no metric can be produced from that path. A future replacement requires a reviewed data contract. |
| CCT-DATA-005 | **Contained** | External ingestion is fail-closed, typed, and has no network, authentication, or remote-code path. |
| CCT-TEST-006 | **Resolved** | LM-level tests require full/G logit and loss divergence, finite nonzero geometry gradient, exact ablation, and parameter matching. |
| CCT-CKPT-007 | **Resolved** | Resume validates tokenizer, model, topology, manifest, training config, and complete Stage C config fingerprints. |
| CCT-ART-008 | **Resolved** | P1 and Stages B–F write generated reports under results directories; regression guards forbid tracked Stage report writes. |
| CCT-DOC-009 | **Resolved** | README, package exports, safe runner, Colab guide, and recursive test guidance identify active CCT v3 and blocked legacy routes. |
| CCT-TRAIN-010 | **Open evidence boundary** | The current experiment is explicitly chunk-local. State-carrying long-context training remains a later gated task. |
| CCT-LEG-011 | **Contained** | Legacy v2 is labeled compatibility/reference only and excluded from safe CCT entry points. |
| CCT-MATH-012 | **Resolved for the nano runtime** | Edge-weight, explicit-step, dissipation, geometry-energy, and state-norm guards fail closed; stress tests cover rejection paths. |
| CCT-EVAL-013 | **Resolved** | Generic evaluation is causal-token-weighted and covered by unequal-token-count test. |
| CCT-TRAIN-014 | **Resolved** | Unimplemented warmup/accumulation fields were removed; deterministic epoch permutations are cached. |
| CCT-INF-015 | **Resolved** | Checkpoints serialize/fingerprint complete Stage C dynamics; inference validates it and slices completion by token prefix. |
| CCT-PERF-016 | **Partially resolved; empirical optimization deferred** | Immutable topology tensors are cached exactly. Python-token-serial execution remains, and throughput optimization is blocked until G3.1 produces a mechanism result. |
| CCT-CFG-017 | **Resolved** | Strict offline CPU schema records EthioBBPE 2.0.0, rejects unknown fields, and cannot authorize production training. |
| CCT-API-018 | **Resolved** | Top-level v2 compatibility and active `cdi.v3` roles are explicitly separated. |
| CCT-DATA-019 | **Resolved** | Governed manifest format is neutral `dcss-cdi-governed-data-manifest-v2`. |

## Detailed Findings

### CCT-ARCH-001 — Geometry Is Unobservable from the Active Language Loss

**Severity:** P0 — gate-blocking.  
**Affected source:** `cdi/v3/ssm.py` lines 473–497; `cdi/v3/laplacian.py` lines 44–65; `cdi/v3/language_model.py` lines 76–102.

The cell first computes a vertex-resolved band update and then subtracts a weighted graph Laplacian correction. The readout subsequently averages every band across vertices before it reaches token logits. For the implemented \(L=S^\top W S\), \(\mathbf{1}^\top L=0\). Therefore the mean readout is invariant under the correction. Because the band generator is shared across vertices and gates depend on token input rather than vertex-resolved state, this invariance persists over time for the mean feature seen by the output head.

The review verified the effect with identical full and geometry-disabled language models. The maximum logit difference was `2.95585778076e-12`, the causal-loss difference was `0`, and the geometry-weight causal-loss gradient had maximum absolute value and L2 norm `0`. In contrast, the vertex-resolved states did change. The current geometry is therefore a state contrast transformation with no usable token-prediction path.

| Required repair | Acceptance test |
|---|---|
| Pre-register one readout-access mechanism that exposes a vertex-contrast component while preserving causal recurrence and the frozen G2.1 data/budget contract. A fixed zero-sum vertex basis plus a parameter-aware contrast readout is a candidate; the exact mechanism must be specified before coding. | With identical weights and nontrivial token input, full and geometry-disabled models must show a predetermined nonzero logit/loss difference above numerical tolerance. |
| Keep the geometry-free variant as an exact removal of the geometry operation, not a second model redesign. | Full-model causal loss must produce finite, nonzero geometry-weight gradient; the geometry-disabled variant must produce no geometry contribution. |
| Run one CCT-G3.1 comparison only after the above unit tests pass. | The full versus geometry-free difference must be measured across the frozen three-seed, 1,000-step G2.1 protocol. |

**Do not do:** add data, steps, state size, or context to conceal this mechanism-level defect.

### CCT-EMP-002 — Harness Decision Does Not Encode the Governing CCT Gate

**Severity:** P0 — gate-blocking.  
**Affected source:** `benchmarks/ethiobbpe_synaxarium_pilot.py` lines 283–297; `Todo.md` CCT-G2.1 transition gate.

The harness selects the lower loss of GRU and Transformer as `best_baseline` and declares `EARNED_NEXT_PILOT` whenever CDI learns and is within a percentage tolerance of that single value. The governing CCT-G2.1 gate also requires CDI to consistently match or beat GRU. In the submitted G2.1 result CDI was within 0.58% of Transformer but above GRU in every seed, so the harness emitted `EARNED_NEXT_PILOT` while the correct CCT decision was `REDESIGN_BEFORE_SCALE`.

| Required repair | Acceptance test |
|---|---|
| Encode all CCT-G3.1/G2.1 decision predicates in the executable report: learning, finite values, split isolation, Transformer tolerance, and per-seed CDI-versus-GRU relation. | A synthetic summary where CDI loses to GRU in one seed must produce a failed CCT verdict even if it is within Transformer tolerance. |
| Distinguish `harness_status` from `cct_transition_status` if both are retained. | `REPORT.md` and `latest.json` must contain one unambiguous final transition status. |

### CCT-RUN-003 — Default Shell Runner Bypasses CCT Discipline

**Severity:** P0 — gate-blocking.  
**Affected source:** `run.sh` lines 19–39.

The default shell script upgrades dependencies, runs tests, and immediately invokes `cdi.v3.production.train_production`. That route downloads external corpora and is not the active CCT pilot path. A user following the convenience entry point can therefore bypass the CCT-G2.1 failure stop and begin a different training system.

| Required repair | Acceptance test |
|---|---|
| Replace the default runner with a non-training CCT status/check command, or make external production execution require an explicit, separate command and a checked release-boundary file. | Running the default script must not download data, authenticate, train, or write a production checkpoint. |
| Print the active CCT goal and explicit block on CCT-G2.2. | A shell test confirms no unapproved command is reachable without an explicit user-approved flag. |

### CCT-DATA-004 — Production Training Leaks Held-Out Documents into Training

**Severity:** P0 — data integrity.  
**Affected source:** `cdi/v3/production/hf_ingest.py` lines 104–129; `cdi/v3/production/train_production.py` lines 75–91.

Ingestion writes every document, including manifest validation and test rows, into one JSONL file. Production training reads that entire file and builds a single training set. The manifest split is not used to select train documents, and validation/test evaluation is not performed. Any metric from this path cannot support a held-out generalization claim.

| Required repair | Acceptance test |
|---|---|
| Load text by manifest split and permit training only on IDs listed in `train`. | A test injects unique sentinel documents into validation/test and proves their IDs never reach packed training batches. |
| Evaluate validation and test through a token-weighted evaluator and bind evaluation evidence to the manifest fingerprint. | Saved report has separate train/validation/test metrics and disjoint ID/hash checks. |

### CCT-DATA-005 — External Ingestion Is Unsafe and Not Governance-Verified

**Severity:** P0 — safety and provenance.  
**Affected source:** `cdi/v3/production/hf_ingest.py` lines 18–129.

The ingestion function may log into Hugging Face using an environment token, calls public network endpoints, requests datasets with `trust_remote_code=True`, hard-codes source license/PII labels, and uses `sys.exit` inside a library function. The code has not established that those hard-coded governance labels are authoritative for every source revision, and remote code execution is unnecessary for a fetch-only data pipeline.

| Required repair | Acceptance test |
|---|---|
| Disable this route pending a reviewed source registry with immutable dataset revision, license source, schema, data-class approval, and content fingerprint. Remove `trust_remote_code=True` unless a reviewed exception is recorded. | Offline/unit tests validate the registry; no ingestion call can execute remote code by default. |
| Return typed exceptions rather than process-wide `sys.exit`. | Callers can catch ingestion failure and preserve a clean process/report. |

### CCT-TEST-006 — Geometry Tests Validate the Wrong Objective

**Severity:** P1 — false confidence.  
**Affected source:** `tests/stage_c/test_stage_c.py` lines 71–91 and 128–133; `benchmarks/stage_c.py` lines 289–326.

The tests show that `geometry.apply` is nonzero and that `edge_log_weights.grad` is not `None` for a squared raw SSM-output objective. They do not require a nonzero gradient norm, do not exercise tied token logits or causal loss, and do not compare full versus geometry-disabled language models. A zero tensor gradient passes the non-`None` test, exactly as in the review probe.

| Required repair | Acceptance test |
|---|---|
| Add a deterministic LM-level geometry-signal test after the CCT-G3.1 redesign. | Assert full versus G logits/loss diverge above a declared tolerance and full geometry gradient norm is finite and greater than a declared epsilon. |
| Add a negative test for the pre-redesign mean-only readout to preserve the diagnosis. | The test must fail if a future change silently reintroduces mean-only geometry cancellation. |
| Report geometry-to-loss activity in Stage C/G3 reports, not raw SSM-only activity. | Gate report includes numerical gradient norm and full-vs-G loss difference. |

### CCT-CKPT-007 — Generic Resume Is Not Fully Bound to Its Saved Contract

**Severity:** P1 — reproducibility.  
**Affected source:** `cdi/v3/training.py` lines 291–319.

Stage D checkpoints save tokenizer artifact, data manifest, training config, topology fingerprint, random state, and optimizer state. `restore_checkpoint`, however, verifies only the checkpoint format and tokenizer fingerprint before loading the model and optimizer. It does not compare the saved topology fingerprint, manifest fingerprint, configuration fingerprint, model fingerprint, or allowed device/dtype contract. Production inference performs substantially stricter checks, but generic training resume does not.

| Required repair | Acceptance test |
|---|---|
| Add strict default validation for model, topology, manifest, configuration, and tokenizer fingerprints; require named, auditable override flags for each exceptional conversion. | Resume fails for a mismatched manifest, topology, config, or model even when tokenizer is identical. |
| Preserve the successful deterministic shuffled-resume test. | Exact continuation still matches after strict validation. |

### CCT-ART-008 — Multiple Benchmark Runners Overwrite Tracked Reports

**Severity:** P1 — artifact integrity.  
**Affected source:** `benchmarks/p1_readiness.py`, `benchmarks/stage_b.py`, `benchmarks/stage_c.py`, `benchmarks/stage_d.py`, `benchmarks/stage_e.py`, and `benchmarks/stage_f.py`.

Although P2 was repaired, these runners still write generated reports directly into tracked `Stages/` paths. A normal execution can dirty `master`, overwrite an earlier report, and make the source tree depend on the latest machine run.

| Required repair | Acceptance test |
|---|---|
| Write all machine-generated report, manifest, and analysis files below the configured ignored results directory. Keep `Stages/` documents manually authored or generated only through an explicit reviewed publish command. | Execute every benchmark smoke path and assert `git status --porcelain` remains empty. |
| Include a reusable report path in terminal output. | Every runner returns an output directory containing `REPORT.md` and `latest.json` where applicable. |

### CCT-DOC-009 — Documentation and Entry Points Describe the Wrong System

**Severity:** P1 — operational correctness.  
**Affected source:** `README.md`, `run_tests.py`, `cdi/__init__.py`, `cdi/v3/__init__.py`, and portions of legacy `colab.md` guidance.

The README says Transformers are required even though they were deliberately removed, advertises unsupported CLI/API signatures, presents legacy v2 data/training as the normal quickstart, and does not identify CCT v3 as the empirical path. `run_tests.py` calls itself a full test orchestrator but discovers only nonrecursive top-level `tests/test_*.py`, omitting stage, production, and P2 tests. The top-level `cdi` import exports v2 while CCT uses `cdi.v3`; the v3 docstring itself says it has no language model although it exports one.

| Required repair | Acceptance test |
|---|---|
| Add one prominent active-path statement: CCT uses `cdi.v3`, EthioBBPE, and the gated Todo workflow. Label v2/production routes as legacy or blocked. | Documentation command tests run in a fresh CPU environment without unsupported options, stale packages, or external training. |
| Make `pytest -q` the full-suite command or make `run_tests.py` recursively collect every test. | Test runner discovery count matches `pytest --collect-only -q`. |
| Align imports and module descriptions with actual public support. | API smoke tests import the documented modules and execute the documented minimal example. |

### CCT-TRAIN-010 — The Active Pilot Is Chunk-Local, Not Long-Context

**Severity:** P1 — evidence boundary.  
**Affected source:** `cdi/v3/training.py` lines 151–182; `benchmarks/ethiobbpe_synaxarium_pilot.py` lines 142–177; `cdi/v3/language_model.py` lines 54–88.

Documents are split into fixed nonoverlapping chunks, and each causal-loss call begins with a zero state. No state is carried from one chunk to the next, including adjacent chunks from the same document. The state-space mechanism is valid inside the configured chunk, but the current training/evaluation protocol does not train or test document-length memory.

| Required repair | Acceptance test |
|---|---|
| Keep this as an explicit short-context boundary for CCT-G3.1. Do not market it as long-context evidence. | G3.1 report states chunk length, reset behavior, and no long-context claim. |
| After G3.1 passes, introduce a separate state-carrying document protocol with clear detach/truncation semantics and matched baseline context controls. | A test proves contiguous chunk processing with carried state matches single-document streaming output under the chosen gradient policy. |

### CCT-LEG-011 — Legacy v2 Training Is Not CCT-Compatible

**Severity:** P1 — execution confusion and invalid evidence risk.  
**Affected source:** `train.py`, `dataset.py`, `cdi/engine.py`, `cdi/config.py`.

The legacy path is architecturally distinct. It owns manual tensors rather than `nn.Module` parameters, builds dense operators around a flat state, uses external English WikiText/SciQ while the active tokenizer is EthioBBPE, pads QA examples without masking pad targets in its loss, and relies on `trust_remote_code=True`. The advertised v2 `small` preset has a 393,216-element flat state before dense operator construction; dense matrices at that scale are not a demonstrated feasible CPU route.

| Required repair | Acceptance test |
|---|---|
| Make legacy status explicit in file headers, README, and entry points; prevent it from being selected by CCT commands. | A CCT smoke command cannot import or launch `train.py` accidentally. |
| If legacy work is retained, give it its own tokenizer/domain/data/masking/reproducibility specification and bounded hardware feasibility test. | No legacy result is labeled CCT without a separate approved protocol. |

### CCT-MATH-012 — Stability Claims Exceed Enforced Runtime Controls

**Severity:** P2 — numerical assurance.  
**Affected source:** `cdi/v3/ssm.py` lines 35–125 and 473–497; `cdi/v3/config.py`; `cdi/v3/diagnostics.py`.

The exact pairwise dissipative update is a real stability strength. However, `input_bound`, `state_norm_bound`, `energy_limit`, `spectral_target`, and `allocation_fraction_limit` are mostly configuration or diagnostic values, not per-step production enforcement. The geometry correction is explicit, while learned edge weights have no spectral upper constraint. The optional `dissipation_scale` argument can be negative without validation.

| Required repair | Acceptance test |
|---|---|
| After CCT-G3.1 establishes a real geometry signal, bound the geometry step against a provable/estimated spectral limit or parameterize a stable correction directly. Validate nonnegative dissipation scaling. | Stress tests over declared inputs/weights either remain within bounds or fail closed with a diagnostic. |
| Separate diagnostic thresholds from enforcement thresholds in configuration. | Configuration audit lists every declared field as enforced, diagnostic-only, or deprecated. |

### CCT-EVAL-013 — Generic Evaluation Is Not Token-Weighted

**Severity:** P2 — metric correctness outside the pilot harness.  
**Affected source:** `cdi/v3/training.py` lines 266–279.

The generic `evaluate` function averages per-batch loss. Because batch padding and active-token counts can differ, this is not the same as corpus cross-entropy. The active Synaxarium harness correctly multiplies each loss by token count before aggregation, so the recorded G2.1 value is not affected by this helper.

| Required repair | Acceptance test |
|---|---|
| Replace generic evaluation with token-weighted aggregation or clearly mark it diagnostic-only. | A deliberately unequal-length fixture matches a hand-computed token-level cross-entropy. |

### CCT-TRAIN-014 — Declared Training Features Are Unimplemented or Inefficient

**Severity:** P2 — contract hygiene.  
**Affected source:** `cdi/v3/training.py` lines 20–63 and 215–263.

`StageDConfig` declares `gradient_accumulation` and `warmup_steps`, but `train_steps` uses neither. When shuffle is enabled, it recreates the entire permutation on every optimizer step instead of once per epoch. This does not invalidate the current G2.1 run, whose relevant shuffle behavior was deterministic, but it makes configuration fields misleading and wastes CPU work at scale.

| Required repair | Acceptance test |
|---|---|
| Implement the declared features with exact resume semantics or remove them from the public config. Cache each epoch permutation deterministically. | Tests verify accumulation/warmup behavior or verify their absence from schema; resumed shuffled runs remain bitwise-equivalent where supported. |

### CCT-INF-015 — Inference Restores Dimensions but Not the Complete Dynamics Contract

**Severity:** P2 — forward compatibility.  
**Affected source:** `cdi/v3/production/inference.py` lines 90–134 and 289–294.

Inference derives `StageCConfig` from tensor shapes and default nano values. This works for the current default configuration, but a valid checkpoint with nondefault `dt`, time ranges, caps, or other dynamic fields cannot be faithfully reconstructed. `complete()` also strips the prompt using decoded string prefix matching rather than token-prefix accounting.

| Required repair | Acceptance test |
|---|---|
| Serialize and fingerprint complete Stage C configuration in checkpoints; reconstruct only from the saved validated config. | A nondefault supported dynamic configuration round-trips exactly and mismatched config is rejected. |
| Decode continuation tokens after removing the known token-prefix length. | Whitespace/normalization fixtures return continuation only without relying on string prefix matching. |

### CCT-PERF-016 — Active Throughput Is Dominated by Python-Serial Work

**Severity:** P2 — performance.  
**Affected source:** `cdi/v3/language_model.py` lines 76–82; `cdi/v3/ssm.py` lines 534–555; `cdi/v3/topology.py` lines 65–82.

Both language-model and SSM chunks execute token-by-token Python loops. Topology properties create index/incidence tensors on access. This design is correct for the nano diagnostic but is not an efficient state-space kernel and cannot justify a speed claim.

| Required repair | Acceptance test |
|---|---|
| Do not optimize before G3.1. After the mechanism has a measurable signal, profile fixed workloads, cache immutable topology tensors as buffers, and introduce vectorization/compiled kernels only with reference-equivalence tests. | Before/after benchmark holds hardware, precision, model budget, context, warmup, token count, and decoding method fixed; output/gradient/state equivalence is documented. |

### CCT-CFG-017, CCT-API-018, and CCT-DATA-019 — Configuration and Namespace Debt

**Severity:** P2/P3 — maintainability.

`ProductionRunConfig` retains `tokenizer_version="stage-d-character-v1"` despite EthioBBPE, mixes P1 CPU restrictions with GPU/large-corpus defaults, and silently filters unknown JSON fields. `cdi.__init__` exports legacy v2 while `cdi.v3.__init__` makes stale Stage B claims and exports optional capability utilities beside active CCT objects. `DataManifest.build` always writes `dcss-cdi-p1-data-manifest-v1` even when a P2 policy is supplied. These inconsistencies do not change the submitted CCT-G2.1 metrics, but they make future misuse more likely.

| Required repair | Acceptance test |
|---|---|
| Establish one versioned CCT configuration schema with EthioBBPE identity, explicit policy phase, strict unknown-key rejection, and no mixed production defaults. | Parse/round-trip tests reject obsolete tokenizer labels and unknown fields. |
| Separate legacy, active CCT, and optional experimental namespaces in import/documentation surfaces. | Each documented import maps to one declared execution role. |
| Use a neutral or phase-specific manifest format version. | P1/P2 manifest formats and fingerprints are unambiguous. |

## Required CCT-G3.1 Pre-Registration Before Any Code Change

CCT-G2.1 failure does not authorize uncontrolled redesign. The first permitted experiment must isolate the observed state-to-readout mechanism. The following items are mandatory before implementation or Colab execution.

| Pre-registration field | Required content |
|---|---|
| Hypothesis | A vertex-contrast-aware readout makes the sparse Laplacian observable to causal token loss; if geometry is useful, full CDI should differ from and outperform the exact geometry-disabled variant under the frozen G2.1 protocol. |
| One changed mechanism | State-to-readout access only. The recurrence, tokenizer, corpus manifest, split, optimizer, context, batch size, seed list, token budget, and evaluation regime remain fixed. |
| Candidate construction | Define a fixed zero-sum vertex basis or equivalent contrast statistic, concatenate it with the existing mean features, and use a declared parameter-aware readout. The full and G variants share that same readout; G alone disables the Laplacian correction. |
| Fairness control | Record both total and trainable parameter counts. If the readout addition exceeds the predetermined matching tolerance, add an explicitly documented parameter-matched control rather than changing hidden capacity. |
| Unit gates | Finite forward/backward values; full/G logits and causal losses differ above tolerance; full geometry gradient norm is finite and greater than epsilon; G has exact zero correction. |
| Empirical gate | Three seeds `[11, 29, 47]`, 321 documents, 1,000 steps, 30,000 causal positions/model/seed, chunk length 16, batch 2, deterministic shuffle, all held-out evaluation, and full/ablation/GRU/Transformer comparison. |
| Decision rule | Record CDI-to-G difference, full CDI-to-GRU and Transformer relations, and whether the mechanism adds repeated value. Do not unlock scale on an inconclusive or negative ablation. |

## Completion Checklist for This Backlog

- [x] CCT-G2.1 decision is recorded as `REDESIGN_BEFORE_SCALE`.
- [x] CCT-ARCH-001 has a reviewed, pre-registered CCT-G3.1 design and local observability gates; empirical value evidence remains pending.
- [x] CCT-EMP-002 is encoded in the executable decision report.
- [x] CCT-RUN-003, CCT-DATA-004, and CCT-DATA-005 are contained before any production/legacy convenience route is used.
- [x] CCT-TEST-006 adds language-model-level geometry reachability coverage.
- [x] CCT-ART-008 removes generated writes to tracked `Stages/` files.
- [x] CCT-DOC-009 makes active versus legacy execution paths unambiguous.
- [x] Non-scaling P2/P3 repairs are completed; long-context and throughput-optimization work remains explicitly gated on CCT-G3.1 evidence.

## Source References

[1]: [Active DCSS recurrence](cdi/v3/ssm.py)  
[2]: [Language model and baselines](cdi/v3/language_model.py)  
[3]: [Matrix-free Laplacian](cdi/v3/laplacian.py)  
[4]: [Active CCT pilot harness](benchmarks/ethiobbpe_synaxarium_pilot.py)  
[5]: [CCT-G2.1 decision](docs/CCT_G2_1_DECISION.md)  
[6]: [Stage C benchmark gate](benchmarks/stage_c.py)  
[7]: [Stage C tests](tests/stage_c/test_stage_c.py)  
[8]: [Training/checkpoint utilities](cdi/v3/training.py)  
[9]: [Production training route](cdi/v3/production/train_production.py)  
[10]: [Production ingestion route](cdi/v3/production/hf_ingest.py)  
[11]: [Default shell runner](run.sh)  
[12]: [Legacy training route](train.py)  
[13]: [Legacy data route](dataset.py)  
[14]: [Verified inference](cdi/v3/production/inference.py)  
[15]: [Repository README](README.md)
