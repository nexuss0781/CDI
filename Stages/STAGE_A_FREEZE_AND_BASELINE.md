# Stage A — Freeze CDI v2 and Establish the Reproducible Baseline

**Status:** Specification only. No implementation is authorized by this document.

## Stage objective

Stage A freezes the current CDI v2 behavior as a reference system and establishes the measurement infrastructure required for every later claim. The stage must separate **repository correctness**, **mathematical correctness**, **language-model behavior**, and **systems performance**. No new architecture may be introduced until the current implementation can be installed, executed, tested, and benchmarked from a clean environment.

The output is a reproducible baseline package named `legacy_v2_reference`. It is not required to be fast or high quality. It is required to be observable, repeatable, and sufficiently stable to serve as a comparison oracle for Stage B onward.

## Scope and non-goals

Stage A may repair packaging, missing test dependencies, deterministic seeding, logging, checkpoint serialization, and benchmark plumbing. It may not change the mathematical equations, model topology, tokenizer semantics, loss terms, optimizer semantics, or v2 forward path except where a defect prevents the documented behavior from executing. Every repair must be recorded as a compatibility patch.

Stage A does not introduce sparse operators, selective state-space dynamics, multi-timescale memory, new datasets, or a new tokenizer. It does not claim any improvement over v2.

## Required repository outputs

| Output | Required content | Completion condition |
|---|---|---|
| `legacy_v2_reference/` or equivalent tag | Immutable reference entry point for current CDI v2 | A clean checkout can invoke it without manual edits. |
| `benchmarks/runner.py` | Single command for correctness, speed, memory, scaling, and LM measurements | Runner records configuration, environment, seed, git revision, and timestamps. |
| `benchmarks/configs/` | Tiny deterministic benchmark configurations | Every configuration declares dimensions, dtype, device, sequence lengths, batch size, and step count. |
| `results/stage_a/` | Raw JSON/CSV results and human-readable summary | Raw results are never overwritten; each run has a unique identifier. |
| `tests/stage_a/` | Baseline reproducibility and serialization tests | All Stage A tests pass in a clean environment. |
| `docs/BASELINE_V2.md` | Baseline behavior, known defects, and measurement protocol | Later stages cite this document rather than informal terminal output. |
| `requirements-lock.*` | Reproducible dependency specification | Install succeeds without undeclared packages. |

The repository must expose an explicit version string, such as `cdi.__version__ = "2.x-reference"`, and every result file must store that version. If the repository is not currently a valid Git checkout, the work must first restore version-control metadata or record a cryptographic source snapshot so that later comparisons remain meaningful.

## Implementation requirements

### Environment and execution

Create a clean installation path using the declared Python version and a lockable dependency set. The test command must run without relying on packages that happen to exist globally. The harness must detect CPU versus CUDA, PyTorch version, available memory, operating-system information, and numerical precision.

Every benchmark invocation must accept a seed and set all relevant random generators. Data-loader workers must have deterministic worker seeds. The runner must record whether deterministic algorithms are enabled and must fail loudly when a requested deterministic mode is unavailable rather than silently changing behavior.

### Model and checkpoint handling

Provide a canonical v2 constructor and a canonical checkpoint format. The checkpoint must include model parameters, tokenizer parameters, configuration, optimizer state, scheduler state if present, global step, random states, source revision, and environment metadata.

The checkpoint test must instantiate a second process, restore the checkpoint, and verify that the next forward logits and loss match the original process within the declared tolerance for the selected dtype.

### Baseline benchmark suite

The runner must provide the following subcommands or equivalent modes:

```text
python -m benchmarks.runner correctness --config tiny --seed 42
python -m benchmarks.runner train-step --config tiny --steps 10 --seed 42
python -m benchmarks.runner scaling --lengths 16,32,64,128 --seed 42
python -m benchmarks.runner language --dataset synthetic --steps 100 --seed 42
python -m benchmarks.runner report --input results/stage_a/<run_id>
```

The suite must report parameter counts, trainable parameter counts, forward latency, backward latency, optimizer-step latency, peak resident memory or allocated device memory, tokens per second, loss, perplexity where defined, gradient norms, and all mathematical diagnostics already exposed by CDI v2.

### Reference datasets

The existing tiny synthetic corpus is the mandatory smoke dataset. WikiText-2 and SciQ may be used only if their versions, downloads, preprocessing hashes, and splits are recorded. The existing handwritten science questions must be labeled as a legacy diagnostic and must not be presented as a general benchmark.

## Evaluation harness

The harness must execute the following test groups in order.

| Test group | Method | Required record |
|---|---|---|
| Import and construction | Import the package, construct tiny/small configs, validate dimensions, build the engine | Config, version, device, dtype, and wall time. |
| Existing theorem tests | Run the repository’s mathematical tests | Test count, pass/fail, failure trace, tolerance settings. |
| Forward determinism | Run identical inputs twice under the same seed | Maximum absolute and relative output differences. |
| Backward determinism | Run loss/backward twice under controlled deterministic mode | Maximum gradient difference by parameter group. |
| Shape and finite checks | Exercise single, batch, empty-padding, and maximum configured lengths | Shapes, dtype, device, finite status. |
| Checkpoint round trip | Save, restore in a new process, compare next logits and loss | Differences and metadata equality. |
| Training smoke | Train on a tiny corpus long enough to overfit | Initial loss, final loss, gradient status, and step count. |
| Scaling baseline | Sweep sequence length and batch size | Latency, memory, tokens/s, and empirical exponents. |

### Required correctness tolerances

Tolerances must be explicit and dtype-aware. The default reference tolerances are `rtol=1e-5, atol=1e-6` for `float32` and `rtol=1e-10, atol=1e-12` for `float64`. If the current implementation requires looser tolerances, the measured tolerance becomes part of the baseline contract and must be justified.

### Required reports

Every run must produce `run.json`, `metrics.csv`, `environment.json`, `config.json`, `stdout.log`, and, on failure, a machine-readable failure record. The report must distinguish a test failure from an expected v2 limitation. A limitation is not a pass.

## Pass/fail gates

| Gate | Pass condition | Failure consequence |
|---|---|---|
| Clean installation | Installation and import succeed in a clean environment | Stop. Repair packaging before proceeding. |
| Test execution | All existing mandatory tests execute; no collection errors | Stop. Missing tests or dependencies are a failure. |
| Mathematical baseline | Existing v2 mathematical tests pass at declared tolerances | Stop. Preserve failure as a v2 defect and do not use that invariant as a later oracle until fixed. |
| Determinism | Same seed produces identical outputs within dtype tolerance | Fail Stage A unless nondeterminism is explicitly isolated and reported. |
| Checkpoint | Restored process reproduces next logits/loss within tolerance | Stop. No later training result is considered reproducible. |
| Gradient flow | Every intended trainable group receives finite gradients on the smoke batch | Stop. A disconnected parameter path invalidates the baseline. |
| Tiny overfit | Loss decreases monotonically after allowed warm-up and reaches a predeclared reduction target, such as ≥90% reduction on the tiny corpus | If it fails, diagnose optimization/data issues before any architecture comparison. |
| Measurement completeness | All required fields and raw artifacts are present | Fail the run; do not infer missing values. |

A Stage A pass does **not** require a target perplexity. It requires a trustworthy measurement process.

## Transition test to Stage B

Stage B may begin only when a fresh environment can execute the following transition protocol:

```text
1. Install the locked dependencies.
2. Run the Stage A correctness suite.
3. Restore the recorded v2 checkpoint.
4. Reproduce the Stage A smoke loss and output fingerprint.
5. Run the scaling benchmark twice.
6. Generate a baseline report with no manual edits.
```

The transition passes only if all mandatory tests pass, the output fingerprint matches, and the benchmark report contains no missing or hand-entered metrics. The Stage A tag/checkpoint and all raw results must be archived before Stage B code is added.

## Failure handling

A failed test must be classified as one of **environment**, **serialization**, **numerical**, **data**, **gradient**, **performance instrumentation**, or **known v2 algorithmic limitation**. The classification, reproduction command, stack trace, and corrective action must be stored in `results/stage_a/failures/`.

No later stage may weaken a Stage A gate to make a new engine appear better. If a baseline bug is repaired, the baseline must be re-run and assigned a new version.

## Stage A exit artifacts

The stage exits with a versioned v2 reference, clean installation instructions, a complete baseline report, checkpoint fingerprints, and a signed or hash-recorded results manifest. Only then is the project ready to build a sparse operator substrate without losing the ability to compare against the original engine.

## References

[1]: https://github.com/nexuss0781/CDI "CDI repository and current v2 implementation"
