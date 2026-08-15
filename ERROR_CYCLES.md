# Error Cycles

## EthioBBPE tokenizer migration

The legacy CDI path tokenized training data with EthioBBPE while the active `CDITokenizer` compatibility wrapper used a small character vocabulary and silently clamped input IDs. This destroyed token identity for IDs outside the character vocabulary and made the data/model contract invalid.

The corrective work replaces the active tokenizer adapter with an EthioBBPE-backed artifact, validates every ID range instead of clamping, and makes checkpoint restoration use the exact serialized tokenizer snapshot.

The first focused Stage D run exposed two obsolete character-tokenizer assertions: one prohibited the EthioBBPE dependency and another expected a Unicode snowman to become an unknown token. Both were replaced with EthioBBPE dependency, range, round-trip, and artifact-restoration checks. The focused Stage D suite then passed: 9 tests passed.

The first artifact-generation invocation ran from outside the repository and could not import `cdi`. Re-running the saved script with `PYTHONPATH=.` generated the 16,000-token EthioBBPE snapshot successfully at `benchmarks/configs/stage_d_tokenizer.json` with fingerprint `9d02c723406029a95f8abfd479a1fc819fc4a0e9f698b655b2f8d7e87bbb554d`.

The updated legacy integration runner initially marked `W_iota` as severed because its finite initial gradient was `6.39e-09`, below the generic `1e-08` diagnostic threshold after the tied output vocabulary increased to 16,000 EthioBBPE tokens. The `W_iota` diagnostic now uses a documented `1e-12` projection threshold while all other checks retain their existing thresholds.

Final validation passed after the migration: the complete `pytest -q` suite passed 245 tests in 45.48 seconds, the focused Stage D suite passed 9 tests, production checkpoint/inference tests passed 7 tests, `run_tests.py` completed every phase successfully, and `bash -n run.sh` passed.

## Real Synaxarium pilot duplicate-content rejection

The first full 60-document pilot build failed before training because the governed data manifest correctly detected duplicate Synaxarium text (`synaxarium-ሕዳር-19` matched `synaxarium-ሕዳር-14`). The pilot loader now computes a SHA-256 content hash before selection, retains only the first stable unique document, and then constructs the 70/15/15 document-level splits. This preserves the manifest’s no-duplicate and no-leakage contract rather than weakening it.

## Pilot-artifact archive filename correction

The first artifact-archiving command used the same misspelled destination filename as the source, so the shell correctly refused the no-op move. The archived verdict is being renamed to `ETHIOBBPE_SYNXARIUM_PILOT_VERDICT.md` before the evidence commit.

## Fresh CPU Colab bootstrap repair

A fresh Colab run cloned a non-main branch, attempted an impossible fast-forward to the feature branch, and then imported CDI before `requirements.txt` installed EthioBBPE. The package index reported `EthioBBPE 2.0.0` as the sole and latest published version, so requirements now pin `EthioBBPE==2.0.0`. The Colab bootstrap now removes any stale checkout, clones `master` directly, installs requirements, imports `ethiobbpe` as an explicit verification step, and only then runs CDI.

## Colab dependency-resolver conflict

The stale Colab requirements install spent time backtracking through `transformers` versions because EthioBBPE 2.0.0 requires `tokenizers>=0.20.0,<0.22`, whereas recent Transformers releases require newer Tokenizers versions. CDI imports neither `transformers` nor its APIs, so the unused runtime dependency was removed from `requirements.txt`. The pinned `EthioBBPE==2.0.0` installation is now compatible with the actual CDI runtime dependencies and resolves without the unrelated Transformer constraint.

## CCT-G0 tracked-artifact mutation repair

The P2 regression runner wrote its generated report to the tracked `Stages/P2_REAL_DATA_PILOT_REPORT.md` file. This left the working tree dirty after an otherwise successful test suite and violated the clean-checkout readiness gate. The runner now writes `REPORT.md` beside `latest.json` in the caller-selected output directory, and its regression test requires that local output contract.

## CCT-G0 dependency-install correction

The first CCT-G0 runner resolved `requirements.txt` with `--dry-run`, which exposed that the active environment lacked the declared SciPy dependency even though the tests happened to pass. A readiness gate must validate the installed contract, not only the resolver plan. The runner now installs `requirements.txt`, runs `pip check`, then records the environment and executes regressions.

## CCT-G3.1 hard geometry-cap execution failure

The first CCT-G3.1 Colab execution on master `bcc7e0b` completed setup and data loading but stopped during the first training path when AdamW drove a `softplus(edge_log_weights)` value above the new `max_geometry_edge_weight` guard. The guard correctly prevented an unbounded explicit Laplacian update, but its post-update hard failure also made the frozen experiment non-executable and produced no `REPORT.md` or `latest.json`.

The repair retains the same maximum edge-weight bound and explicit-step spectral envelope but changes the existing edge-weight map to `max_geometry_edge_weight * sigmoid(edge_log_weights)`. The raw parameter is initialized by converting the former softplus effective weights to equivalent logits, preserving the initial operator, parameter count, full/geometry-free comparison, and G3.1 budget. A dedicated regression now verifies finite differentiable gradients and strict positive-below-cap weights even for extreme raw logits. The CCT-G3.1 pre-registration contains Amendment A; a clean rerun is required before any empirical conclusion.

## CCT-G3.1 Colab host-memory safety threshold

The user reported that Colab repeatedly suspended the long CPU CCT-G3.1 job, with no reliable opportunity to recover an artifact after the runtime stopped. The pilot contract now supports a configurable process/container resident-memory monitor. The dedicated G3.1 command defaults to `--max-host-memory-gb 11`; it checks before and after every training step, every evaluation batch, and every model/seed release. It records configured, final, and peak GiB in completed artifacts and raises a stage-specific fail-closed error at the threshold. This is an execution-safety control only: it does not alter model dimensions, tokenizer, data split, context, seeds, optimizer, or token budget.

## CCT-G3.1 geometry-free inactive-gradient contract

The guarded CCT-G3.1 Colab run advanced past setup and the smooth edge-weight repair but stopped at the exact geometry-free CDI variant. The generic training loop correctly rejected `ssm.cell.geometry.edge_log_weights` because it had no gradient; however, that parameter is intentionally disconnected when the pre-registered ablation disables only the Laplacian correction. Treating it as an active parameter made the required control untrainable.

The repair adds an explicit `expected_inactive_trainable_parameters()` declaration to the geometry-free DCSS language model and permits only its named geometry edge parameter to have no gradient. The trainer verifies every declared name exists and retains finite-gradient checks for all other trainable parameters. A regression trains the geometry-free variant for one step and confirms that the declared edge gradient alone is absent. The full suite passed 271 tests after the change.

## 11 GiB CDI eager-kernel optimization

The initial optimized design reused stacked generator tensors and a differentiable geometry operator across each active chunk, but the recurrence still repacked the three-band state at every token. The second pass introduced a packed-state fused kernel that carries one stacked state tensor through the token loop and converts back to the public `CohomodynamicState` only at the chunk boundary. The refactor preserved the exact dense-path equivalence contract and removed repeated per-token stack/operator construction from the hot path.

A local official-protocol eager diagnostic on the optimization branch measured CDI at 3,422.10 / 2,962.18 / 2,696.02 token-positions/s for lengths 16 / 64 / 256, compared with the repository's retained 849.92 / 764.94 / 702.05 baseline. Peak RSS was 0.586 GiB, below the 11 GiB guard. The full regression suite passed 304 tests. The matched Transformer remained faster in this CPU/eager run, so no Transformer-superiority claim is made.

The first compiled rerun was blocked by missing Python development headers; installing `python3.12-dev` removed that environment issue, but the full three-length Inductor compilation exceeded the local five-minute execution window. The compiled result therefore remains unclaimed and requires a separate clean run on a sufficiently provisioned build environment.
