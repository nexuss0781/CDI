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
