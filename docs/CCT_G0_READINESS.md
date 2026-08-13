# CCT-G0 Reproducible Execution Readiness

> **Scope:** This record covers CCT-G0 only. It records the reproducible environment, dependency integrity, full regression result, and clean-working-tree decision required before any later CCT sprint.

## Verdict

**`READY_FOR_NEXT_GOAL`**

The CCT-G0 runner completed successfully on a clean `master` checkout. It installed the declared dependency contract, verified package integrity, recorded the runtime environment, ran the full regression suite, and confirmed that test execution did not modify tracked source artifacts.

| Evidence field | Recorded result |
|---|---|
| Validated code revision | `a03814705a73a3cd36658e3d0780a982593070f9` |
| Branch | `master` |
| Tokenizer backend | `EthioBBPE==2.0.0` |
| Dependency integrity | `python -m pip check` reported no broken requirements |
| Python | CPython 3.12.3 |
| PyTorch | 2.13.0+cu130, default dtype `torch.float32` |
| Hardware used for this validation | CPU-only; CUDA unavailable |
| Full regression result | `246 passed` |
| Working tree after test run | Clean |

## CCT-G0 Repairs Included

The readiness result includes two reproducibility repairs made before validation. The dependency contract pins EthioBBPE 2.0.0 and removes an unused Transformers requirement that conflicted with EthioBBPE’s Tokenizers constraint. The P2 runner now writes generated reports into its configured results directory instead of modifying a tracked `Stages` artifact during tests.

## Re-run Command

Run the committed verifier from a clean `master` checkout:

```bash
./scripts/run_cct_g0.sh
```

The runner writes `REPORT.md`, `environment.json`, `environment.txt`, dependency-install output, `pip_check.txt`, `pytest.txt`, the exact command script, requirements snapshot, and code revision under `results/cct_g0/<commit>/`.

## Transition

CCT-G0 is complete. The next permitted activity is the already-scoped **CCT-G2.1 full-corpus 1,000-step diagnostic**. No later CCT goal is unlocked by this record.
