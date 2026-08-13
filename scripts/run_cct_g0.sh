#!/usr/bin/env bash
# CCT-G0 reproducible execution readiness runner.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "$(git branch --show-current)" != "master" ]]; then
  echo "CCT-G0 requires the master branch." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "CCT-G0 requires a clean working tree before validation." >&2
  git status --short >&2
  exit 1
fi

COMMIT="$(git rev-parse --short HEAD)"
OUTPUT_DIR="${1:-results/cct_g0/${COMMIT}}"
mkdir -p "$OUTPUT_DIR"

cp "$0" "$OUTPUT_DIR/commands.sh"
chmod +x "$OUTPUT_DIR/commands.sh"
git rev-parse HEAD > "$OUTPUT_DIR/code_revision.txt"
cat requirements.txt > "$OUTPUT_DIR/requirements.txt"

python -m pip install -r requirements.txt | tee "$OUTPUT_DIR/requirements_install.txt"
python -m pip check | tee "$OUTPUT_DIR/pip_check.txt"
python scripts/cct_g0_environment.py --output "$OUTPUT_DIR/environment.json" | tee "$OUTPUT_DIR/environment.txt"
pytest -q | tee "$OUTPUT_DIR/pytest.txt"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "CCT-G0 regression run modified the working tree." >&2
  git status --short >&2
  exit 1
fi

cat > "$OUTPUT_DIR/REPORT.md" <<REPORT
# CCT-G0 Reproducible Execution Readiness

**Verdict:** \`READY_FOR_NEXT_GOAL\`

| Field | Value |
|---|---|
| Branch | \`master\` |
| Code revision | \`$(git rev-parse HEAD)\` |
| Working tree | clean after dependency and regression validation |
| Tokenizer backend | \`EthioBBPE==2.0.0\` verified by the environment record |
| Regression evidence | \`pytest.txt\` |

The recorded command, dependency plan, runtime environment, and regression output are stored in this directory.
REPORT

echo "CCT-G0 READY_FOR_NEXT_GOAL; report=$OUTPUT_DIR/REPORT.md"
