#!/usr/bin/env bash
# CCT-safe repository entry point.
#
# This script intentionally does not fetch external data, authenticate, or begin
# training. CCT-G2.1 concluded REDESIGN_BEFORE_SCALE, so the only executable
# default is reproducible readiness validation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: ./run.sh [readiness|status|help]

Commands:
  readiness  Run the clean-master CCT-G0 environment and regression verifier.
  status     Print the current governed CCT execution state without side effects.
  help       Print this message.

The legacy v2 and external production routes are intentionally blocked here.
CCT-G2.2 scale work is blocked pending the CCT-G3.1 mechanism decision.
EOF
}

status() {
  echo "CDI CCT safe entry point"
  echo "branch: $(git branch --show-current)"
  echo "revision: $(git rev-parse --short HEAD)"
  echo "active gate: CCT-G3.1 — one pre-registered geometry-observability mechanism"
  echo "blocked: legacy v2 training, external production ingestion, CCT-G2.2 scale ladder"
  echo "authoritative checklist: Todo.md"
}

command="${1:-readiness}"
case "$command" in
  readiness)
    exec ./scripts/run_cct_g0.sh
    ;;
  status)
    status
    ;;
  help|-h|--help)
    usage
    ;;
  legacy|production|train|*)
    echo "ERROR: '$command' is not an approved CCT command." >&2
    usage >&2
    exit 2
    ;;
esac
