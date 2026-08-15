#!/usr/bin/env bash
# Single CDI training entry point. The implementation lives only in bash.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/bash.sh" "$@"
