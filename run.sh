#!/usr/bin/env bash
# CDI single entry point: mount Drive, select persistence, and run M1.1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Colab's Drive mount is performed from Python because google.colab.drive is a
# Python API. Outside Colab, the script uses a local .drive fallback so the
# entry point remains testable without pretending it is persistent Drive.
python - <<'PY'
from pathlib import Path

try:
    from google.colab import drive  # type: ignore
except ImportError:
    pass
else:
    mount_root = Path('/content/drive/MyDrive')
    if not mount_root.exists():
        print('Mounting Google Drive at /content/drive ...')
        drive.mount('/content/drive')
    else:
        print('Google Drive is already mounted at /content/drive.')
PY

if [[ -d /content/drive/MyDrive ]]; then
  export CDI_DRIVE_ROOT="/content/drive/MyDrive/CDI"
  echo "Persistent artifact root: $CDI_DRIVE_ROOT"
else
  export CDI_DRIVE_ROOT="${CDI_DRIVE_ROOT:-$ROOT/.drive}"
  echo "Google Drive is unavailable; using local fallback: $CDI_DRIVE_ROOT"
fi

case "${1:-run}" in
  run|train)
    # First run or resume: existing M1.1 evidence is never overwritten.
    exec bash "$ROOT/bash.sh"
    ;;
  retrain)
    # Explicit retraining: create a new isolated Drive session and preserve the
    # original report/checkpoint and every previous retraining session.
    export CDI_NEW_SESSION=1
    export CDI_SESSION_ID="${CDI_SESSION_ID:-m1_1_retrain_$(date -u +%Y%m%dT%H%M%SZ)}"
    echo "New M1.1 session: $CDI_SESSION_ID"
    exec bash "$ROOT/bash.sh"
    ;;
  status)
    echo "CCT safe entry point (CDI)."
    echo "CCT-G3.1 compatibility route: CDI."
    echo "CDI entry point ready."
    echo "Repository: $ROOT"
    echo "Persistent root: $CDI_DRIVE_ROOT"
    echo "Usage: bash run.sh       # run or show persisted M1.1"
    echo "       bash run.sh retrain # start a new isolated M1.1 session"
    ;;
  *)
    echo "not an approved CCT command: ${1:-}" >&2
    echo "Usage: bash run.sh [run|retrain|status]" >&2
    exit 2
    ;;
 esac
