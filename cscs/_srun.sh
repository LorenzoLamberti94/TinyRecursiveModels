#!/bin/bash
# Backend dispatcher: runs the given command under srun, entering whichever
# rootless PyTorch base is selected by TRM_BACKEND in config.sh.
# Usage (from an sbatch script):  bash cscs/_srun.sh <command...>
set -euo pipefail
source "$(dirname "$0")/config.sh"

case "${TRM_BACKEND}" in
  uenv)
    exec srun --uenv="${TRM_UENV}" --view="${TRM_UENV_VIEW}" "$@"
    ;;
  container)
    exec srun --environment=trm "$@"
    ;;
  *)
    echo "ERROR: TRM_BACKEND must be 'uenv' or 'container' (got '${TRM_BACKEND}')" >&2
    exit 1
    ;;
esac
