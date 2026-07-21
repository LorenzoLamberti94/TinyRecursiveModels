#!/bin/bash
# ============================================================================
# 01 — Build the Python environment once, on top of the selected rootless base.
# Enters the base (uenv or container) and runs _build_venv.sh inside it.
#
#   source cscs/config.sh
#   bash cscs/01_setup_env.sh
#
# Runs a short debug-partition job (needs a GPU for the adam-atan2 smoke test).
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/config.sh"

SRUN="srun -A ${CSCS_ACCOUNT} -p debug -t 00:30:00 --nodes=1 --ntasks=1 --gpus-per-node=1"

case "${TRM_BACKEND}" in
  uenv)
    echo ">>> using uenv ${TRM_UENV} (view ${TRM_UENV_VIEW})"
    uenv image pull "${TRM_UENV}" 2>/dev/null || true    # no-op if already pulled
    ${SRUN} --uenv="${TRM_UENV}" --view="${TRM_UENV_VIEW}" bash "${HERE}/_build_venv.sh"
    ;;
  container)
    echo ">>> using container ${TRM_IMAGE} (env 'trm')"
    ${SRUN} --environment=trm bash "${HERE}/_build_venv.sh"
    ;;
  *)
    echo "ERROR: TRM_BACKEND must be 'uenv' or 'container'"; exit 1 ;;
esac
