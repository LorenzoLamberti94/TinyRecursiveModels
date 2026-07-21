#!/bin/bash
# ============================================================================
# 01 — Build the Python environment INSIDE the container (once).
# Run on a compute node via the container engine (has nvcc for adam-atan2):
#
#   source cscs/config.sh
#   srun -A $CSCS_ACCOUNT -p debug -t 00:30:00 --nodes=1 --ntasks=1 \
#        --environment=trm bash cscs/01_setup_env.sh
#
# Creates a venv at $TRM_VENV that OVERLAYS the container's torch
# (--system-site-packages), so we keep NGC's GH200-optimized PyTorch and
# only add the repo's extra deps + compile adam-atan2 for sm_90.
# ============================================================================
set -euo pipefail
source "$(dirname "$0")/config.sh"

echo ">>> container python: $(python --version), torch: $(python -c 'import torch;print(torch.__version__, torch.version.cuda)')"

python -m venv --system-site-packages "${TRM_VENV}"
source "${TRM_VENV}/bin/activate"
pip install --upgrade pip wheel setuptools

# Repo deps EXCEPT torch (keep the container's) and adam-atan2 (built below).
grep -vE '^(torch|adam-atan2)\b' "${TRM_ROOT}/requirements.txt" > /tmp/trm_reqs.txt
pip install -r /tmp/trm_reqs.txt

# adam-atan2: compile against the container's torch for Hopper (sm_90).
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    pip install --no-cache-dir --no-build-isolation adam-atan2==0.0.3

echo ">>> smoke test:"
python - <<'PY'
import torch, adam_atan2
from adam_atan2 import AdamATan2
p = torch.nn.Parameter(torch.randn(8, device="cuda"))
opt = AdamATan2([p], lr=1e-3); p.sum().backward(); opt.step()
print("OK  cuda:", torch.cuda.is_available(), "| device:", torch.cuda.get_device_name(0))
PY
echo ">>> env ready at ${TRM_VENV}"
