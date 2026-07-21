#!/bin/bash
# Builds the venv OVERLAY on top of whatever PyTorch base is active when this
# runs (uenv view or NGC container). Identical logic for both backends:
# keep the base's GH200-optimized torch, add repo deps, compile adam-atan2 sm_90.
# Not called directly — invoked by 01_setup_env.sh inside the base.
set -euo pipefail
source "$(dirname "$0")/config.sh"

echo ">>> base python: $(python --version)"
echo ">>> base torch : $(python -c 'import torch;print(torch.__version__, torch.version.cuda)')"

rm -rf "${TRM_VENV}"
python -m venv --system-site-packages "${TRM_VENV}"
source "${TRM_VENV}/bin/activate"
pip install --upgrade pip wheel setuptools

# Repo deps EXCEPT torch (keep the base's) and adam-atan2 (built below).
grep -vE '^(torch|adam-atan2)\b' "${TRM_ROOT}/requirements.txt" > /tmp/trm_reqs.txt
pip install -r /tmp/trm_reqs.txt

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
echo ">>> venv ready at ${TRM_VENV}"
