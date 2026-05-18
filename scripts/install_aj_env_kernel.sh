#!/usr/bin/env bash
# Install the Jupyter kernel for aj_env (same pattern as ~/activate_diffusion.sh +
# ~/diffusion_venv_py312: see from_diffusion_agent.md in this repo).
# Run from repo root:  bash scripts/install_aj_env_kernel.sh
#
# This will:
#   1. Copy activate_aj_env.sh to ~/activate_aj_env.sh (always refreshed from repo)
#   2. If needed, create ~/aj_env_venv with --system-site-packages and pip-install
#      ipykernel + notebook deps + editable aj (ipykernel is not on the module Python)
#   3. Install the kernel spec to ~/.local/share/jupyter/kernels/aj_env/

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KERNEL_DIR="${HOME}/.local/share/jupyter/kernels/aj_env"
ACTIVATE_SH="${HOME}/activate_aj_env.sh"

mkdir -p "${KERNEL_DIR}"

# Always sync activation script from repo (home copy was often stale / hand-edited).
cp -f "${REPO_ROOT}/activate_aj_env.sh" "${ACTIVATE_SH}"
chmod +x "${ACTIVATE_SH}"
echo "Installed ${ACTIVATE_SH} (from repo)"

# --- Overlay venv for Jupyter / pip-only packages (diffusion-agent pattern) ----
_aj_need_venv_bootstrap() {
  if [[ ! -x "${HOME}/aj_env_venv/bin/python" ]]; then
    return 0
  fi
  if ! "${HOME}/aj_env_venv/bin/python" -c "import ipykernel" 2>/dev/null; then
    return 0
  fi
  return 1
}

if [[ "${AJ_SKIP_VENV_BOOTSTRAP:-0}" != "1" ]] && _aj_need_venv_bootstrap; then
  echo "Bootstrapping ~/aj_env_venv (--system-site-packages); see from_diffusion_agent.md"
  # shellcheck disable=SC1090
  bash -c "set -euo pipefail
    source \"${ACTIVATE_SH}\"
    if [[ ! -d \"${HOME}/aj_env_venv\" ]]; then
      python3 -m venv --system-site-packages \"${HOME}/aj_env_venv\"
    fi
    source \"${HOME}/aj_env_venv/bin/activate\"
    pip install -U pip wheel
    pip install ipykernel matplotlib mpmath tqdm networkx
    pip install -e \"${REPO_ROOT}\""
  echo "Finished ~/aj_env_venv bootstrap."
fi

# Kernel spec: launch script + kernel.json with correct path
cp -f "${REPO_ROOT}/jupyter_kernels/aj_env/launch_kernel.sh" "${KERNEL_DIR}/launch_kernel.sh"
chmod +x "${KERNEL_DIR}/launch_kernel.sh"

cat > "${KERNEL_DIR}/kernel.json" << EOF
{
  "argv": [
    "${KERNEL_DIR}/launch_kernel.sh",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python (aj_env)",
  "language": "python"
}
EOF

echo "Kernel installed to ${KERNEL_DIR}"
echo "In Jupyter, choose kernel: Python (aj_env)"
