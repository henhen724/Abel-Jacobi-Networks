#!/bin/bash
# Install the Jupyter kernel for aj_env (same pattern as ~/diffusion_venv / diffusion_env).
# Run from repo root:  bash scripts/install_aj_env_kernel.sh
#
# This will:
#   1. Copy activate_aj_env.sh to ~/activate_aj_env.sh (if not already there)
#   2. Install the kernel spec to ~/.local/share/jupyter/kernels/aj_env/

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KERNEL_DIR="$HOME/.local/share/jupyter/kernels/aj_env"
ACTIVATE_SH="$HOME/activate_aj_env.sh"

mkdir -p "$KERNEL_DIR"

# Activation script in home (so launch_kernel.sh can source ~/activate_aj_env.sh)
if [[ ! -f "$ACTIVATE_SH" ]]; then
  cp "$REPO_ROOT/activate_aj_env.sh" "$ACTIVATE_SH"
  chmod +x "$ACTIVATE_SH"
  echo "Installed $ACTIVATE_SH"
else
  echo "Already exists: $ACTIVATE_SH (not overwritten)"
fi

# Kernel spec: launch script + kernel.json with correct path
cp "$REPO_ROOT/jupyter_kernels/aj_env/launch_kernel.sh" "$KERNEL_DIR/launch_kernel.sh"
chmod +x "$KERNEL_DIR/launch_kernel.sh"

cat > "$KERNEL_DIR/kernel.json" << EOF
{
  "argv": [
    "$KERNEL_DIR/launch_kernel.sh",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python (aj_env)",
  "language": "python"
}
EOF

echo "Kernel installed to $KERNEL_DIR"
echo "In Jupyter, choose kernel: Python (aj_env)"
