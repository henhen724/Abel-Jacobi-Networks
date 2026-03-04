#!/bin/bash
# Wrapper that loads the aj_env conda environment before starting the IPython kernel.
# Jupyter calls this script instead of calling python directly.
# Expects activate_aj_env.sh to be in HOME (copy from repo or symlink: ln -s REPO/activate_aj_env.sh ~/activate_aj_env.sh).

# Optional: use conda aj_env if available (source ~/activate_aj_env.sh first)
source ~/activate_aj_env.sh >/dev/null 2>&1 || true

# Use python3 so the kernel runs under Python 3 (module-loaded or conda), not system python (often 2.7)
exec python3 -m ipykernel_launcher "$@"
