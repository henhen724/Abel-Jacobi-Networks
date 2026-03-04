#!/bin/bash
# Activation script for the Abel–Jacobi (aj_env) conda environment.
# Used by the Jupyter kernel so notebooks run with the correct Python and deps.
#
# Usage:
#   source /path/to/activate_aj_env.sh
# Or after copying to home:
#   source ~/activate_aj_env.sh

# Find conda and initialize (common locations)
if [[ -z "$CONDA_EXE" ]]; then
  for _conda in "$HOME/miniconda3/etc/profile.d/conda.sh" \
                "$HOME/anaconda3/etc/profile.d/conda.sh" \
                "$HOME/miniforge3/etc/profile.d/conda.sh"; do
    if [[ -f "$_conda" ]]; then
      source "$_conda"
      break
    fi
  done
fi

# Activate project env (aj_env from rules, or aj from environment.yml)
if conda activate aj_env 2>/dev/null; then
  : # aj_env found
elif conda activate aj 2>/dev/null; then
  : # aj found
else
  echo "activate_aj_env.sh: could not activate conda env 'aj_env' or 'aj'" >&2
  return 1
fi
