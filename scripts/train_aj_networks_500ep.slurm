#!/bin/bash
#SBATCH --job-name=aj_train_500
#SBATCH --output=aj_train_500_%j.out
#SBATCH --error=aj_train_500_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#
# Train Forward AJ, Inverse AJ, and single-layer ReLU on MNIST for 500 epochs.
# Same setup as notebooks/aj_networks_train.ipynb.
#
# Optional env overrides:
#   DATA_ROOT   - MNIST root (default: $PWD/data)
#   TABLES_DIR  - forward tables dir (optional)
#   OUT_DIR     - checkpoint dir (default: $PWD/checkpoints)
#   DRY_RUN=1   - 2 epochs per model, small subset (use to check for crashes)
#
# Dry run (local, no submit):
#   DRY_RUN=1 bash scripts/train_aj_networks_500ep.slurm
# Submit:
#   sbatch scripts/train_aj_networks_500ep.slurm

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/users/hshunt/Abel-Jacobi-Networks}"
cd "$REPO_ROOT"

# Sherlock: load env via activation script (no conda)
if [[ -f "$HOME/activate_aj_env.sh" ]]; then
  source "$HOME/activate_aj_env.sh"
else
  module load math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312 2>/dev/null || true
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

DATA_ROOT="${DATA_ROOT:-$PWD/data}"
TABLES_DIR="${TABLES_DIR:-}"
OUT_DIR="${OUT_DIR:-$PWD/checkpoints}"
DRY_RUN="${DRY_RUN:-0}"

ARGS=(
  --data-root "$DATA_ROOT"
  --epochs 500
  --train-subset 20000
  --batch-size 64
  --out-dir "$OUT_DIR"
)
[[ -n "$TABLES_DIR" ]] && ARGS+=(--tables-dir "$TABLES_DIR")
[[ "$DRY_RUN" = "1" ]] && ARGS+=(--dry-run) && echo "DRY_RUN=1: 2 epochs per model"

echo "=== AJ networks 500-epoch training ==="
echo "Repo root   : $REPO_ROOT"
echo "Data root   : $DATA_ROOT"
echo "Tables dir  : ${TABLES_DIR:-<auto>}"
echo "Out dir     : $OUT_DIR"
echo "Dry run     : $DRY_RUN"
echo ""

python3 -u "$REPO_ROOT/scripts/train_aj_networks_500ep.py" "${ARGS[@]}"

echo ""
echo "Done."
