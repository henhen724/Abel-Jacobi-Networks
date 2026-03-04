#!/bin/bash
#SBATCH --job-name=aj_mnist_test
#SBATCH --output=aj_mnist_test_%j.out
#SBATCH --error=aj_mnist_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#
# Test accuracy of classical Abel-Jacobi networks (forward and inverse) on MNIST.
# Same MNIST classification test as AJ_training_genus30.ipynb.
#
# Optional env overrides (set before sbatch or in your environment):
#   TABLES_DIR     - directory with aj_integrals_genus30.pt and aj_omegas_genus30.pt (forward test)
#   FORWARD_CKPT   - path to forward model checkpoint
#   INVERSE_CKPT   - path to inverse model checkpoint
#   DATA_ROOT      - MNIST data root (default: $PWD/data)
#   DRY_RUN        - if set to 1, use --test-subset 256 for a quick test (do this before full submit)
#
# Dry run (run before submitting the real job):
#   module load math python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 py-numpy/1.26.3_py312
#   python3 scripts/aj_mnist_test_accuracy.py --data-root ./data --test-subset 256
#   Then: sbatch scripts/run_aj_mnist_test.slurm   (or DRY_RUN=1 sbatch ... for a quick job)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="/home/users/hshunt/Abel-Jacobi-Networks"
cd "$REPO_ROOT"

# Use module loader (no conda on this server). Ensures Python 3.12 + PyTorch + torchvision + numpy.
# Requires: pip install --user tqdm; pip install --user -e .  (once per user)
module load math
module load python/3.12.1
module load py-pytorch/2.4.1_py312
module load py-torchvision/0.19.1_py312
module load py-numpy/1.26.3_py312

DATA_ROOT="${DATA_ROOT:-$PWD/data}"
TABLES_DIR="${TABLES_DIR:-}"
FORWARD_CKPT="${FORWARD_CKPT:-}"
INVERSE_CKPT="${INVERSE_CKPT:-}"

ARGS=(
  --data-root "$DATA_ROOT"
  --test-batch-size 256
)
[[ -n "$TABLES_DIR" ]] && ARGS+=(--tables-dir "$TABLES_DIR")
[[ -n "$FORWARD_CKPT" ]] && ARGS+=(--forward-checkpoint "$FORWARD_CKPT")
[[ -n "$INVERSE_CKPT" ]] && ARGS+=(--inverse-checkpoint "$INVERSE_CKPT")
[[ "${DRY_RUN:-0}" = "1" ]] && ARGS+=(--test-subset 256) && echo "DRY_RUN=1: using test subset 256"

echo "=== AJ MNIST test accuracy (forward + inverse) ==="
echo "Data root: $DATA_ROOT"
echo "Tables dir: ${TABLES_DIR:-<none - forward test skipped>}"
echo "Forward ckpt: ${FORWARD_CKPT:-<none>}"
echo "Inverse ckpt: ${INVERSE_CKPT:-<none>}"
echo ""

python3 "$REPO_ROOT/scripts/aj_mnist_test_accuracy.py" "${ARGS[@]}"

echo ""
echo "Done."
