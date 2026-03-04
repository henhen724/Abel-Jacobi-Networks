#!/bin/bash
#SBATCH --job-name=aj_build_tables
#SBATCH --output=aj_build_tables_%j.out
#SBATCH --error=aj_build_tables_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#
# Build (or load) forward AJ lookup tables using src.util data-loader logic.
#
# Optional env overrides:
#   DATA_ROOT      - dataset root (default: $PWD/data)
#   TABLES_DIR     - optional explicit table directory (checked first)
#   GENUS          - default 30
#   GRID_SIZE      - default 96
#   AUTO_BUILD     - 1 (default) build tables if not found, 0 to fail if missing
#   FORCE_REBUILD  - 1 to rebuild even if files already exist
#   R_MIN,R_MAX,I_MIN,I_MAX,SEED
#
# Submit:
#   sbatch scripts/build_forward_tables.slurm
#   TABLES_DIR=/path/to/tables sbatch scripts/build_forward_tables.slurm

set -euo pipefail

REPO_ROOT="/home/users/hshunt/Abel-Jacobi-Networks"
cd "$REPO_ROOT"

# Sherlock environment (no conda): load modules via activation script.
source "$HOME/activate_aj_env.sh"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-$PWD/data}"
TABLES_DIR="${TABLES_DIR:-}"
GENUS="${GENUS:-30}"
GRID_SIZE="${GRID_SIZE:-96}"
AUTO_BUILD="${AUTO_BUILD:-1}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
R_MIN="${R_MIN:--6.0}"
R_MAX="${R_MAX:-6.0}"
I_MIN="${I_MIN:--6.0}"
I_MAX="${I_MAX:-6.0}"
SEED="${SEED:-123}"

export DATA_ROOT TABLES_DIR GENUS GRID_SIZE AUTO_BUILD FORCE_REBUILD
export R_MIN R_MAX I_MIN I_MAX SEED

echo "=== AJ Forward Table Build ==="
echo "Repo root     : $REPO_ROOT"
echo "Data root     : $DATA_ROOT"
echo "Tables dir    : ${TABLES_DIR:-<auto>}"
echo "Genus         : $GENUS"
echo "Grid size     : $GRID_SIZE"
echo "Auto build    : $AUTO_BUILD"
echo "Force rebuild : $FORCE_REBUILD"
echo ""

python3 - <<'PY'
import os
from pathlib import Path

import torch

from src import util as aj_util

data_root = Path(os.environ["DATA_ROOT"]).expanduser().resolve()
tables_dir_env = os.environ.get("TABLES_DIR", "").strip()
tables_dir = Path(tables_dir_env).expanduser().resolve() if tables_dir_env else None
genus = int(os.environ["GENUS"])
grid_size = int(os.environ["GRID_SIZE"])
auto_build = os.environ["AUTO_BUILD"] == "1"
force_rebuild = os.environ["FORCE_REBUILD"] == "1"
r_min = float(os.environ["R_MIN"])
r_max = float(os.environ["R_MAX"])
i_min = float(os.environ["I_MIN"])
i_max = float(os.environ["I_MAX"])
seed = int(os.environ["SEED"])

# Tables are generated/loaded on CPU.
device = torch.device("cpu")

if force_rebuild:
    target = tables_dir if tables_dir is not None else (data_root / f"AJ_Tables_g{genus}")
    print(f"FORCE_REBUILD=1: rebuilding tables in {target}")
    aj_util.build_and_save_forward_tables(
        out_dir=target,
        genus=genus,
        grid_size=grid_size,
        r_min=r_min,
        r_max=r_max,
        i_min=i_min,
        i_max=i_max,
        seed=seed,
    )
    tables, found_dir = aj_util.get_or_build_forward_tables(
        device=device,
        tables_dir=target,
        data_root=data_root,
        genus=genus,
        auto_build=False,
        grid_size=grid_size,
        r_min=r_min,
        r_max=r_max,
        i_min=i_min,
        i_max=i_max,
        seed=seed,
    )
else:
    tables, found_dir = aj_util.get_or_build_forward_tables(
        device=device,
        tables_dir=tables_dir,
        data_root=data_root,
        genus=genus,
        auto_build=auto_build,
        grid_size=grid_size,
        r_min=r_min,
        r_max=r_max,
        i_min=i_min,
        i_max=i_max,
        seed=seed,
    )

print(f"Using forward tables directory: {found_dir}")
print(f"Loaded genus: {tables['genus']}")
print(f"I_plus shape: {tuple(tables['I_plus'].shape)}")
print(f"Om_plus shape: {tuple(tables['Om_plus'].shape)}")
PY

echo ""
echo "Done."

