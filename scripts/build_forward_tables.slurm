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
#SBATCH --gres=gpu:1
#
# Build (or load) forward AJ lookup tables using GPU-parallel + checkpoint logic.
#
# Optional env overrides:
#   DATA_ROOT      - dataset root (default: $PWD/data)
#   TABLES_DIR     - optional explicit table directory (checked first)
#   GENUS          - default 30
#   GRID_SIZE      - default 96
#   AUTO_BUILD     - 1 (default) build tables if not found, 0 to fail if missing
#   FORCE_REBUILD  - 1 to rebuild even if files already exist
#   R_MIN,R_MAX,I_MIN,I_MAX,SEED
#   INTEGRATION_STEPS - trapezoid steps for integral approximation (default: 128)
#   CHUNK_POINTS      - number of grid points per GPU chunk (default: 2048)
#   SAVE_EVERY_CHUNKS - checkpoint frequency (default: 8)
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
export PYTHONUNBUFFERED=1

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
INTEGRATION_STEPS="${INTEGRATION_STEPS:-128}"
CHUNK_POINTS="${CHUNK_POINTS:-2048}"
SAVE_EVERY_CHUNKS="${SAVE_EVERY_CHUNKS:-8}"

export DATA_ROOT TABLES_DIR GENUS GRID_SIZE AUTO_BUILD FORCE_REBUILD
export R_MIN R_MAX I_MIN I_MAX SEED INTEGRATION_STEPS CHUNK_POINTS SAVE_EVERY_CHUNKS

echo "=== AJ Forward Table Build ==="
echo "Repo root     : $REPO_ROOT"
echo "Data root     : $DATA_ROOT"
echo "Tables dir    : ${TABLES_DIR:-<auto>}"
echo "Genus         : $GENUS"
echo "Grid size     : $GRID_SIZE"
echo "Auto build    : $AUTO_BUILD"
echo "Force rebuild : $FORCE_REBUILD"
echo "Int steps     : $INTEGRATION_STEPS"
echo "Chunk points  : $CHUNK_POINTS"
echo "Save every    : $SAVE_EVERY_CHUNKS chunks"
echo ""

python3 -u - <<'PY'
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
integration_steps = int(os.environ["INTEGRATION_STEPS"])
chunk_points = int(os.environ["CHUNK_POINTS"])
save_every_chunks = int(os.environ["SAVE_EVERY_CHUNKS"])

build_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Build device: {build_device}")
load_device = torch.device("cpu")

found = aj_util.find_forward_tables_dir(tables_dir=tables_dir, data_root=data_root, genus=genus)

if force_rebuild:
    target = tables_dir if tables_dir is not None else (data_root / f"AJ_Tables_g{genus}")
    print(f"FORCE_REBUILD=1: rebuilding tables in {target}")
    found = aj_util.build_and_save_forward_tables_gpu_checkpointed(
        out_dir=target,
        genus=genus,
        grid_size=grid_size,
        r_min=r_min,
        r_max=r_max,
        i_min=i_min,
        i_max=i_max,
        seed=seed,
        integration_steps=integration_steps,
        chunk_points=chunk_points,
        save_every_chunks=save_every_chunks,
        resume=True,
        device=build_device,
    )
else:
    if found is None and auto_build:
        target = tables_dir if tables_dir is not None else (data_root / f"AJ_Tables_g{genus}")
        print(f"No existing tables found; building in {target}")
        found = aj_util.build_and_save_forward_tables_gpu_checkpointed(
            out_dir=target,
            genus=genus,
            grid_size=grid_size,
            r_min=r_min,
            r_max=r_max,
            i_min=i_min,
            i_max=i_max,
            seed=seed,
            integration_steps=integration_steps,
            chunk_points=chunk_points,
            save_every_chunks=save_every_chunks,
            resume=True,
            device=build_device,
        )
    elif found is None:
        raise FileNotFoundError("Forward AJ tables not found and AUTO_BUILD=0.")

tables = aj_util.load_forward_tables(found, load_device, genus=genus)
if tables is None:
    raise RuntimeError(f"Failed to load tables from {found}")
found_dir = found

print(f"Using forward tables directory: {found_dir}")
print(f"Loaded genus: {tables['genus']}")
print(f"I_plus shape: {tuple(tables['I_plus'].shape)}")
print(f"Om_plus shape: {tuple(tables['Om_plus'].shape)}")
PY

echo ""
echo "Done."

