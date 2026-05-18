#!/usr/bin/env python3
"""
Check that spiral forward table loading works from the command line.
Uses same get_or_build_forward_tables + tables_for_spiral_forward as the notebook.

Run from repo root with the aj_env conda environment active:
  conda activate aj_env
  python scripts/check_spiral_forward_tables.py

Or:  conda run -n aj_env python scripts/check_spiral_forward_tables.py

If precomputed tables exist in data/AJ_Tables_g{genus}, they are loaded; otherwise
tables are built (slow) and saved. Uses genus=2 (WIDTH=4) by default.
"""
from pathlib import Path
import sys
import numpy as np

# Repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    from src import util as aj_util
except ImportError:
    # Run from repo root with PYTHONPATH=src
    import util as aj_util

import torch
import torch.nn as nn
import torch.nn.functional as F

# Config (match notebook: WIDTH=4 -> genus 2)
WIDTH = 4
GENUS_SPIRAL = WIDTH // 2
CUTS_SEED = 123
CUTS_RADIUS = 4.0
CUTS_JITTER = 0.25
GRID_BOUND = 4.0

REPO_ROOT = aj_util.get_repo_root(Path.cwd())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pack_complex_table(table_gHW):
    if torch.is_tensor(table_gHW):
        re = table_gHW.real.float()
        im = table_gHW.imag.float()
    else:
        re = torch.from_numpy(table_gHW.real).float()
        im = torch.from_numpy(table_gHW.imag).float()
    return torch.cat([re, im], dim=0).unsqueeze(0).contiguous()


class ForwardAJ2D(nn.Module):
    def __init__(self, tables, num_classes=2, scale=6.0):
        super().__init__()
        g = tables["genus"]
        I_plus = tables["I_plus"]
        Om_plus = tables["omega_plus"]
        grid_r = tables["grid_r"]
        grid_i = tables["grid_i"]
        branch_pts = tables["branch_pts"]
        mu = torch.from_numpy(tables["mu"]).float() if not torch.is_tensor(tables["mu"]) else tables["mu"].float()
        sigma = torch.from_numpy(tables["sigma"]).float() if not torch.is_tensor(tables["sigma"]) else tables["sigma"].float()
        self.g = g
        self.scale = scale
        self.register_buffer("I_plus", _pack_complex_table(I_plus))
        self.register_buffer("Om_plus", _pack_complex_table(Om_plus))
        self.register_buffer("mu", mu.view(1, 1, -1))
        self.register_buffer("sigma", sigma.view(1, 1, -1))
        grid_r = np.asarray(grid_r)
        grid_i = np.asarray(grid_i)
        self.register_buffer("r_min", torch.tensor(float(np.min(grid_r))))
        self.register_buffer("r_max", torch.tensor(float(np.max(grid_r))))
        self.register_buffer("i_min", torch.tensor(float(np.min(grid_i))))
        self.register_buffer("i_max", torch.tensor(float(np.max(grid_i))))
        bp = np.asarray(branch_pts)
        bp_real = bp.real if bp.dtype.kind == "c" else bp[:, 0]
        bp_imag = bp.imag if bp.dtype.kind == "c" else bp[:, 1]
        self.register_buffer("bp_real", torch.from_numpy(bp_real).float())
        self.register_buffer("bp_imag", torch.from_numpy(bp_imag).float())
        anchor = (np.min(grid_r) + np.max(grid_r)) / 2 + 1j * (np.min(grid_i) + np.max(grid_i)) / 2
        self.register_buffer("anchor_real", torch.tensor(float(anchor.real)))
        self.register_buffer("anchor_imag", torch.tensor(float(anchor.imag)))
        self.classifier = nn.Linear(2 * g, num_classes)

    def forward(self, xy):
        B = xy.size(0)
        x, y = xy[:, 0], xy[:, 1]
        xr = self.r_min + (self.r_max - self.r_min) * (x / self.scale + 1.0) / 2.0
        yi = self.i_min + (self.i_max - self.i_min) * (y / self.scale + 1.0) / 2.0
        gx1 = 2.0 * (xr - self.r_min) / (self.r_max - self.r_min) - 1.0
        gy1 = 2.0 * (yi - self.i_min) / (self.i_max - self.i_min) - 1.0
        grid1 = torch.stack([gx1, gy1], dim=-1).view(B, 1, 1, 2)
        xr2 = self.anchor_real.expand(B)
        yi2 = self.anchor_imag.expand(B)
        gx2 = 2.0 * (xr2 - self.r_min) / (self.r_max - self.r_min) - 1.0
        gy2 = 2.0 * (yi2 - self.i_min) / (self.i_max - self.i_min) - 1.0
        grid2 = torch.stack([gx2, gy2], dim=-1).view(B, 1, 1, 2)
        I1 = F.grid_sample(self.I_plus.expand(B, -1, -1, -1), grid1, mode="bilinear", align_corners=True).view(B, -1)
        I2 = F.grid_sample(self.I_plus.expand(B, -1, -1, -1), grid2, mode="bilinear", align_corners=True).view(B, -1)
        I_std1 = (I1 - self.mu.view(-1)) / self.sigma.view(-1)
        I_std2 = (I2 - self.mu.view(-1)) / self.sigma.view(-1)
        coords = I_std1 + I_std2
        return self.classifier(coords)


def main():
    print("Loading forward tables (same args as spiral notebook, no tables_subdir)...")
    tables, tables_found = aj_util.get_or_build_forward_tables(
        device,
        data_root=REPO_ROOT / "data",
        genus=GENUS_SPIRAL,
        grid_size=32,
        r_min=-GRID_BOUND,
        r_max=GRID_BOUND,
        i_min=-GRID_BOUND,
        i_max=GRID_BOUND,
        base_point=complex(-5.0, -5.0),
        seed=CUTS_SEED,
        radius=CUTS_RADIUS,
        jitter=CUTS_JITTER,
        auto_build=True,
    )
    print(f"Forward tables: {tables_found}")
    tables_2d = aj_util.tables_for_spiral_forward(tables)
    print("Building ForwardAJ2D and running one forward pass...")
    model = ForwardAJ2D(tables_2d, num_classes=2, scale=14.0).to(device)
    x = torch.randn(4, 2, device=device) * 5.0
    out = model(x)
    assert out.shape == (4, 2), f"expected (4, 2), got {out.shape}"
    print("OK: spiral forward table loading and ForwardAJ2D forward pass work.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
