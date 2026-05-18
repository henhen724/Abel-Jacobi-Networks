"""Grid lookup Abel–Jacobi activation (no image/CNN trunk)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pack_complex_table(table_gHW: torch.Tensor) -> torch.Tensor:
    """Pack complex (g, H, W) table as (1, 2g, H, W) real tensor for grid_sample."""
    re, im = table_gHW.real, table_gHW.imag
    return torch.cat([re, im], dim=0).unsqueeze(0).contiguous()


class AJGridActivationNorm(nn.Module):
    """
    Bilinear lookup of precomputed AJ integrals on a grid, with per-point sheet sign
    and channel-wise standardization. Sums signed contributions to a (B, 2g) coordinate vector.
    """

    def __init__(
        self,
        I_plus: torch.Tensor,
        Om_plus: torch.Tensor,
        grid_r: torch.Tensor,
        grid_i: torch.Tensor,
        branch_pts: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ):
        super().__init__()
        self.g = I_plus.shape[0]
        self.register_buffer("I_plus", pack_complex_table(I_plus))
        self.register_buffer("Om_plus", pack_complex_table(Om_plus))
        self.register_buffer("mu", mu.view(1, 1, -1))
        self.register_buffer("sigma", sigma.view(1, 1, -1))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("r_min", torch.tensor(float(grid_r.min())))
        self.register_buffer("r_max", torch.tensor(float(grid_r.max())))
        self.register_buffer("i_min", torch.tensor(float(grid_i.min())))
        self.register_buffer("i_max", torch.tensor(float(grid_i.max())))
        self.register_buffer("bp_real", branch_pts.real.float())
        self.register_buffer("bp_imag", branch_pts.imag.float())

    def _map_raw_to_bounds(self, raw_xy: torch.Tensor):
        xr = self.r_min + (self.r_max - self.r_min) * torch.sigmoid(raw_xy[..., 0])
        yi = self.i_min + (self.i_max - self.i_min) * torch.sigmoid(raw_xy[..., 1])
        return xr, yi

    def _norm_to_grid(self, xr: torch.Tensor, yi: torch.Tensor):
        gx = 2.0 * (xr - self.r_min) / (self.r_max - self.r_min) - 1.0
        gy = 2.0 * (yi - self.i_min) / (self.i_max - self.i_min) - 1.0
        return gx, gy

    def forward(
        self,
        raw_xy: torch.Tensor,
        sheet_logits: torch.Tensor,
        return_aux: bool = True,
    ):
        B, g, _ = raw_xy.shape
        if g != self.g:
            raise ValueError(f"expected g={self.g}, got {g}")
        xr, yi = self._map_raw_to_bounds(raw_xy)
        gx, gy = self._norm_to_grid(xr, yi)
        grid = torch.stack([gx, gy], dim=-1).view(B * g, 1, 1, 2)
        I = F.grid_sample(
            self.I_plus.expand(B * g, -1, -1, -1),
            grid,
            mode="bilinear",
            align_corners=True,
        ).view(B, g, -1)
        I_std = (I - self.mu) / self.sigma
        sign = torch.tanh(sheet_logits).unsqueeze(-1)
        contrib = sign * I_std
        coords = self.gamma * contrib.sum(dim=1)
        aux = None
        if return_aux:
            margin = 0.95
            bpen = (
                (gx.abs() - margin).clamp_min(0) ** 2
                + (gy.abs() - margin).clamp_min(0) ** 2
            ).mean()
            dx = xr.unsqueeze(-1) - self.bp_real
            dy = yi.unsqueeze(-1) - self.bp_imag
            d2 = dx * dx + dy * dy
            tau = 0.07
            rpen = torch.exp(-d2 / (2 * tau * tau)).mean()
            aux = {"bound_penalty": bpen, "branch_penalty": rpen}
        return coords, aux
