#!/usr/bin/env python3
"""
Train Forward AJ, Inverse AJ, and single-layer ReLU on MNIST for 500 epochs.

Same setup as notebooks/aj_networks_train.ipynb: genus-30 forward tables,
genus-30 inverse (toy Ω/K), MNIST with 20k train subset, batch 64.
Single-layer ReLU: Linear(784, 128) -> ReLU -> Linear(128, 10).

Usage:
  python scripts/train_aj_networks_500ep.py --data-root ./data [--epochs 500]
  DRY_RUN=1 python scripts/train_aj_networks_500ep.py --data-root ./data  # 2 epochs per model
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from aj.classical import InverseAbelJacobiNetwork
from aj.classical.grid_activation import AJGridActivationNorm

from src import util as aj_util


def mnist_cnn_trunk() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((1, 1)),
    )


class AJMNIST_Anchored(nn.Module):
    def __init__(self, genus, I_plus, Om_plus, grid_r, grid_i, branch_pts, anchors_xy, mu, sigma, embed_dim=8):
        super().__init__()
        self.genus = genus
        self.conv = mnist_cnn_trunk()
        self.embed = nn.Parameter(torch.empty(genus, embed_dim))
        nn.init.uniform_(self.embed, -2.0, 2.0)
        self.point_head = nn.Linear(64 + embed_dim, 3, bias=False)
        nn.init.xavier_uniform_(self.point_head.weight)
        self.point_bias = nn.Parameter(torch.zeros(genus, 3))
        self.aj = AJGridActivationNorm(I_plus, Om_plus, grid_r, grid_i, branch_pts, mu, sigma)
        self.classifier = nn.Linear(2 * genus, 10)
        rmin, rmax = float(grid_r.min()), float(grid_r.max())
        imin, imax = float(grid_i.min()), float(grid_i.max())

        def logit(p):
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return float(np.log(p / (1 - p)))

        with torch.no_grad():
            for i in range(genus):
                x0, y0 = float(anchors_xy[i, 0]), float(anchors_xy[i, 1])
                px = (x0 - rmin) / (rmax - rmin)
                py = (y0 - imin) / (imax - imin)
                self.point_bias[i, 0] = logit(px)
                self.point_bias[i, 1] = logit(py)
                target_sign = 0.8 if (i % 2 == 0) else -0.8
                self.point_bias[i, 2] = float(torch.atanh(torch.tensor(target_sign)))

    def forward(self, x, return_aux=False):
        B = x.size(0)
        h = self.conv(x).view(B, -1)
        h_exp = h.unsqueeze(1).expand(-1, self.genus, -1)
        emb = self.embed.unsqueeze(0).expand(B, -1, -1)
        out = self.point_head(torch.cat([h_exp, emb], dim=2)) + self.point_bias.unsqueeze(0)
        raw_xy, sheet_logits = out[..., :2], out[..., 2]
        coords, aux = self.aj(raw_xy, sheet_logits, return_aux=True)
        logits = self.classifier(coords)
        if return_aux:
            return logits, aux
        return logits


class TorusFeatures(nn.Module):
    def __init__(self, dim: int, K: int = 2, freqs=None, learnable: bool = False):
        super().__init__()
        if freqs is None:
            freqs = torch.tensor([0.5, 1.0], dtype=torch.float32)[:K]
        else:
            freqs = torch.as_tensor(freqs, dtype=torch.float32)[:K]
        if learnable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)
        self.dim, self.K = dim, len(freqs)

    def forward(self, u):
        B, D = u.shape
        f = self.freqs.view(1, 1, -1).to(u.device)
        ang = u.unsqueeze(-1) * f
        return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1).view(B, D * 2 * self.K)


class AJMNIST_AxisPeriodic(nn.Module):
    def __init__(
        self, genus, I_plus, Om_plus, grid_r, grid_i, branch_pts, anchors_xy, mu, sigma,
        embed_dim=4, K=2, learnable_freqs=False,
    ):
        super().__init__()
        self.base = AJMNIST_Anchored(
            genus, I_plus, Om_plus, grid_r, grid_i, branch_pts, anchors_xy, mu, sigma,
            embed_dim=embed_dim,
        )
        D = 2 * genus
        self.torus = TorusFeatures(D, K=K, learnable=learnable_freqs)
        self.classifier = nn.Linear(D * 2 * K, 10)

    def forward(self, x, return_aux=False):
        B = x.size(0)
        h = self.base.conv(x).view(B, -1)
        h_exp = h.unsqueeze(1).expand(-1, self.base.genus, -1)
        emb = self.base.embed.unsqueeze(0).expand(B, -1, -1)
        out = self.base.point_head(torch.cat([h_exp, emb], dim=2)) + self.base.point_bias.unsqueeze(0)
        raw_xy, sheet_logits = out[..., :2], out[..., 2]
        coords, aux = self.base.aj(raw_xy, sheet_logits, return_aux=True)
        feats = self.torus(coords)
        logits = self.classifier(feats)
        if return_aux:
            return logits, aux
        return logits


def _branch_points_from_tables(tables):
    bp = tables["branch_pts_t"]
    if torch.is_complex(bp):
        return torch.stack([bp.real.float(), bp.imag.float()], dim=1)
    bp = bp.detach().cpu().numpy()
    if np.iscomplexobj(bp):
        return torch.tensor(np.stack([bp.real, bp.imag], axis=1), dtype=torch.float32)
    return torch.tensor(bp, dtype=torch.float32)


def _toy_period_matrix(genus: int):
    Omega = (1.0 + 2.0j) * np.eye(genus, dtype=np.complex128)
    Omega += 0.1 * (np.ones((genus, genus)) - np.eye(genus))
    Omega = (Omega + Omega.T) / 2
    return Omega, np.zeros(genus, dtype=np.complex128)


def _toy_log_sigma(v: np.ndarray) -> complex:
    v = np.asarray(v, dtype=np.complex128).ravel()
    if v.size == 1:
        return -(v[0] ** 3) / 6.0
    return -(v[0] ** 3) / 6.0 - 0.5 * v[0] * (v[1] ** 2)


def build_inverse_aj_net(tables, device, genus: int):
    Omega_init, K_init = _toy_period_matrix(genus)
    return InverseAbelJacobiNetwork(
        genus=genus,
        branch_points=_branch_points_from_tables(tables),
        base_point=(-8.0, -8.0),
        init_divisor_points=torch.zeros(genus, 2, dtype=torch.float32),
        use_kleinian_p=True,
        log_sigma_fun=_toy_log_sigma,
        Omega_init=Omega_init,
        K_init=K_init,
    ).to(device)


class MNISTInversePNet(nn.Module):
    def __init__(self, inv_net: InverseAbelJacobiNetwork):
        super().__init__()
        self.genus = inv_net.genus
        self.conv = mnist_cnn_trunk()
        self.to_u = nn.Linear(64, 2 * self.genus)
        self.inv_net = inv_net
        self.classifier = nn.Linear(2 * self.genus, 10)

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        u = self.to_u(h).view(-1, self.genus, 2)
        _ = self.inv_net(u)
        return self.classifier(u.reshape(x.size(0), 2 * self.genus))


def parse_args():
    p = argparse.ArgumentParser(description="Train AJ and ReLU networks on MNIST (500 ep).")
    p.add_argument("--data-root", type=Path, default=Path("data"), help="MNIST root")
    p.add_argument("--tables-dir", type=Path, default=None, help="Forward tables dir (optional)")
    p.add_argument("--epochs", type=int, default=None, help="Epochs per model (default 500, or 2 if DRY_RUN=1)")
    p.add_argument("--train-subset", type=int, default=20_000, help="Train subset size (0 = full)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-dir", type=Path, default=None, help="Checkpoint dir (default REPO_ROOT/checkpoints)")
    p.add_argument("--dry-run", action="store_true", help="Short run: 2 epochs, small subset (overrides --epochs)")
    return p.parse_args()


def main():
    args = parse_args()
    dry = args.dry_run or (os.environ.get("DRY_RUN", "0") == "1")
    if dry:
        args.epochs = 2
        args.train_subset = min(args.train_subset, 2000)
        print("DRY_RUN: 2 epochs per model, train subset capped at 2000")
    if args.epochs is None:
        args.epochs = 500
    if args.out_dir is None:
        args.out_dir = REPO_ROOT / "checkpoints"
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data_root = args.data_root.expanduser().resolve()
    aj_util.ensure_mnist_available(str(data_root))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}, train subset: {args.train_subset}, batch: {args.batch_size}")
    print(f"Checkpoints: {args.out_dir}")
    print()

    # Shared data loaders (same transform as notebook)
    train_loader_full, test_loader = aj_util.get_mnist_loaders(
        root=str(data_root), test_batch_size=256, num_workers=0
    )
    train_ds = train_loader_full.dataset
    if args.train_subset > 0:
        n = min(args.train_subset, len(train_ds))
        train_ds = torch.utils.data.Subset(train_ds, list(range(n)))
        print(f"Train subset: {n} samples")
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    ce = nn.CrossEntropyLoss()

    # ---------- 1. Forward AJ ----------
    print("=== Forward AJ (genus 30) ===")
    tables_dir = args.tables_dir.resolve() if args.tables_dir else None
    tables, tables_found = aj_util.get_or_build_forward_tables(
        device=device,
        tables_dir=tables_dir,
        data_root=data_root,
        genus=30,
        auto_build=True,
        grid_size=96,
    )
    print(f"Tables from: {tables_found}")
    # Use mu_t/sigma_t from load_forward_tables (same as AJ_training_genus30.ipynb).
    model_fwd = AJMNIST_AxisPeriodic(
        genus=tables["genus"],
        I_plus=tables["I_plus"],
        Om_plus=tables["Om_plus"],
        grid_r=tables["grid_r"],
        grid_i=tables["grid_i"],
        branch_pts=tables["branch_pts_t"],
        anchors_xy=tables["anchors_xy_t"],
        mu=tables["mu_t"],
        sigma=tables["sigma_t"],
        embed_dim=4,
        K=2,
        learnable_freqs=False,
    ).to(device)
    opt_fwd = aj_util.forward_aj_adamw(model_fwd, lr_base=3e-4, lr_fast=1e-3, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    fwd_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = aj_util.train_forward_aj_epoch_amp(
            model_fwd,
            train_loader,
            opt_fwd,
            device,
            scaler=fwd_scaler,
            use_amp=use_amp,
        )
        te_loss, te_acc = aj_util.eval_epoch(model_fwd, test_loader, device)
        if ep % 50 == 0 or ep == 1 or ep == args.epochs:
            print(f"[Forward AJ] Epoch {ep:4d} | train {tr_loss:.4f}/{tr_acc:.2f}% | test {te_loss:.4f}/{te_acc:.2f}%")

    ckpt_fwd = args.out_dir / "aj_forward_mnist_genus30_500ep.pt"
    torch.save({"state_dict": model_fwd.state_dict()}, ckpt_fwd)
    print(f"Saved {ckpt_fwd}\n")

    # ---------- 2. Inverse AJ (genus 30) ----------
    print("=== Inverse AJ (genus 30) ===")
    inv_core = build_inverse_aj_net(tables, device, genus=30)
    model_inv = MNISTInversePNet(inv_core).to(device)
    opt_inv = torch.optim.AdamW(model_inv.parameters(), lr=3e-4, weight_decay=1e-4)

    for ep in range(1, args.epochs + 1):
        model_inv.train()
        tot, correct, count = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt_inv.zero_grad(set_to_none=True)
            logits = model_inv(x)
            loss = ce(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model_inv.parameters(), 1.0)
            opt_inv.step()
            tot += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            count += x.size(0)
        tr_loss, tr_acc = tot / count, 100.0 * correct / count
        te_loss, te_acc = aj_util.eval_epoch(model_inv, test_loader, device)
        if ep % 50 == 0 or ep == 1 or ep == args.epochs:
            print(f"[Inverse AJ g=30] Epoch {ep:4d} | train {tr_loss:.4f}/{tr_acc:.2f}% | test {te_loss:.4f}/{te_acc:.2f}%")

    ckpt_inv = args.out_dir / "mnist_inverse_p_g30_500ep.pt"
    torch.save({"state_dict": model_inv.state_dict()}, ckpt_inv)
    print(f"Saved {ckpt_inv}\n")

    # ---------- 3. Single-layer ReLU ----------
    print("=== Single-layer ReLU (784 -> 128 -> 10) ===")
    model_relu = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 10),
    ).to(device)
    opt_relu = torch.optim.AdamW(model_relu.parameters(), lr=3e-4, weight_decay=1e-4)

    for ep in range(1, args.epochs + 1):
        model_relu.train()
        tot, correct, count = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt_relu.zero_grad(set_to_none=True)
            logits = model_relu(x)
            loss = ce(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model_relu.parameters(), 1.0)
            opt_relu.step()
            tot += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            count += x.size(0)
        tr_loss, tr_acc = tot / count, 100.0 * correct / count
        te_loss, te_acc = aj_util.eval_epoch(model_relu, test_loader, device)
        if ep % 50 == 0 or ep == 1 or ep == args.epochs:
            print(f"[ReLU 1-layer] Epoch {ep:4d} | train {tr_loss:.4f}/{tr_acc:.2f}% | test {te_loss:.4f}/{te_acc:.2f}%")

    ckpt_relu = args.out_dir / "relu1layer_mnist_500ep.pt"
    torch.save({"state_dict": model_relu.state_dict()}, ckpt_relu)
    print(f"Saved {ckpt_relu}\n")

    print("Done.")


if __name__ == "__main__":
    main()
