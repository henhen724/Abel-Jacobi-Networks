"""
Train InverseAbelJacobiNetwork (P-function mode) on MNIST.

This mirrors the AJ_training_genus30.ipynb benchmark structure:
- MNIST train/test splits
- conv feature trunk
- classifier head

For practicality, defaults are small/fast. Increase --genus/--epochs/subset sizes
for larger runs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from aj.classical.inverse_network import InverseAbelJacobiNetwork


def toy_log_sigma(v: np.ndarray) -> complex:
    """
    Construct log sigma so P_{i,0}(u) approximately returns u_i:
      logσ = -(u0^3)/6 - (u0/2) * sum_{i>0} u_i^2
      => -∂00 logσ = u0, -∂i0 logσ = u_i (i>0)
    """
    v = np.asarray(v, dtype=np.complex128)
    if v.size == 0:
        return 0.0 + 0.0j
    out = -(v[0] ** 3) / 6.0
    if v.size > 1:
        out -= 0.5 * v[0] * np.sum(v[1:] ** 2)
    return out


def make_branch_points(genus: int) -> torch.Tensor:
    pts = []
    for j in range(genus + 1):
        x = -1.5 + 3.0 * j / max(1, genus)
        pts.append([x, -0.5])
        pts.append([x, 0.5])
    return torch.tensor(pts[: 2 * genus + 2], dtype=torch.float32)


class MNISTInversePNet(nn.Module):
    def __init__(self, genus: int, device: torch.device):
        super().__init__()
        self.genus = genus
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.to_u = nn.Linear(64, 2 * genus)
        self.classifier = nn.Linear(genus, 10)

        branch_points = make_branch_points(genus).to(device)
        Omega_init = 1j * np.eye(genus, dtype=np.complex128)
        K_init = np.zeros(genus, dtype=np.complex128)
        init_div = torch.zeros(genus, 2, dtype=torch.float32, device=device)
        self.inverse_net = InverseAbelJacobiNetwork(
            genus=genus,
            branch_points=branch_points,
            base_point=(-3.0, -3.0),
            init_divisor_points=init_div,
            use_kleinian_p=True,
            log_sigma_fun=toy_log_sigma,
            Omega_init=Omega_init,
            K_init=K_init,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        h = self.conv(x).view(b, -1)
        u = self.to_u(h).view(b, self.genus, 2)
        coeffs = self.inverse_net(u)
        return self.classifier(coeffs)


def _grad_l2_norm(model: nn.Module) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        sq += float((g * g).sum().item())
    return float(np.sqrt(max(sq, 0.0)))


def _param_l2_norm(param: torch.Tensor) -> float:
    return float(torch.linalg.norm(param.detach()).item())


def train_epoch(model, loader, opt, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    tot, correct, n = 0.0, 0, 0
    grad_norm_sum = 0.0
    grad_norm_max = 0.0
    num_steps = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        grad_norm = _grad_l2_norm(model)
        grad_norm_sum += grad_norm
        grad_norm_max = max(grad_norm_max, grad_norm)
        num_steps += 1
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    grad_mean = grad_norm_sum / max(1, num_steps)
    return tot / n, 100.0 * correct / n, grad_mean, grad_norm_max


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    tot, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        tot += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return tot / n, 100.0 * correct / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--genus", type=int, default=2)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-subset", type=int, default=0, help="0 means full MNIST train")
    p.add_argument("--test-subset", type=int, default=0, help="0 means full MNIST test")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--output-dir", type=str, default="./checkpoints/inverse_p_mnist")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train_ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    if args.train_subset > 0:
        train_ds = torch.utils.data.Subset(train_ds, list(range(min(args.train_subset, len(train_ds)))))
    if args.test_subset > 0:
        test_ds = torch.utils.data.Subset(test_ds, list(range(min(args.test_subset, len(test_ds)))))

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=max(args.batch_size, 128),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = MNISTInversePNet(args.genus, device).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_name = f"g{args.genus}_e{args.epochs}_bs{args.batch_size}_{int(time.time())}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    history_path = os.path.join(run_dir, "metrics.jsonl")
    summary_path = os.path.join(run_dir, "summary.json")

    print(f"Training MNIST Inverse-P model: genus={args.genus}, epochs={args.epochs}, device={device}")
    print(f"Checkpoint/stat dir: {run_dir}")
    best_acc = -1.0
    best_epoch = -1
    history = []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, grad_mean, grad_max = train_epoch(model, train_loader, opt, device)
        te_loss, te_acc = eval_epoch(model, test_loader, device)
        dt = time.time() - t0
        lr = float(opt.param_groups[0]["lr"])
        coeff_norm = _param_l2_norm(model.inverse_net.coeffs)
        divisor_norm = _param_l2_norm(model.inverse_net.divisor_points)
        row = {
            "epoch": ep,
            "train_loss": float(tr_loss),
            "train_acc": float(tr_acc),
            "test_loss": float(te_loss),
            "test_acc": float(te_acc),
            "grad_norm_mean": float(grad_mean),
            "grad_norm_max": float(grad_max),
            "coeff_l2_norm": float(coeff_norm),
            "divisor_points_l2_norm": float(divisor_norm),
            "lr": lr,
            "epoch_time_sec": float(dt),
        }
        history.append(row)
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        if te_acc > best_acc:
            best_acc = te_acc
            best_epoch = ep
            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "metrics": row,
                },
                os.path.join(run_dir, "best.pt"),
            )
        if (ep % max(1, args.checkpoint_every)) == 0:
            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "metrics": row,
                },
                os.path.join(run_dir, f"epoch_{ep:03d}.pt"),
            )
        print(
            f"Epoch {ep:02d} | train {tr_loss:.4f}/{tr_acc:.2f}% | "
            f"test {te_loss:.4f}/{te_acc:.2f}% | "
            f"grad(mean/max) {grad_mean:.3f}/{grad_max:.3f} | "
            f"|coeff|={coeff_norm:.3f}, |div|={divisor_norm:.3f} | {dt:.1f}s"
        )

    summary = {
        "run_name": run_name,
        "genus": args.genus,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_subset": args.train_subset,
        "test_subset": args.test_subset,
        "best_test_acc": float(best_acc),
        "best_epoch": int(best_epoch),
        "history_len": len(history),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Done. Best test acc={best_acc:.2f}% at epoch {best_epoch}.")
    print(f"Saved: {history_path}, {summary_path}, best.pt, epoch_*.pt")


if __name__ == "__main__":
    main()

