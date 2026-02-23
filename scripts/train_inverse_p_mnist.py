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


def train_epoch(model, loader, opt, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    tot, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return tot / n, 100.0 * correct / n


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
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-subset", type=int, default=1024)
    p.add_argument("--test-subset", type=int, default=512)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train_ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    if args.train_subset > 0:
        train_ds = torch.utils.data.Subset(train_ds, list(range(min(args.train_subset, len(train_ds)))))
    if args.test_subset > 0:
        test_ds = torch.utils.data.Subset(test_ds, list(range(min(args.test_subset, len(test_ds)))))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=max(args.batch_size, 128), shuffle=False, num_workers=0)

    model = MNISTInversePNet(args.genus, device).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    print(f"Training MNIST Inverse-P model: genus={args.genus}, epochs={args.epochs}, device={device}")
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, device)
        te_loss, te_acc = eval_epoch(model, test_loader, device)
        print(f"Epoch {ep:02d} | train {tr_loss:.4f}/{tr_acc:.2f}% | test {te_loss:.4f}/{te_acc:.2f}%")


if __name__ == "__main__":
    main()

