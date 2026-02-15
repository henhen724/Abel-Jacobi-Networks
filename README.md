# Abel–Jacobi Networks

Research code for **tropical** and **classical** Abel–Jacobi maps on hyperelliptic curves, with applications to metric graphs, period integrals, and neural network training.

## Overview

This repository contains Jupyter notebooks that:

- **Tropical Abel–Jacobi** — Represent hyperelliptic curves as metric graphs (chains of loops), compute the tropical Abel–Jacobi map for divisors, and visualize the image in the Jacobian torus.
- **Classical Abel–Jacobi** — Precompute period integrals and Abel–Jacobi lookup tables for higher genus (e.g. genus 30) and use them in training.

## Contents

| Notebook | Description |
|----------|-------------|
| `tropical_abel_jacobi.ipynb` | Tropical perspective: build chain-of-loops metric graphs, compute tropical Abel–Jacobi coordinates for divisors, and plot the graph and Jacobian projection. |
| `higher_genus_lookup_tables.ipynb` | Precompute Abel–Jacobi integrals and period matrices (ω) for a hyperelliptic curve of fixed genus on a grid; saves tables (e.g. for genus 30) for use in training. |
| `AJ_training_genus30.ipynb` | Training pipeline using pre-computed genus-30 Abel–Jacobi tables; compares with 2D projection and 2g×2g baselines. |
| `planewave_analysis.ipynb` | Analysis of periodic/plane-wave style approximations (e.g. sums with Gaussian decay) used in the Abel–Jacobi setting. |

## Requirements

- **Core:** Python 3, NumPy, Matplotlib, NetworkX  
- **Tropical notebook:** No extra installs beyond the above.  
- **Higher-genus / training:** PyTorch, mpmath, tqdm; notebooks assume Google Colab and Google Drive for saving/loading large tables (paths can be adjusted for local use).

## Quick start (tropical)

```bash
# Optional: create a venv and install
pip install numpy matplotlib networkx jupyter
jupyter notebook tropical_abel_jacobi.ipynb
```

Run the notebook to build a chain-of-loops graph, define a divisor, and view the tropical Abel–Jacobi coordinates and plots.

## License

Private research repository. All rights reserved.
