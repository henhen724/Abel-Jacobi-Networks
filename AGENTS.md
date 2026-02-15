# Instructions for Cursor Agents

Use this file when working in the **Abel–Jacobi Networks** repository.

## Project context

- **Domain:** Tropical and classical Abel–Jacobi maps on hyperelliptic curves; metric graphs (chains of loops); period integrals and lookup tables; neural network training using Abel–Jacobi features.
- **Artifacts:** The **`aj`** Python package (library) plus Jupyter notebooks (`.ipynb`). The library provides tropical and classical Abel–Jacobi APIs; notebooks can import from `aj` or use inlined code. Some notebooks assume Google Colab and Google Drive paths; support both local and Colab when changing paths or adding features.

## Conventions

- **Notebooks:** Keep narrative in markdown cells; put reusable logic in functions with docstrings. Use `numpy`/`matplotlib`/`networkx` for tropical notebooks; `torch`/`mpmath`/`tqdm` for higher-genus and training notebooks.
- **Math:** Preserve mathematical notation in comments and docstrings (e.g. genus `g`, cycle lengths, divisors as lists of `(node, weight)`).
- **Paths:** Do not hardcode absolute paths. Prefer config variables at the top of the notebook (e.g. `SAVE_DIR`, `DRIVE_FOLDER`) so they can be switched for local vs Colab.
- **Dependencies:** Document new dependencies in the README and, when adding `!pip install`, keep a minimal list (e.g. `torch`, `numpy`, `mpmath`, `tqdm`).

## When editing

- **`aj` library:** Top-level `aj` exposes `build_chain_of_loops`, `cycle_data`, `tropical_abel_jacobi_forward`, `make_hyperelliptic_cuts`, `abel_jacobi_forward`. Tropical code lives in `aj.tropical`, classical in `aj.classical`. The alias `tropical_abel_jacobi_divisor` = `tropical_abel_jacobi_forward` is kept for notebook compatibility.
- **tropical_abel_jacobi.ipynb:** Graph construction uses `length` edge attribute and node names like `v0`, `l0_a`, `l0_b`. Do not change the contract of `build_chain_of_loops`, `cycle_data`, or `tropical_abel_jacobi_divisor` without updating call sites and README.
- **higher_genus_lookup_tables.ipynb / AJ_training_genus30.ipynb:** File names and keys (e.g. `aj_integrals_genus30.pt`, `omega_plus`, `grid_r`, `grid_i`) are shared between notebooks; keep them in sync when changing save/load format.
- **New notebooks:** Add a short description and requirements to the README table and to AGENTS.md if they introduce new conventions.

## Testing and runs

- Tropical notebook should run with no external files. Higher-genus and training notebooks require generated tables or Drive mount; document required inputs and any `.pt`/Drive layout in the README or in the notebook’s first cell.
