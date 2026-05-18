# How the diffusion Python venv was set up

This describes **`~/diffusion_venv_py312`**, the overlay venv used with **`~/activate_diffusion.sh`** on Sherlock. The exact creation command is recorded in the venv metadata file `~/diffusion_venv_py312/pyvenv.cfg` (field `command`).

## Idea

- **Heavy stack** (Python 3.12, PyTorch, NumPy, SciPy, torchvision, scikit-learn) comes from **Sherlock modules**, not from pip inside the venv.
- The venv is a **pip overlay**: extra packages install into the venv’s `site-packages`, while imports for the module stack still work because the venv was created with **`--system-site-packages`**.

## Steps (reproduce on a login node)

1. Load the same modules you intend to use at runtime (so `which python3` is the module interpreter you want—here **`python/3.12.1`** from the cluster).

2. Create the venv **with system site packages**:

   ```bash
   python3 -m venv --system-site-packages "$HOME/diffusion_venv_py312"
   ```

   On this account, `pyvenv.cfg` shows that `python3` was the module binary under `/share/software/user/open/python/3.12.1/bin/python3.12` when the venv was created.

3. Add an activation script (e.g. **`~/activate_diffusion.sh`**) that you **`source`** (not run). It should:

   - `module purge` (optional) then **`module load`** the full dependency set (e.g. `devel`, `math`, `python/3.12.1`, `py-pytorch/2.4.1_py312`, `py-torchvision/0.19.1_py312`, `py-scipy/…`, `py-scikit-learn/…`, `py-numpy/…`).
   - Then **`source "$HOME/diffusion_venv_py312/bin/activate"`**.

   Order matters: modules first, then the venv, so every job and notebook sees one consistent environment.

4. Install **only** extra pip-only dependencies with the environment active, e.g. from the diffusion repo:

   ```bash
   source ~/activate_diffusion.sh && pip install -r requirements.txt
   ```

## Why not a plain `venv` without `--system-site-packages`?

Without it, the venv typically does not see the module-installed PyTorch/NumPy tree, and you would duplicate large wheels in the venv or break linkage on the cluster. **`--system-site-packages`** keeps the module stack visible and layers pip installs on top.

## Note on naming

There is also a **`~/diffusion_venv`** directory on disk; the workflow documented in the convolutional_diffusion project rules is the **py312** path above plus **`activate_diffusion.sh`**.

## Abel–Jacobi (this repo) on Sherlock

Use the **same** pattern: cluster modules for PyTorch/NumPy/SciPy, then a **`--system-site-packages`** venv at **`~/aj_env_venv`** for pip-only pieces. Sherlock’s module **`python3` does not include `ipykernel`**, so Jupyter’s **Python (aj_env)** kernel must run **`~/aj_env_venv/bin/python -m ipykernel_launcher`** after that venv exists.

From the repo root on a login node:

```bash
bash scripts/install_aj_env_kernel.sh
```

That script refreshes **`~/activate_aj_env.sh`** from the repo, creates **`~/aj_env_venv`** with **`--system-site-packages`** when needed, **`pip install`s** `ipykernel` plus notebook/library deps, and installs the kernel spec under **`~/.local/share/jupyter/kernels/aj_env/`**. Then in Sherlock Jupyter, pick kernel **Python (aj_env)**.
