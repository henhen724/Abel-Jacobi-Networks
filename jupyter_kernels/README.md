# Jupyter kernel: Python (aj_env)

Kernel spec for the project’s conda environment, following the same pattern as `~/diffusion_venv` (activation script + launch wrapper).

- **Install:** from repo root run `bash scripts/install_aj_env_kernel.sh`
- **Installed location:** `~/.local/share/jupyter/kernels/aj_env/` (launch script + generated `kernel.json`)
- **Activation script:** `~/activate_aj_env.sh` (copied from repo `activate_aj_env.sh` by the install script)

The `kernel.json` here uses a placeholder path; the install script writes the real path when copying to `~/.local/share/jupyter/kernels/aj_env/`.
