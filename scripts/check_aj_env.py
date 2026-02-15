#!/usr/bin/env python3
"""Check that the aj environment has the required packages and the library is importable."""

import sys

def main():
    errors = []
    # Core (tropical + classical numpy)
    for name in ("numpy", "matplotlib", "networkx", "mpmath", "tqdm"):
        try:
            __import__(name)
        except ImportError as e:
            errors.append(f"Missing: {name} ({e})")
    # Library
    try:
        import aj
        from aj.tropical import build_chain_of_loops, tropical_abel_jacobi_forward
        from aj.classical import make_hyperelliptic_cuts, abel_jacobi_forward
    except ImportError as e:
        errors.append(f"Could not import aj library: {e}")
    if errors:
        print("Environment check failed:")
        for e in errors:
            print("  -", e)
        print("\nCreate the environment with:")
        print("  conda env create -f environment.yml && conda activate aj")
        print("  or: python -m venv .venv && source .venv/bin/activate && pip install -e .")
        return 1
    print("aj environment OK. Version:", aj.__version__)
    return 0

if __name__ == "__main__":
    sys.exit(main())
