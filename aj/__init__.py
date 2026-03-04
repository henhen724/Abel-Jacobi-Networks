"""
Development-time shim package for ``aj`` when using a ``src/aj`` layout.

When the repository is on ``sys.path`` but the project has not been
installed (``pip install -e .``), this stub forwards imports so that
``import aj`` and ``import aj.classical`` continue to work.

In installed environments, the real ``aj`` package from the wheel takes
precedence and this file is not used.
"""

from importlib import import_module as _import_module

_impl = _import_module("src.aj")

# Re-export public names from src.aj
for _name, _value in list(_impl.__dict__.items()):
    if _name.startswith("_"):
        continue
    globals()[_name] = _value

__all__ = getattr(_impl, "__all__", [n for n in globals() if not n.startswith("_")])

# Share the same package path so that subpackages like aj.classical resolve.
try:
    __path__ = _impl.__path__  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - defensive
    pass

