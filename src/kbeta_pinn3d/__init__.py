"""Light‑weight namespace for the PINN demo – leaves training code untouched."""

from importlib.metadata import version as _v

__version__: str = _v(__package__ or "kbeta_pinn3d")

__all__ = ["__version__"]  # nothing else is exported implicitly
