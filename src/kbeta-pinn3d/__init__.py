"""3‑D cylindrical PINN demo shipped with the Kourkoutas‑β paper."""
__all__ = ["pinn3d"]

from .pinn3d import main as run_cli  # exposes `python -m kbeta_pinn3d`
