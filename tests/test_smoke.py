"""
tests/test_smoke.py
––––––––––––––––––––
Very‑light integration test for kbeta‑pinn3d:

• Asserts that the top‑level package can be imported.
• Launches a one‑epoch run via the CLI (Adam‑95, no viz, no diagnostics).

The test is intentionally tiny so that it executes within the default
10‑second GitHub Actions timeout even on the free macOS arm64 runners.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_import_package() -> None:
    """Import should succeed and expose the version string."""
    import kbeta_pinn3d as pkg  # noqa: F401
    assert hasattr(pkg, "__version__")


def test_cli_one_epoch(tmp_path: Path) -> None:
    """
    Run `python -m kbeta_pinn3d.pinn3d --epochs 1 --optimizer adam95`
    inside an empty working directory to avoid polluting the repo tree.
    """
    cmd = [
        sys.executable,
        "-m",
        "kbeta_pinn3d.pinn3d",
        "--epochs", "1",
        "--optimizer", "adam95",
    ]
    # The subprocess inherits our virtual‑env, so MLX is already available.
    # `check=True` → pytest will fail if the return‑code is non‑zero.
    subprocess.run(cmd, cwd=tmp_path, check=True)
