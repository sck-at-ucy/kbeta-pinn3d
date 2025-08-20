# tests/test_smoke.py
"""
Smoke tests for the installed kbeta_pinn3d package.

These confirm:
1. The package imports correctly and exposes a version string.
2. The CLI entry point runs a tiny job without crashing.
"""

from __future__ import annotations

import shutil
import subprocess
import sys


def test_import_package() -> None:
    """Import should succeed and expose the version string."""
    import kbeta_pinn3d as pkg

    assert hasattr(pkg, "__version__")
    assert isinstance(pkg.__version__, str)
    assert len(pkg.__version__) > 0


def test_cli_one_epoch(tmp_path) -> None:
    """
    Run a 1-epoch training job to check the CLI entry point works.

    Uses the console script `kbeta-pinn3d` if installed (wheel case).
    Falls back to `python -m kbeta_pinn3d.pinn3d` if running editable.
    """
    if shutil.which("kbeta-pinn3d"):
        cmd = ["kbeta-pinn3d", "--epochs", "1", "--optimizer", "adam95"]
    else:
        cmd = [
            sys.executable,
            "-m",
            "kbeta_pinn3d.pinn3d",
            "--epochs",
            "1",
            "--optimizer",
            "adam95",
        ]

    subprocess.run(cmd, cwd=tmp_path, check=True)
