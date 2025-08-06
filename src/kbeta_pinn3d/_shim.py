# _shim.py – just run the original monolithic module
import runpy
import sys


def entry() -> None:  # ← console‑script target
    # Pass through any CLI flags exactly as the user typed them
    runpy.run_module("kbeta_pinn3d.pinn3d", run_name="__main__")
