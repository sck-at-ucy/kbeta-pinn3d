# ──────────────────────────────────────────────────────────────────────────────
# visualization.py – optional field‑plotting helpers for the cylindrical PINN
# ──────────────────────────────────────────────────────────────────────────────
"""
High‑level utilities to inspect the learned temperature field **after training**.

Functions
---------
evaluate_slice(...)        → build an r‑θ slice at constant z (MLX arrays)
plot_slice_2d(...)         → 2‑D filled contour of a single slice
plot_stacked_slices(...)   → several slices in one 3‑D figure
plot_scatter_3d(...)       → sparse 3‑D scatter of T inside the domain

These routines require the *visualisation* dependency group:

    pip install kbeta-pinn3d[viz]

If the stack is missing, each plotting routine raises a clear
`ImportError` explaining how to install it, while simply **importing**
`kbeta_pinn3d` never fails.

Copyright
---------
MIT © 2025 Stavros Kassinos
"""
# -----------------------------------------------------------------------------


from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx
import numpy as np


# -----------------------------------------------------------------------------#
# 1.  Computational helper – no heavy deps
# -----------------------------------------------------------------------------#
def evaluate_slice(
    T_fn,
    r_min: float,
    r_max: float,
    z_value: float,
    N_theta: int = 100,
    N_r: int = 100,
):
    """Return **MLX** arrays (R, Θ, T) for a constant‑z slice."""
    theta_lin = mx.linspace(0, 2 * math.pi, N_theta)
    rows_r, rows_th = [], []
    for th in theta_lin:
        r_out = r_max + 0.25 * r_max * math.sin(3 * float(th))
        rows_r.append(mx.linspace(r_min, r_out, N_r))
        rows_th.append(mx.full((N_r,), th))
    R = mx.stack(rows_r)
    TH = mx.stack(rows_th)
    Z = mx.full(R.shape, z_value)
    flat = mx.stack(
        [mx.reshape(R, (-1,)), mx.reshape(TH, (-1,)), mx.reshape(Z, (-1,))], axis=1
    )
    T = mx.reshape(mx.vmap(T_fn)(flat), R.shape)
    return R, TH, T


# -----------------------------------------------------------------------------#
# 2.  Plotting helpers – heavy deps guarded by try/except
# -----------------------------------------------------------------------------#
def _require_viz_stack():
    """Import matplotlib (and friends) or raise a friendly error."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ImportError(
            "Optional visualisation stack missing. "
            "Install it via  pip install kbeta-pinn3d[viz]  "
            "or run without the --viz flag."
        ) from e
    return plt


# .............................................................................
def plot_slice_2d(
    R,
    TH,
    T,
    *,
    z_val: float,
    r_min: float,
    r_max: float,
    label: str = "T",
    outdir: str | Path = "./plots",
):
    """Save a filled‑contour r‑θ slice and return the PNG path."""
    plt = _require_viz_stack()
    from matplotlib import pyplot as _  # satisfies type checkers

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fname = outdir / f"{label.lower()}_slice_z{z_val:.2f}.png"

    X, Y = R * mx.cos(TH), R * mx.sin(TH)
    levels = np.linspace(0.0, 1.01, 100)

    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, T, levels=levels, cmap="viridis")
    plt.colorbar(label="Temperature")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"{label} field at z = {z_val:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    return fname


# .............................................................................
def plot_stacked_slices(
    T_fn,
    r_min: float,
    r_max: float,
    length_z: float,
    slice_z_values: Sequence[float],
    N_theta: int = 100,
    N_r: int = 100,
    shared_colorbar: bool = True,
    label: str = "T",
    outdir: str | Path = "./plots",
):
    """Render several r‑θ slices in one 3‑D figure."""
    plt = _require_viz_stack()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fname = outdir / f"{label.lower()}_stacked_slices.png"

    cm = plt.cm.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    global_min, global_max, cache = float("inf"), -float("inf"), []
    for z in slice_z_values:
        R, TH, T = evaluate_slice(T_fn, r_min, r_max, z, N_theta, N_r)
        cache.append((R, TH, z, T))
        global_min = min(global_min, float(mx.min(T)))
        global_max = max(global_max, float(mx.max(T)))

    for R, TH, z, T in cache:
        X, Y = R * mx.cos(TH), R * mx.sin(TH)
        Z = mx.full(X.shape, z)
        norm = (T - global_min) / (global_max - global_min + 1e-9)
        ax.plot_surface(
            X,
            Y,
            Z,
            facecolors=cm(norm),
            rstride=1,
            cstride=1,
            antialiased=False,
            shade=False,
            linewidth=0,
        )

    ax.set_xlim(-1.3 * r_max, 1.3 * r_max)
    ax.set_ylim(-1.3 * r_max, 1.3 * r_max)
    ax.set_zlim(0, length_z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"{label}: stacked slices in 3‑D")

    if shared_colorbar:
        mappable = plt.cm.ScalarMappable(cmap=cm)
        mappable.set_clim(global_min, global_max)
        plt.colorbar(mappable, ax=ax, shrink=0.6, label=label)

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    return fname


# .............................................................................
def plot_scatter_3d(
    T_fn,
    r_min: float,
    r_max: float,
    length_z: float,
    N: int = 30,
    label: str = "T",
    outdir: str | Path = "./plots",
):
    """Sparse 3‑D scatter of T inside the physical domain."""
    plt = _require_viz_stack()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fname = outdir / f"{label.lower()}_scatter3d.png"

    # regular Cartesian grid (mask later)
    R_bound = r_max + 0.25 * r_max
    x = np.linspace(-R_bound, R_bound, N)
    y = np.linspace(-R_bound, R_bound, N)
    z = np.linspace(0, length_z, N)
    Xv, Yv, Zv = np.meshgrid(x, y, z, indexing="ij")
    pts = np.stack([Xv.ravel(), Yv.ravel(), Zv.ravel()], axis=1)

    # mask: keep points inside distorted cylinder
    r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    th = np.mod(np.arctan2(pts[:, 1], pts[:, 0]), 2 * np.pi)
    r_out = r_max + 0.25 * r_max * np.sin(3 * th)
    mask = (r >= r_min) & (r <= r_out)
    cyl = mx.array(np.stack([r[mask], th[mask], pts[mask, 2]], axis=1))

    T = mx.vmap(T_fn)(cyl)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pts[mask, 0], pts[mask, 1], pts[mask, 2], c=T, cmap="viridis", s=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.colorbar(sc, label=label)
    plt.title(f"{label}: 3‑D scatter")

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    return fname
