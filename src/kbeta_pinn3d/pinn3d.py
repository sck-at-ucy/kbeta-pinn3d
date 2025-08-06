"""
===============================================================================
3‑D Cylindrical PINN for Steady Heat Conduction  (single‑process, pure‑MLX)
-------------------------------------------------------------------------------
Author   : Stavros Kassinos  <kassinos.stavros@ucy.ac.cy>
Revision : v0.1.0    (August 2025)
Hardware : Apple‑Silicon GPU (Metal / MLX backend)

PROBLEM -----------------------------------------------------------------------
Solve ∇²T = 0 inside a distorted cylinder (r,θ,z) subject to mixed boundaries:
  • Inner cylinder r = r_min   : Dirichlet  T = 1
  • Inlet plane   z = 0       : Dirichlet  T = 1
  • Outlet plane  z = L_z     : Neumann    ∂T/∂z = 0
  • Outer wall    r = r_out(θ): piece‑wise heat‑flux
  • 2π‑periodicity in θ for T and ∂T/∂θ

Our Physics‑Informed Neural Network (PINN) is a fully‑connected MLP trained
with MLX’s JIT compiler.  Either **Adam** (β₂ = 0.95 / 0.999) or the paper’s
**Kourkoutas‑β** optimiser can be selected from the CLI.

KEY FEATURES ------------------------------------------------------------------
• Completely **MPI‑free** – runs out‑of‑the‑box on MacBook M‑series.
• **Analytic cylindrical Laplacian** – no second‑order autodiff graph bloat.
• **Mixed BC + piece‑wise flux** drive large gradient variance.
• Optional **sun‑spike / β₂ tracing hooks** (`--collect_spikes`) for the plots
  shown in the paper.
• **Lightweight visualisation stack** (`[viz]` extra) to generate 2‑D slices
  and 3‑D scatter plots after training.

LICENSE -----------------------------------------------------------------------
MIT © 2025 Stavros Kassinos  – see LICENSE file for full text.
"""

# pinn3d.py (inside kbeta_pinn3d)
from __future__ import annotations

import argparse
import math
import os
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from kbeta.optim import KourkoutasSoftmaxFlex

mx.set_default_device(mx.gpu)


# -----------------------------------------------------------------------------
# 0. CLI ─ enables / disables plotting without pulling extra deps
# -----------------------------------------------------------------------------
def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3‑D cylindrical heat‑conduction PINN (MX‑GPU, single process)"
    )

    # ── optimiser / generic training flags ──────────────────────────────────
    p.add_argument(
        "--optimizer",
        choices=["adam95", "adam999", "kourkoutas"],
        default="kourkoutas",
        help="Optimiser to use",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=6_000,
        help="Number of training epochs",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Model‑initialisation & data seed (collocation mesh is fixed)",
    )

    # ── visualisation & diagnostics ─────────────────────────────────────────
    p.add_argument(
        "--viz",
        action="store_true",
        help="Run the optional temperature field visualisation at the end",
    )
    p.add_argument(
        "--kour_diagnostics",
        action="store_true",
        help="Enable lightweight diagnostics in KourkoutasSoftmaxFlex "
        "(adds ≈2 %% overhead)",
    )

    # ── *tracking* group (Sun‑spike / β₂ plots) ─────────────────────────────
    p.add_argument(
        "--collect_spikes",
        action="store_true",
        help="Store per‑layer Sun‑spike / β₂ statistics for violin & density plots "
        "(implies --kour_diagnostics when the optimiser is Kourkoutas)",
    )
    p.add_argument(
        "--window",
        type=int,
        default=None,
        metavar="N",
        help="Epochs per aggregation bin in the violin plot "
        "(maps to tracking.window; defaults to 500 if omitted)",
    )
    p.add_argument(
        "--plot_stride",
        type=int,
        default=None,
        metavar="EPOCHS",
        help="Down‑sample violin categories (default 10×window). "
        "Use 1 to keep every window collected.",
    )
    p.add_argument(
        "--outdir",
        default="OUTPUTS_PINN3D",
        metavar="PATH",
        help="Root directory for checkpoints & plots (default: %%(default)s)",
    )

    args = p.parse_args()

    # ── convenience: spike collection → diagnostics auto‑enable ─────────────
    if args.collect_spikes and not args.kour_diagnostics:
        print("[info] --collect_spikes implies --kour_diagnostics → auto‑enabled")
        args.kour_diagnostics = True

    return args


ARGS = _parse_cli()

# -----------------------------------------------------------------------------
# I/O‑root & helpers  (NEW CODE)
# -----------------------------------------------------------------------------
BASE_OUT: Path = Path(ARGS.outdir).expanduser().resolve()
PLOTS_DIR: Path = BASE_OUT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)  # one mkdir ≃ zero overhead


def _subdir(name: str) -> str:
    """
    Return *str* path to BASE_OUT / plots / <name> after creating it.
    All downstream code stays identical – we only swapped literals.
    """
    d = PLOTS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


COLLOC_SEED = 12345  # fixed for the whole project ─ collocation mesh

my_seed = ARGS.seed
MODEL_SEED = my_seed


OPTIMIZER_SELECTED = ARGS.optimizer.upper()
num_epochs = ARGS.epochs


# --- generate data with a fixed seed -------------------
mx.random.seed(COLLOC_SEED)
np.random.seed(COLLOC_SEED)

print(f"mx-Random: {mx.random.state}")
print(f"np-Random: {COLLOC_SEED}")
print(f"number of epochs: {num_epochs}")

# =============================================================================
# 1. Define Domain & Sampling
# =============================================================================
r_min = 0.2
r_max = 1.0
length_z = 10.0 * r_max  # cylinder extends from z=0 to z=10*r_max

num_interior = 4000  # Number of interior sample points
num_boundary = 2000  # Number of sample points for each boundary region

# --- 3D Interior Points ---
theta_interior = mx.random.uniform(0, 2 * math.pi, (num_interior,))
z_interior = mx.random.uniform(0, length_z, (num_interior,))
r_outer_interior = r_max + 0.25 * r_max * mx.sin(3 * theta_interior)
u = mx.random.uniform(0, 1, (num_interior,))
r_interior = r_min + u * (r_outer_interior - r_min)

interior_points_3D = mx.stack([r_interior, theta_interior, z_interior], axis=1)

print("Interior Points 3D:")
print(f"  r range = [{float(mx.min(r_interior)):.3f}, {float(mx.max(r_interior)):.3f}]")
print(f"  z range = [{float(mx.min(z_interior)):.3f}, {float(mx.max(z_interior)):.3f}]")
print("Sample Interior (first 5):", interior_points_3D[:5])


# -----------------------------------------------------------------------------
# BOUNDARIES:
# Define separate sets of boundary points for:
#   1) Inner cylinder r = r_min  (T=1)
#   2) Inlet plane z=0         (T=1)
#   3) Outlet plane z=length_z  (dT/dz=0)
#   4) Outer boundary r=r_out(theta) (flux condition)
# -----------------------------------------------------------------------------

# --- Inner Cylinder (r=r_min, all theta,z) => T=1
theta_inner = mx.random.uniform(0, 2 * math.pi, (num_boundary,))
z_inner = mx.random.uniform(0, length_z, (num_boundary,))
r_inner = mx.full((num_boundary,), r_min)
boundary_inner_3D = mx.stack([r_inner, theta_inner, z_inner], axis=1)

# --- Inlet Plane (z=0), r in [r_min, r_outer(theta)], theta in [0,2π]
theta_inlet = mx.random.uniform(0, 2 * math.pi, (num_boundary,))
r_outer_inlet = r_max + 0.25 * r_max * mx.sin(3 * theta_inlet)
u_inlet = mx.random.uniform(0, 1, (num_boundary,))
r_inlet = r_min + u_inlet * (r_outer_inlet - r_min)
z_inlet = mx.zeros_like(r_inlet)  # exactly z=0
boundary_inlet_3D = mx.stack([r_inlet, theta_inlet, z_inlet], axis=1)

# --- Outlet Plane (z=length_z), we want dT/dz=0
theta_outlet = mx.random.uniform(0, 2 * math.pi, (num_boundary,))
r_outer_outlet = r_max + 0.25 * r_max * mx.sin(3 * theta_outlet)
u_outlet = mx.random.uniform(0, 1, (num_boundary,))
r_outlet = r_min + u_outlet * (r_outer_outlet - r_min)
z_outlet = mx.full((num_boundary,), length_z)
boundary_outlet_3D = mx.stack([r_outlet, theta_outlet, z_outlet], axis=1)

# --- Outer Boundary (r = r_out(theta)), for all theta,z
theta_outer = mx.random.uniform(0, 2 * math.pi, (num_boundary,))
z_outer = mx.random.uniform(0, length_z, (num_boundary,))
r_outer_vals = r_max + 0.25 * r_max * mx.sin(3 * theta_outer)
boundary_outer_3D = mx.stack([r_outer_vals, theta_outer, z_outer], axis=1)

mx.eval(interior_points_3D)
mx.eval(boundary_inner_3D)
mx.eval(boundary_inlet_3D)
mx.eval(boundary_outlet_3D)
mx.eval(boundary_outer_3D)

print("Boundary sets shapes:")
print("  Inner Cylinder:", boundary_inner_3D.shape)
print("  Inlet Plane:", boundary_inlet_3D.shape)
print("  Outlet Plane:", boundary_outlet_3D.shape)
print("  Outer Cylinder:", boundary_outer_3D.shape)


print(f"interior points shape {interior_points_3D.shape}")

# =============================================================================
# 2. Reset random seeds for model initialization
# =============================================================================

mx.random.seed(MODEL_SEED)
np.random.seed(MODEL_SEED)
print(f"mx-Random: {mx.random.state}")
print(f"np-Random: {my_seed}")


# =============================================================================
# 3. Define the 3D MLP Model (inputs: [r, theta, z])
# =============================================================================
class MLP_3D(nn.Module):
    def __init__(
        self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int
    ):
        """
        A fully-connected neural network (MLP) to learn T(r, θ, z).

        This MLP is used in a PINN context to approximate the temperature field
        in a cylindrical domain with PDE and boundary constraints.

        Args:
            num_layers (int): Number of linear layers. Must be >= 1.
            input_dim (int): Dimensionality of inputs (3 for [r, θ, z]).
            hidden_dim (int): Number of neurons in each hidden layer.
            output_dim (int): Dimensionality of outputs (1 for T).
        """
        super().__init__()
        self.layers = []
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Perform the forward pass through the MLP.

        Args:
            x (mx.array): Input array with shape [..., 3] representing [r, theta, z].

        Returns:
            mx.array: The output of the MLP with shape [..., output_dim].
        """
        for layer in self.layers[:-1]:
            # Use the SiLU activation function for hidden layers.
            x = nn.silu(layer(x))
        return self.layers[-1](x)


# =============================================================================
# 4. Periodicity in θ
# =============================================================================

N_periodic_3D = 400
r_periodic_3D = mx.random.uniform(r_min, r_max, (N_periodic_3D,))
z_periodic_3D = mx.random.uniform(0, length_z, (N_periodic_3D,))

# Force some fraction of them to be at z=0 (pinned)
n_pin = N_periodic_3D // 5  # 20% pinned at z=0
z_periodic_3D[:n_pin] = 0.0

theta0_3D = mx.zeros_like(r_periodic_3D)
theta2pi_3D = mx.full(r_periodic_3D.shape, 2 * math.pi)

theta0_points_3D = mx.stack([r_periodic_3D, theta0_3D, z_periodic_3D], axis=1)
theta2pi_points_3D = mx.stack([r_periodic_3D, theta2pi_3D, z_periodic_3D], axis=1)


def periodicity_loss_3D():
    """
    Enforce T(θ=0) = T(θ=2π) and ∂T/∂θ(θ=0) = ∂T/∂θ(θ=2π) for points sharing
    the same r, z but differing by 2π in θ.

    Returns:
        mx.array: Mean squared difference for T and its θ-derivative.
    """
    T0 = mx.vmap(T_3D)(theta0_points_3D)
    T2pi = mx.vmap(T_3D)(theta2pi_points_3D)

    grad0 = mx.vmap(grad_T_3D)(theta0_points_3D)
    grad2pi = mx.vmap(grad_T_3D)(theta2pi_points_3D)
    dT_dtheta_0 = grad0[:, 1]
    dT_dtheta_2pi = grad2pi[:, 1]

    return mx.mean((T0 - T2pi) ** 2 + (dT_dtheta_0 - dT_dtheta_2pi) ** 2)


# Instantiate the MLP for 3D domain
mlp_model = MLP_3D(num_layers=16, input_dim=3, hidden_dim=128, output_dim=1)


# =============================================================================
# 5. Define T(r, theta, z) and the 3D Laplacian in cylindrical coordinates
# =============================================================================
def T_3D(x):
    """
    Compute the temperature at a single point (r, θ, z).

    This function wraps the MLP_3D model so we can easily compute gradients
    w.r.t. inputs.

    Args:
        x (mx.array): shape (3,) for [r, θ, z], or broadcastable to (..., 3).

    Returns:
        float or mx.array: Temperature value(s) predicted by the MLP.
    """
    return mlp_model(x)[0]


# Compute the gradient of T with respect to [r, theta, z]
grad_T_3D = mx.grad(T_3D)


def cylindrical_laplacian_T(x):
    """
    Compute the Laplacian of T in cylindrical coordinates.

    ∇²T = (1/r)*∂/∂r(r * ∂T/∂r) + (1/r²)*∂²T/∂θ² + ∂²T/∂z²

    Args:
        x (mx.array): shape (3,). The point [r, θ, z].

    Returns:
        mx.array: A scalar representing ∇²T at x.
    """
    r = x[0]

    # (1) term_r = 1/r * d/dr( r * dT/dr )
    def F(x):
        return x[0] * grad_T_3D(x)[0]  # r * (dT/dr)

    dF_dr = mx.grad(F)(x)[0]
    term_r = dF_dr / r

    # (2) term_theta = (1/r^2) * d²T/dθ²
    d2T_dtheta2 = mx.grad(lambda xx: grad_T_3D(xx)[1])(x)[1]
    term_theta = d2T_dtheta2 / (r**2)

    # (3) term_z = d²T/dz²
    d2T_dz2 = mx.grad(lambda xx: grad_T_3D(xx)[2])(x)[2]

    return term_r + term_theta + d2T_dz2


# =============================================================================
# 6. Boundary Condition Helpers
# =============================================================================
def full_like(a, fill_value, dtype=None):
    """
    Create an array with the same shape as 'a', filled with a constant value.

    Args:
        a (mx.array): The reference array whose shape is to be mimicked.
        fill_value: The constant value to fill the new array.
        dtype (optional): Data type of the output array.

    Returns:
        mx.array: An array with the same shape as 'a' filled with fill_value.
    """
    return mx.full(a.shape, fill_value, dtype=dtype)


# 6.1 Inner Cylinder => T=1
def boundary_condition_inner_3D(x):  # Legacy helper -- kept for clarity
    """
    Enforce T=1 at the inner cylinder (r=r_min).

    Args:
        x (mx.array): shape (N, 3) of boundary points.

    Returns:
        mx.array: shape (N,), all ones.
    """
    return mx.ones_like(x[:, 0])


# 6.2 Inlet Plane => T=1
def boundary_condition_inlet_3D(x):  # Legacy helper -- kept for clarity
    """
    Enforce T=1 at the inlet plane (z=0).

    Args:
        x (mx.array): shape (N, 3).

    Returns:
        mx.array: shape (N,), all ones.
    """
    return mx.ones_like(x[:, 0])


# 6.3 Outlet Plane => dT/dz=0
def compute_dT_dz_3D(x):
    """
    Compute partial derivative ∂T/∂z for points on the outlet plane.

    Useful for imposing Neumann boundary condition (dT/dz=0).

    Args:
        x (mx.array): shape (N, 3).

    Returns:
        mx.array: shape (N,), each the dT/dz at the point.
    """
    grads = mx.vmap(grad_T_3D)(x)  # shape (N,3)
    dT_dz = grads[:, 2]
    return dT_dz


# 6.4 Outer Boundary => piecewise flux from 0 to 0.5
def piecewise_flux(z):
    """
    Compute a piecewise-defined flux for the outer boundary as a function of z.

    Regions:
      - 0 for z < 2.5*r_max
      - Ramp to 0.5 in [2.5*r_max, 7.5*r_max]
      - Constant 0.5 for z > 7.5*r_max

    Args:
        z (mx.array): shape (N,).

    Returns:
        mx.array: shape (N,), flux values.
    """
    ramp_start = 2.5 * r_max
    ramp_end = 7.5 * r_max
    flux_max = 0.5

    cond1 = z < ramp_start
    cond2 = mx.logical_and(z >= ramp_start, z <= ramp_end)

    fraction = (z - ramp_start) / (ramp_end - ramp_start)
    flux_ramp = flux_max * fraction

    flux_val = mx.where(
        cond1, mx.zeros_like(z), mx.where(cond2, flux_ramp, full_like(z, flux_max))
    )
    return flux_val


def compute_normal_derivative_3D_outer(x):
    """
    Compute the normal derivative ∂T/∂n on the distorted outer boundary.

    The outer boundary is defined by r = r_out(theta), and the outward normal is computed
    in the (r, theta) plane (ignoring z). The function returns the directional derivative
    along the outward normal.

    Args:
        x (mx.array): Array of boundary points with shape (N, 3).

    Returns:
        mx.array: The computed normal derivative ∂T/∂n for each point.
    """
    theta = x[:, 1]
    r_out = r_max + 0.25 * r_max * mx.sin(3 * theta)
    dr_dtheta = 0.75 * r_max * mx.cos(3 * theta)

    norm_factor = mx.sqrt(dr_dtheta**2 + r_out**2)
    n_r = r_out / norm_factor
    n_theta = -dr_dtheta / norm_factor

    grad_vals = mx.vmap(grad_T_3D)(x)
    dT_dr = grad_vals[:, 0]
    dT_dtheta = grad_vals[:, 1]

    dT_dn = dT_dr * n_r + (dT_dtheta / r_out) * n_theta
    return dT_dn


# =============================================================================
# 7. Define Final PDE Loss Function
# =============================================================================


def loss_fn_3D_realistic():
    """
    Compute the total PINN loss for the 3D steady heat diffusion problem.

    This includes:
      1) PDE residual (∇²T=0) at interior points.
      2) Boundary conditions:
         - Inner cylinder r=r_min (Dirichlet T=1),
         - Inlet plane z=0 (Dirichlet T=1),
         - Outlet plane z=length_z (Neumann dT/dz=0),
         - Distorted outer boundary with piecewise flux.
      3) θ-periodicity: T(θ=0)=T(θ=2π) and matching derivatives.

    Returns:
        mx.array: A scalar representing the weighted sum of all losses.
    """
    # (A) PDE interior loss
    lap_vals = mx.vmap(cylindrical_laplacian_T)(interior_points_3D)
    loss_interior = mx.mean(lap_vals**2)

    # (B) Inner boundary loss: T=1
    T_inner = mx.vmap(T_3D)(boundary_inner_3D)
    loss_inner = mx.mean((T_inner - 1.0) ** 2)

    # (C) Inlet plane loss: T=1
    T_inlet = mx.vmap(T_3D)(boundary_inlet_3D)
    loss_inlet = mx.mean((T_inlet - 1.0) ** 2)

    # (D) Outlet plane loss: dT/dz=0
    dTdz_outlet = compute_dT_dz_3D(boundary_outlet_3D)
    loss_outlet = mx.mean(dTdz_outlet**2)

    # (E) Outer boundary loss: normal derivative matches piecewise flux
    dT_dn_outer = compute_normal_derivative_3D_outer(boundary_outer_3D)
    z_out_vals = boundary_outer_3D[:, 2]
    flux_z = -piecewise_flux(z_out_vals)
    loss_outer = mx.mean((dT_dn_outer - flux_z) ** 2)

    # (F) Periodicity loss in θ
    loss_periodic = periodicity_loss_3D()

    # Return the weighted sum of all losses
    return mx.mean(loss_interior) + 0.05 * (
        +25 * loss_inner
        + 100 * loss_inlet
        + 25 * loss_outlet
        + 50 * loss_outer
        + 50 * loss_periodic
    )


# =============================================================================
# 8. Training Loop
# =============================================================================
# ---------------- learning schedule ----------------
init_lr = 1e-2  # start
target_lr = 1e-5  # value we want the end of ramp_steps
ramp_steps = 40_000  # “epochs” / optimizer steps

# 1) cosine ramp init_lr → target_lr over the first ramp_steps
cosine_part = optim.cosine_decay(init_lr, decay_steps=ramp_steps, end=target_lr)

# 2) constant part: simple lambda that ignores the incoming step
constant_part = lambda _: target_lr

# 3) stitch them together: after ramp_steps switch to the constant
lr_schedule = optim.join_schedules(
    [cosine_part, constant_part], [ramp_steps]  # boundary where we transition
)

if OPTIMIZER_SELECTED == "ADAM95":
    optimizer = optim.Adam(
        learning_rate=lr_schedule, betas=[0.90, 0.95], eps=1e-8, bias_correction=True
    )  # Optimizer instance
    print(optimizer)
    print(
        "ADAM95  "
        f"β1,β2={optimizer.betas} | "
        f"eps={optimizer.eps:.2e} |"
        f"bias_correction={optimizer.bias_correction}"
    )

elif OPTIMIZER_SELECTED == "ADAM999":
    optimizer = optim.Adam(
        learning_rate=lr_schedule, betas=[0.90, 0.999], eps=1e-8, bias_correction=True
    )  # Optimizer instance
    print(optimizer)
    print(
        "ADAM999  "
        f"β1,β2={optimizer.betas} | "
        f"eps={optimizer.eps:.2e} |"
        f"bias_correction={optimizer.bias_correction}"
    )


elif OPTIMIZER_SELECTED == "KOURKOUTAS":
    from mlx.utils import tree_flatten

    # 1) Build a map from param-object -> stable path (string or tuple)
    def build_param_to_path_map(model):
        param_to_path = {}
        flat_params = tree_flatten(model.parameters())
        for path, param in flat_params:
            param_to_path[param] = path  # e.g. "transformer_encoder.layers.3.ln2.bias"
        return param_to_path

    # 2) Create the dictionary *before* building the optimizer
    param_to_path_map = build_param_to_path_map(mlp_model)

    # 3) Define a layer_key_fn that uses your map
    def my_layer_key_fn(param):
        return param_to_path_map.get(param, "unknown_param")

    def my_layer_key_fn_shape_and_path(param):
        # Retrieve the stable path from the map:
        param_path = param_to_path_map.get(param, "unknown_param")

        # Then encode shape info:
        if param.ndim == 1:
            shape_key = ("1D", param.shape[0])
        elif param.ndim == 2:
            shape_key = ("2D", param.shape)
        else:
            shape_key = (f"{param.ndim}D", param.shape)

        # Return a tuple combining shape info and param path
        # You can decide how you want to combine or just use one or the other.
        return (shape_key, param_path)

    optimizer = KourkoutasSoftmaxFlex(
        learning_rate=lr_schedule,
        beta1=0.90,
        beta2_max=0.999,
        beta2_min=0.88,
        eps=1e-8,
        alpha=0.93,
        # alpha = alpha_schedule,
        tiny_spike=1.0e-9,
        tiny_denom=1.0e-8,
        decay=0.98,
        adaptive_tiny=True,
        max_ratio=3,
        warmup_steps=0,
        bias_correction="beta2max",
        layer_key_fn=lambda p: p.shape,
        # layer_key_fn=my_layer_key_fn,  #p: ("1D", p.shape[0]) if p.ndim==1 else id(p),  #my_layer_key_fn,
        # layer_key_fn=my_layer_key_fn_shape_and_path,
        # layer_key_fn=my_layer_key_fn_shape,
        # layer_key_fn=lambda p: id(p),   #my_layer_key_fn,
        # layer_key_fn=lambda p: "all",
        diagnostics=ARGS.kour_diagnostics,
    )

    print(optimizer)
    print(
        "KOUR  "
        f"β1={optimizer.beta1} | "
        f"β2_max={optimizer.beta2_max} | "
        f"β2_min={optimizer.beta2_min} | "
        f"α={optimizer.alpha} | "
        f"tiny_spike={optimizer.tiny_spike:.2e} | "
        f"tiny_denom={optimizer.tiny_denom:.2e} | "
        f"adaptT={optimizer.adaptive_tiny} | "
        f"decay={optimizer.decay or 'off'} | "
        f"maxR={optimizer.max_ratio or 'off'} | "
        f"warmup={int(optimizer.warmup_steps.item())} | "
        f"eps={optimizer.eps:.2e}"
    )


else:
    print("ERROR OPTIMIZER SELECTION")

mx.eval(mlp_model.parameters())
optimizer.init(mlp_model.parameters())
print(optimizer.state.keys())
state = [mlp_model.state, optimizer.state, mx.random.state]

mx.eval(state)


@partial(mx.compile, inputs=state, outputs=state)
def core_train_step():
    """
    Perform a single training iteration for the 3D PINN.

    1) Compute the total loss using `loss_fn_3D_realistic`.
    2) Compute gradients w.r.t. model parameters.
    3) Return (loss, grads) so the optimizer can apply an update.

    Returns:
        tuple: (loss, grads)
            - loss (mx.array): Scalar value of the current loss.
            - grads (dict/tree): Gradients of the loss w.r.t. MLP parameters.
    """
    loss_and_grad_fn = nn.value_and_grad(mlp_model, loss_fn_3D_realistic)
    loss, grads = loss_and_grad_fn()
    optimizer.update(mlp_model, grads)
    return loss


sunspike_dict = {}  # global or outside the loop
betas2_dict = {}

# ------------------------------------------------------------------
# 9. Allocate empty *buffers* once, outside the training loop
# ------------------------------------------------------------------
buffer_spikes: list[float] = []
buffer_betas2: list[float] = []
WINDOW = ARGS.window if ARGS.window is not None else 500  # epochs per aggregation bin

tic = time.perf_counter()
for epoch in range(num_epochs):

    loss = core_train_step()
    mx.eval(mlp_model, optimizer.state)

    # ── cheap per‑step statistics collection (only if requested) ───────
    if (
        OPTIMIZER_SELECTED == "KOURKOUTAS"
        and ARGS.kour_diagnostics
        and ARGS.collect_spikes
    ):
        sps, b2s = optimizer.snapshot_sunspike_history()  # O(#layers)
        buffer_spikes.extend(sps)
        buffer_betas2.extend(b2s)

    # ── end‑of‑window commit & buffer reset ────────────────────────────
    if (epoch + 1) % WINDOW == 0 and ARGS.collect_spikes:
        win_label = epoch + 1  # e.g. 500, 1000, …
        sunspike_dict[win_label] = buffer_spikes[:]  # copy, don’t alias
        betas2_dict[win_label] = buffer_betas2[:]

        buffer_spikes.clear()  # start fresh for next window
        buffer_betas2.clear()

    # ── periodic console feedback ──────────────────────────────────────
    if (epoch + 1) % 500 == 0:
        if OPTIMIZER_SELECTED == "KOURKOUTAS":
            print(
                f"Epoch {epoch + 1:6d} | "
                f"lr={optimizer.learning_rate:.5f} | "
                f"loss={float(loss):.6e} | "
                f"α={float(optimizer.state['alpha']):.2f}"
            )

            if ARGS.kour_diagnostics:
                diags = optimizer.snapshot_diagnostics()
                print(
                    "   ↳ "
                    f"denom_min={diags['diag_denom_min']:.2e} | "
                    f"upd/ρ_max={diags['diag_max_ratio']:.1f} | "
                    f"upd_norm_max={diags['diag_upd_norm_max']:.1e} | "
                    f"v̂_max={diags['diag_vhat_max']:.1e}"
                )
        else:  # e.g. Adam
            print(
                f"Epoch {epoch + 1:6d} | "
                f"lr={optimizer.learning_rate:.5f} | "
                f"loss={float(loss):.6e}"
            )


toc = time.perf_counter()
tpi = (toc - tic) / 60 / num_epochs
print(f"Time per epoch {tpi: .5f} (min)")


# =============================================================================
# 10. Optional Visualization
# =============================================================================
# 8A: Visualize the temperature filed in slices and 3D scatter plot.
# Example usage:
if ARGS.viz:
    from .utils.visualization import evaluate_slice, plot_scatter_3d, plot_slice_2d

    slice_dir = _subdir("slices")  # <── new
    for z in [0.0, 1.0 * r_max, 2.5 * r_max, 5.0 * r_max, 7.5 * r_max, 10.0 * r_max]:
        R, TH, T = evaluate_slice(T_3D, r_min, r_max, z)
        plot_slice_2d(R, TH, T, z_val=z, r_min=r_min, r_max=r_max, outdir=slice_dir)

    plot_scatter_3d(T_3D, r_min, r_max, length_z, N=30, outdir=_subdir("scatter3d"))

# Spike‑plots (paths updated) -------------------------------------------
if ARGS.collect_spikes and OPTIMIZER_SELECTED == "KOURKOUTAS":
    from .utils.plotting import save_density_heatmap, save_violin

    PLOT_STRIDE = ARGS.plot_stride if ARGS.plot_stride is not None else 10 * WINDOW

    save_violin(
        sunspike_dict,
        sample_every=PLOT_STRIDE,
        label="Sunspike",
        outdir=_subdir("sunspike_violin"),
    )
    save_density_heatmap(
        sunspike_dict,
        label="Sunspike",
        outdir=_subdir("sunspike_heatmap"),
        num_bins=20,
        value_range=(0.0, 1.0),
    )

    save_violin(
        betas2_dict,
        sample_every=PLOT_STRIDE,
        label="Beta2",
        outdir=_subdir("beta2_violin"),
    )
    save_density_heatmap(
        betas2_dict,
        label="Beta2",
        outdir=_subdir("beta2_heatmap"),
        num_bins=20,
        value_range=(0.88, 1.0),
    )


# --------------------------------------------------------------------------- #
# 11.  Allow “python pinn3d.py …” execution                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # All top‑level code above has already run, so nothing to do here.
    # Keeping a stub maintains consistency with linting tools expecting
    # an entry‑point guard.
    pass
