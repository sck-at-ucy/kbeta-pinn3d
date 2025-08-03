# 3‑D Cylindrical PINN for Steady Heat Conduction

> **Author**  Stavros Kassinos · University of Cyprus  
> **Version** `v1.0` (February 2025)  
> **Framework** [MLX](https://github.com/ml-explore/mlx) (GPU / Apple Silicon)

---

## 1 . Problem statement

We solve the steady‑state heat‑diffusion (Laplace) equation  

\[
\nabla^{2} T = 0
\]

inside a **distorted cylindrical domain** in \((r, \theta, z)\) with mixed boundary
conditions:

| Location | Boundary condition |
|----------|--------------------|
| **Inner cylinder** \(r=r_{\min}\) | Dirichlet: \(T=1\) |
| **Inlet plane** \(z=0\) | Dirichlet: \(T=1\) |
| **Outlet plane** \(z=L_z\) | Neumann: \(\partial T/\partial z = 0\) |
| **Outer wall** \(r=r_{\max}+0.25\,r_{\max}\sin 3\theta\) | piece‑wise flux |
| **Azimuth** \(\theta\) | \(2\pi\)-periodic \(T,\partial T/\partial\theta\) |

A Physics‑Informed Neural Network (PINN) enforces the PDE residual at
interior collocation points and the boundary/periodicity constraints
at dedicated surface points.

---

## 2 . Repository layout

```
.
├── pinn/
│   ├── train_heat_3d_Asilomar.py    ← main training script
│   └── utils/
│       ├── plotting.py              ← violin & heat‑map helpers
│       └── visualization.py         ← 2‑D/3‑D field plots (optional)
├── kourkoutas/
│   └── Kourkoutas_optimizer.py      ← Adam‑style optimiser w/ sun‑spike β₂
└── README.md
```

---

## 3 . Quick start

### 3.1 Prerequisites

* Python ≥ 3.11  
* **MLX** (GPU build)  
  ```bash
  pip install mlx
  ```
* Plotting stack (optional)  
  ```bash
  pip install matplotlib seaborn pandas
  ```

### 3.2 Run a short training job

```bash
# vanilla Adam, 2 000 epochs
python -m pinn.train_heat_3d_Asilomar        --optimizer adam        --epochs 2000        --viz
```

```bash
# Kourkoutas optimiser, diagnostics & spike collection every 500 epochs
python -m pinn.train_heat_3d_Asilomar        --optimizer kourkoutas        --epochs 20000        --viz        --kour_diagnostics        --collect_spikes
```

Outputs:

```
Epoch  500 | lr=0.009996 | loss=1.46e-01 | α=0.93
   ↳ denom_min=2.10e-03 | upd/ρ_max=2.5 | upd_norm_max=1.2e-02 | v̂_max=4.7e-03
...
Time per epoch 0.0016 min
plots/
 ├─ sunspike_violin/
 ├─ sunspike_heatmap/
 ├─ beta2_violin/
 └─ beta2_heatmap/
```

---

## 4 . Command‑line interface

| Flag | Default | Description |
|------|---------|-------------|
| `--optimizer {adam,kourkoutas}` | `kourkoutas` | Choose optimiser |
| `--epochs N` | `6000` | Number of training iterations |
| `--seed N` | `0` | Model & numpy random seed (collocation mesh is fixed) |
| `--viz` | *(off)* | Generate slice/3‑D field plots after training |
| `--kour_diagnostics` | *(off)* | Enable lightweight per‑epoch diagnostics (≈ 2 % cost) |
| `--collect_spikes` | *(off)* | Store **sun‑spike**/β₂ history → violin & heat‑maps |

### Windowed spike sampling

Diagnostics are *always* computed in‑kernel when `--kour_diagnostics` is on,
but spike/β₂ values are only pushed to the history buffers every `WINDOW`
epochs (default = 500).  
You can change this by editing `WINDOW` near the top of
`train_heat_3d_Asilomar.py`.

---

## 5 . Visualisation utilities

All routines live in **`pinn/utils/plotting.py`** and take plain Python
dicts:

```python
from utils.plotting import save_violin, save_density_heatmap

save_violin(sunspike_dict, sample_every=5000,
            label="Sunspike", outdir="plots/sunspike_violin")

save_density_heatmap(betas2_dict,
                     label="Beta2",
                     outdir="plots/beta2_heatmap",
                     num_bins=20, value_range=(0.88,1.0))
```

Field visualisation (2‑D slices, stacked 3‑D surfaces, scatter) is kept
in **`utils/visualization.py`** and is triggered automatically by `--viz`.

---

## 6 . Performance tips

* Keep **`--kour_diagnostics`** off for production runs — path traces show
  < 2 % overhead but the absolute fastest path is still without diagnostics.
* Reduce the collocation sizes (`num_interior`, `num_boundary`) during
  hyper‑parameter sweeps.
* MLX auto‑compiles the training step; avoid modifying the
  `@mx.compile` decorated function inside tight loops.

---

## 7 . License

[MIT](LICENSE) © 2025 Stavros Kassinos

---

## 8 . Citation

If you use this code in academic work, please cite:

```
@software{Kassinos_PINN_2025,
  author  = {Stavros Kassinos},
  title   = {3‑D Cylindrical PINN for Heat Conduction},
  year    = {2025},
  version = {v1.0},
  url     = {https://github.com/<your‑repo>}
}
```
