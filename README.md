[![CI (macOS arm64)](https://github.com/sck-at-ucy/kbeta-pinn3d/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sck-at-ucy/kbeta-pinn3d/actions/workflows/ci.yml)

| Branch | Status |
|--------|--------|
| `main` | ![CI‑main](https://github.com/sck-at-ucy/kbeta-pinn3d/actions/workflows/ci.yml/badge.svg?branch=main) |
| `dev`  | ![CI‑dev](https://github.com/sck-at-ucy/kbeta-pinn3d/actions/workflows/ci.yml/badge.svg?branch=dev)  |

# kbeta‑pinn3d – *A 3‑D Cylindrical Physics‑Informed Neural Network powered by Kourkoutas‑β*  🌞🦎🧊📐

> Research companion code for our upcoming paper  
> **“Kourkoutas‑β: Soft‑max Momentum with Adaptive Variance for Mesh‑Accelerated Deep Learning.”**  
> This repository contains the **3‑D steady‑heat PINN** workload that showcases the optimiser on a complex mixed‑boundary problem.

[Download this README](https://raw.githubusercontent.com/sck-at-ucy/kbeta-pinn3d/main/README.md?download=1)

---

## Table of Contents
1. [Why a 3‑D PINN?](#why-a-3-d-pinn)
2. [Model highlights](#model-highlights)
3. [Project layout](#project-layout)
4. [Quick start](#quick-start)
5. [Installation options](#installation-options)
6. [Training from scratch](#training-from-scratch)
7. [Using your own geometry](#using-your-own-geometry)
8. [Tests and linting](#tests-and-linting)
9. [CLI options](#cli-options)
10. [Relation to Kourkoutas‑β](#relation-to-kourkoutas-β)
11. [Citation](#citation)
12. [License](#license)

---

## Why a 3‑D PINN?
Classical ML benchmarks rarely **stress second‑moment tracking** because their
loss landscapes are well‑conditioned.  
The **cylindrical PINN** provides:

* Extreme scale separation (inner vs outer radius & long aspect‑ratio $begin:math:text$L_z/r$end:math:text$).  
* **Piece‑wise flux** & Neumann edges that provoke gradient spikes.  
* A moderate parameter budget (≈ 200 k) → runs on a single Apple‑GPU in < 30 min.

This makes it an *excellent* stress‑test for Kourkoutas‑β’s adaptive β₂ logic.

---

## Model highlights
| Feature | What it means | Why it matters |
|---------|---------------|----------------|
| **Cylindrical Laplacian** coded *analytically* | No autodiff‑derived PDE residual – we write the terms explicitly. | keeps compute graph tiny; MLX JIT can fuse the custom op. |
| **Mixed BCs** (Dirichlet, Neumann, flux) | Complex outer wall $begin:math:text$r=r_{\\max}+0.25\\,r_{\\max}\\sin3\\theta$end:math:text$. | amplifies gradient variance → showcases optimiser behaviour. |
| **Periodic θ coupling** | Enforces both $begin:math:text$T$end:math:text$ and $begin:math:text$\\partial T/\\partial\\theta$end:math:text$. | avoids mesh duplication; tests multi‑loss balancing. |
| **Spike/β₂ tracking hooks** built‑in | 1‑line opt‑in via `--collect_spikes`. | produces violin & density plots (see *plot_utils*). |
| **Pure‑MLX implementation** | Runs out‑of‑the‑box on Apple Silicon (`pip install mlx`). | zero PyTorch/TensorFlow deps. |

---

## Project layout
```
kbeta-pinn3d
├── src/kbeta_pinn3d/
│   ├── __init__.py          # exposes `pinn3d.main`
│   ├── pinn3d.py            # monolithic training script
│   └── utils/
│       ├── plotting.py      # sun‑spike, β₂ violin / heat‑maps
│       └── visualization.py # 2‑D slice & 3‑D scatter helpers
├── tests/
│   ├── test_imports.py      # import smoke test
│   └── test_forward.py      # tiny forward pass
├── .github/workflows/ci.yml # macOS‑14 MLX CI
└── pyproject.toml
```

---

## Quick start
```bash
git clone git@github.com:sck-at-ucy/kbeta-pinn3d.git
cd kbeta-pinn3d
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"   # installs MLX & plotting stack

# 1‑minute smoke run (2 000 epochs, Adam‑95)
python -m kbeta_pinn3d.pinn3d        --optimizer adam95        --epochs 2000        --viz
```

---

## Installation options
Choose the *extra* set that best fits your workflow:

* **Bare‑bones command‑line only**

  ```bash
  pip install kbeta-pinn3d
  ```

* **Add visualisation stack — matplotlib, seaborn, pandas**

  ```bash
  pip install kbeta-pinn3d[viz]
  ```

* **Add Developer tools — pytest, ruff, mypy, pre‑commit hooks**

  ```bash
  pip install kbeta-pinn3d[dev]
  ```
  
* **Everything (viz + dev)**

  ```bash
  pip install kbeta-pinn3d[viz,dev]
  ```  
  
Tip: working from a local clone?
Activate your virtual‑env and run
  ```bash
  pip install -e ".[viz,dev]"
  ```  
to get an editable install with the full extra set.  

---

## Training from scratch
```bash
# Full experiment (20 k epochs) with Kourkoutas‑β + diagnostics
python -m kbeta_pinn3d.pinn3d \
       --optimizer kourkoutas \
       --epochs    20000      \
       --kour_diagnostics     \
       --collect_spikes       \
       --viz
```
Output directories:

```
plots/
 ├─ sunspike_violin/   *.png
 ├─ sunspike_heatmap/  *.png
 ├─ beta2_violin/      *.png
 ├─ beta2_heatmap/     *.png
 └─ fields/            slice_Z=0.0.png, ...
```

---

## Using your own geometry
All geometry & sampling constants sit at the **top of `pinn3d.py`**:
```python
r_min     = 0.2          # inner radius
r_max     = 1.0          # mean outer radius
length_z  = 10.0 * r_max # cylinder length
num_interior = 4000
num_boundary = 2000
```
Change them, re‑run, done.  
Boundary helpers (`piecewise_flux`, `compute_normal_derivative_3D_outer`) are
stand‑alone functions; swap in your own.

---

## Tests and linting
```bash
pytest -q            # should print ‘2 passed’
ruff check .         # style / import / naming
mypy src             # optional static typing pass
```

CI enforces all of the above on **macOS‑14 (arm64)** runners.

---

## CLI options 
Overriding defaults:

| Flag | Default | Purpose |
|------|---------|---------|
| `--optimizer {adam95, adam999, kourkoutas}` | `kourkoutas` | Select the optimiser. |
| `--epochs N` | `6000` | Number of training iterations. |
| `--seed N` | `0` | Random seed for **both** mesh & model init. |
| `--viz` | *(off)* | Produce 2‑D/3‑D field plots after training. |
| `--kour_diagnostics` | *(off)* | Collect lightweight per‑epoch diagnostics (≈ 2 % overhead). |
| `--collect_spikes` | *(off)* | Store **sun‑spike**/β₂ history for violin & heat‑maps. |

Example runs

# Adam with β₂ = 0.95 for 2 k epochs + field plots
```bash
python -m kbeta_pinn3d.pinn3d --optimizer adam95 --epochs 2000 --viz
```

---

# Full 100 k‑epoch paper run with Kourkoutas‑β diagnostics & spike plots
```bash
python -m kbeta_pinn3d.pinn3d \
       --optimizer kourkoutas \
       --epochs    100000      \
       --kour_diagnostics     \
       --collect_spikes       \
       --viz
```

## Relation to Kourkoutas‑β
This workload is the **PDE‑heavy sibling** to the 2‑D Transformer demo in
[`kbeta-transformer2d`](https://github.com/sck-at-ucy/kbeta-transformer2d).  
Both share the same optimiser code (`kbeta.optim.KourkoutasSoftmaxFlex`) but
stress *different* regimes:

| Repo | Regime tested |
|------|---------------|
| `transformer2d` | Dense **autoregressive** gradients, low wall‑clock per step |
| `pinn3d` | Sparse **PDE‑residual** gradients, high variance, complex BCs |

---

## Citation
```
@software{Kassinos_PINN_2025,
  author  = {Stavros Kassinos},
  title   = {3‑D Cylindrical PINN for Heat Conduction},
  year    = {2025},
  version = {v1.0},
  url     = {https://github.com/sck-at-ucy/kbeta-pinn3d}
}
```

---

## License
Distributed under the MIT License – see [`LICENSE`](LICENSE) for full text.

Happy diffusing 🔥🌀❄️




