"""
===============================================================================
Plotting utilities for kbeta‑pinn3d
-------------------------------------------------------------------------------
Author   : Stavros Kassinos  <kassinos.stavros@ucy.ac.cy>
Revision : v0.1.0    (August 2025)

CONTENTS ----------------------------------------------------------------------
• save_violin(...)                – violin + swarm + baseline + medians
• save_density_heatmap(...)       – linear‑scale epoch × value heat‑map
• save_density_heatmap_logscale() – log‑scale variant

DEPENDENCIES ---------------------------------------------------------------  
These helpers are **lazy‑imported** inside each function so that users who run
the PINN without the `[viz]` extra never pay the Matplotlib / Seaborn penalty.  
Missing‑stack errors are caught and re‑raised with a friendly hint:

    pip install kbeta-pinn3d[viz]

LICENCE -----------------------------------------------------------------------
MIT © 2025 Stavros Kassinos  – see project‑level LICENSE.
"""
# Allow re‑exports via «from plotting import *»
__all__ = [
    "save_violin",
    "save_density_heatmap",
    "save_density_heatmap_logscale",
]

def save_violin(
        values_dict,
        *,
        label="Beta2",
        outdir="./violin_plots",
        sample_every=5,
        baseline_value=0.999,
        baseline_label="Adam β₂ = 0.999",
        show_medians=True,
):
    """
    Violin plot of per‑epoch distributions with baseline & median overlay.
    """
    import os
    
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ModuleNotFoundError as e:          # pragma: no cover
        raise ImportError(
            "Optional visualisation stack missing or mis‑configured."
            "Run  pip install kbeta-pinn3d[viz]  to enable --viz features."
        ) from e

    os.makedirs(outdir, exist_ok=True)

    # ── build tidy dataframe ────────────────────────────────
    rows = [
        (epoch, float(v))
        for epoch, vals in values_dict.items()
        if epoch % sample_every == 0
        for v in vals
    ]
    if not rows:
        print("Nothing to plot – check sample_every / values_dict.")
        return

    df = pd.DataFrame(rows, columns=["epoch", "value"])
    df["epoch"] = df["epoch"].astype(str)  # treat as discrete categories

    # compute category order once
    order = sorted(df["epoch"].unique(), key=int)
    print(order)

    # ── figure & violin ────────────────────────────────────
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(
        data=df,
        x="epoch",
        y="value",
        color="#8ecae6",
        order=order,
        inner=None,
        linewidth=1.2,
        cut=0,
    )
    sns.swarmplot(
        data=df,
        x="epoch",
        y="value",
        color="k",
        alpha=0.3,
        size=1,
        ax=ax,
    )

    # ── baseline line ──────────────────────────────────────
    if baseline_value is not None:
        ax.axhline(
            baseline_value,
            ls="--",
            lw=1.0,
            color="red",
            label=baseline_label,
            zorder=3,
        )
        ax.legend(
            title="", frameon=False, handlelength=1.2,
            loc="upper right", borderpad=0.2,
        )

    # ── median overlay ─────────────────────────────────────
    if show_medians:
        med = df.groupby("epoch")["value"].median().reindex(order)
        ax.scatter(order, med.values, s=30, color="white", edgecolor="black", zorder=4)
        ax.plot(order, med.values, color="black", lw=1, alpha=0.6, zorder=4)

    # ── axes cosmetics ─────────────────────────────────────
    ymin, ymax = df["value"].min(), df["value"].max()
    if baseline_value is not None:
        ymax = max(ymax, baseline_value)
        ymin = min(ymin, baseline_value)
    pad = 0.02 * (ymax - ymin if ymax > ymin else 1.0)
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_title(f"{label} Distribution per Epoch (Violin Plot)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    ax.legend(frameon=False, loc="upper right")

    plt.tight_layout()
    outfile = os.path.join(outdir, f"{label.lower()}_violin_plot.png")
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Violin plot saved to {outfile}")



def save_density_heatmap(values_dict, label="Sunspike",
                                      num_bins=50, value_range=(0.0, 1.0),
                                      outdir="./density_heatmap"):
    """
    Creates and saves a 2D heatmap, where the y-axis is epoch and the x-axis is
    bins of the distribution (e.g. sunspike or beta2). The color indicates how
    many values fell into each bin at that epoch.

    Parameters
    ----------
    values_dict : dict
        A dictionary mapping epoch -> list of values (e.g. sunspike or beta2)
        observed in that epoch. For example:
            { 1: [0.01, 0.02, ...],
              2: [0.015, 0.03, ...],
              ... }
    label : str
        A short name/label for the distribution, e.g. "Sunspike" or "Beta2".
    num_bins : int
        Number of bins to use for the distribution axis.
    value_range : tuple
        The (min, max) range for the distribution values, e.g. (0.0, 1.0).
    outdir : str
        Directory to save the output heatmap image.

    Returns
    -------
    None
    """

    import os

    import numpy as np
    
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:          # pragma: no cover
        raise ImportError(
            "Optional visualisation stack missing or mis‑configured."
            "Run  pip install kbeta-pinn3d[viz]  to enable --viz features."
        ) from e

    os.makedirs(outdir, exist_ok=True)

    # Sort epochs so we iterate in ascending order
    all_epochs = sorted(values_dict.keys())
    if not all_epochs:
        print("No data to plot in values_dict.")
        return

    # Prepare a 2D array for histogram counts:
    # rows = number of epochs, columns = num_bins
    epoch_hist = np.zeros((len(all_epochs), num_bins), dtype=np.float32)

    bin_edges = np.linspace(value_range[0], value_range[1], num_bins + 1)

    for i, epoch in enumerate(all_epochs):
        vals = np.array(values_dict[epoch], dtype=np.float32)
        counts, _ = np.histogram(vals, bins=bin_edges)
        epoch_hist[i, :] = counts

    plt.figure(figsize=(10, 6))

    # 'origin=lower' => smallest epoch at the bottom
    plt.imshow(
        epoch_hist,
        extent=[value_range[0], value_range[1], all_epochs[0], all_epochs[-1]],
        aspect='auto',
        origin='lower',
        vmin=0.90,
        cmap='magma'  # 'viridis' #'magma'  # or 'viridis', 'plasma', etc.
    )
    plt.colorbar(label="Count")

    plt.xlabel(label)
    plt.ylabel("Epoch")
    plt.title(f"Density Heatmap of {label} by Epoch")

    # Optionally, invert the y-axis if you want epoch=1 on top:
    # plt.gca().invert_yaxis()

    outfile = os.path.join(outdir, f"{label.lower()}_density_heatmap.png")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Density heatmap saved to {outfile}")



def save_density_heatmap_logscale(
    values_dict,
    label="Sunspike",
    num_bins=10,
    value_range=(0.0, 1.0),
    outdir="./density_heatmap_log",
):
    """
    Like save_distribution_density_heatmap, but uses a log-scale colormap to
    make lower counts more visible. Bins with 0 remain invisible, but
    bins with count=1 or 2 now show up distinctly instead of blending in
    with the color for zero.
    """
    
    import os

    import numpy as np
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ModuleNotFoundError as e:          # pragma: no cover
        raise ImportError(
            "Optional visualisation stack missing or mis‑configured."
            "Run  pip install kbeta-pinn3d[viz]  to enable --viz features."
        ) from e
    
    
    os.makedirs(outdir, exist_ok=True)
    all_epochs = sorted(values_dict.keys())
    if not all_epochs:
        print("No data to plot in values_dict.")
        return

    epoch_hist = np.zeros((len(all_epochs), num_bins), dtype=np.float32)
    bin_edges = np.linspace(value_range[0], value_range[1], num_bins + 1)

    for i, epoch in enumerate(all_epochs):
        vals = np.array(values_dict[epoch], dtype=np.float32)
        counts, _ = np.histogram(vals, bins=bin_edges)
        epoch_hist[i, :] = counts

    # We add +1 to avoid log(0). Then LogNorm can show zero-count cells as black.
    epoch_hist_log = epoch_hist + 1.0

    plt.figure(figsize=(10, 6))
    plt.imshow(
        epoch_hist_log,
        extent=[value_range[0], value_range[1], all_epochs[0], all_epochs[-1]],
        aspect='auto',
        origin='lower',
        cmap='plasma',
        norm=LogNorm(),
    )
    plt.colorbar(label="Log-scaled Count")
    plt.xlabel(label)
    plt.ylabel("Epoch")
    plt.title(f"Log-Scale Density Heatmap of {label} by Epoch")

    outfile = os.path.join(outdir, f"{label.lower()}_density_heatmap_log.png")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Log-scale density heatmap saved to {outfile}")

