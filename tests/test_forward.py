# tests/test_forward.py
"""
Smoke‑test that a very small MLP built with MLX can execute a forward pass
and produce finite output values.  No optimiser, no training loop – just
confirms that the MLX runtime is working in the CI environment.

Run with:  pytest -q
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class TinyMLP(nn.Module):
    """3-layer perceptron: 3 → 8 → 8 → 1."""
    def __init__(self) -> None:
        super().__init__()
        self.layers = [
            nn.Linear(3, 8),
            nn.Linear(8, 8),
            nn.Linear(8, 1),
        ]

    def __call__(self, x: mx.array) -> mx.array:  # noqa: D401
        for layer in self.layers[:-1]:
            x = nn.silu(layer(x))
        return self.layers[-1](x)


def test_forward_pass() -> None:
    """Model(x) produces the expected shape and finite values (no NaNs/Infs)."""
    model = TinyMLP()
    x = mx.random.uniform(-1.0, 1.0, (4, 3))  # 4 sample points, 3 features
    y = model(x)

    # Shape check
    assert y.shape == (4, 1), "Unexpected output shape"

    # Robust finiteness check (NumPy fallback covers all MLX versions)
    y_np = np.array(y)
    assert np.isfinite(y_np).all(), "Output contains NaNs or Infs"
