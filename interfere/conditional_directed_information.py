"""causal_information.py

Implements an estimator for *conditional directed information*

    D_{KL}( P(Y|X) \parallel P(Y|do(X=x_0)) \mid P_X )

based on multidimensional observational and interventional samples.

The estimator supports **multivariate** (vector‑valued) variables `X` and
`Y` using \ :pyfunc:`numpy.histogramdd` to build empirical joint
probability mass functions.

Example
-------
>>> import numpy as np
>>> from conditional_directed_information import directed_information
>>> rng = np.random.default_rng(0)
>>> # 2‑dimensional X, 1‑dimensional Y
>>> x_obs = rng.normal(size=(10_000, 2))
>>> y_obs = (x_obs @ np.array([1.2, -0.5]))[:, None] + rng.normal(scale=0.5, size=(10_000, 1))
>>> x0 = np.array([[2.0, -1.0]])                       # intervention value
>>> y_do = (x0 @ np.array([1.2, -0.5]))[:, None] + rng.normal(scale=0.5, size=(5_000, 1))
>>> di_bits = directed_information(x_obs, y_obs, y_do)
>>> print(f"DI ≈ {di_bits:.3f} bits")
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def directed_information(
    x_obs: ArrayLike,
    y_obs: ArrayLike,
    y_do: ArrayLike,
    bins_x: Union[int, Sequence[int], str] = "fd",
    bins_y: Union[int, Sequence[int], str] = "fd",
    pseudocount: float = 1e-12,
    log_base: float = 2.0,
) -> float:
    """Estimate conditional directed information from samples.

    Args:
        x_obs: Observational samples of X with shape (n_samples, d_x)
        y_obs: Observational samples of Y with shape (n_samples, d_y)
        y_do: Interventional samples of Y with shape (n_samples, d_y)
        bins_x: Histogram rule or explicit bin counts for X dimensions
        bins_y: Histogram rule or explicit bin counts for Y dimensions
        pseudocount: Small value added to avoid zeros
        log_base: Base for logarithm in KL divergence calculation

    Returns:
        Estimated conditional directed information
    """
    # Validate and shape inputs
    x_obs_arr: NDArray[np.floating] = _to_2d(np.asarray(x_obs, dtype=float))
    y_obs_arr: NDArray[np.floating] = _to_2d(np.asarray(y_obs, dtype=float))
    y_do_arr: NDArray[np.floating] = _to_2d(np.asarray(y_do, dtype=float))

    if x_obs_arr.shape[0] != y_obs_arr.shape[0]:
        raise ValueError("x_obs and y_obs must have the same number of rows.")

    # Build joint histogram for (X, Y) under observational regime
    edges_x = _bin_edges_per_dim(x_obs_arr, bins_x)
    edges_y = _bin_edges_per_dim(y_obs_arr, bins_y)

    joint_counts, _ = np.histogramdd(
        np.hstack([x_obs_arr, y_obs_arr]),
        bins=edges_x + edges_y,
    )
    joint_p = (joint_counts + pseudocount).astype(float)
    joint_p /= joint_p.sum()

    # Marginal P_X: sum over Y dimensions
    d_y = y_obs_arr.shape[1]
    p_x = joint_p.sum(axis=tuple(range(-d_y, 0)))

    # Conditional P(Y|X): divide along broadcasted axes
    with np.errstate(divide="ignore", invalid="ignore"):
        p_y_given_x = joint_p / p_x[(...,) + (None,) * d_y]
    p_y_given_x = np.nan_to_num(p_y_given_x)

    # Histogram for Y under intervention do(X=x0)
    counts_y_do, _ = np.histogramdd(y_do_arr, bins=edges_y)
    p_y_do = (counts_y_do + pseudocount).astype(float)
    p_y_do /= p_y_do.sum()

    # Flatten distributions for KL calculation
    n_x_bins = p_x.size
    n_y_bins = p_y_do.size

    p_x_flat = p_x.reshape(n_x_bins)
    p_y_do_flat = p_y_do.reshape(n_y_bins)
    p_y_given_x_flat = p_y_given_x.reshape(n_x_bins, n_y_bins)

    # Calculate KL divergence for each X value
    kl_per_x = np.zeros(n_x_bins)
    for i in range(n_x_bins):
        p = p_y_given_x_flat[i]
        q = p_y_do_flat
        mask = p > 0
        if mask.any():
            kl_per_x[i] = np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask]))) / np.log(log_base)

    # Weight by P(X) and sum
    di = float(np.sum(p_x_flat * kl_per_x))
    return di

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _to_2d(arr: NDArray[np.floating]) -> NDArray[np.floating]:
    """Ensure array is 2‑D with shape (n_samples, n_features)."""
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    raise ValueError("Input array must be 1‑D or 2‑D.")


def _auto_bin_count(data: NDArray[np.floating], rule: str) -> int:
    """Return automatic bin count for 1‑D data using rule."""
    n = data.size

    if rule == "sqrt":
        return max(1, int(np.ceil(np.sqrt(n))))
    if rule == "sturges":
        return max(1, int(np.ceil(np.log2(n) + 1)))
    if rule == "fd":
        iqr = np.subtract(*np.percentile(data, [75, 25]))
        if iqr == 0:
            iqr = np.std(data) * (4 / (3 * n)) ** 0.2
        bin_width = 2 * iqr / np.cbrt(n)
        return max(1, int(np.ceil((data.max() - data.min()) / bin_width)))
    raise ValueError(f"Unknown binning rule '{rule}'.")


def _bin_edges_per_dim(
    data: NDArray[np.floating],
    bins: Union[int, Sequence[int], str],
) -> List[NDArray[np.floating]]:
    """Compute histogram bin edges for each dimension."""
    d = data.shape[1]

    # Resolve bin counts per dimension
    if isinstance(bins, int):
        counts = [bins] * d
    elif isinstance(bins, str):
        counts = [_auto_bin_count(data[:, i], bins) for i in range(d)]
    else:
        if len(bins) != d:
            raise ValueError("Length of bins sequence must match dimensionality.")
        counts = list(bins)

    edges: List[NDArray[np.floating]] = []
    for i in range(d):
        edges_i = np.histogram_bin_edges(data[:, i], bins=counts[i])
        edges.append(edges_i)
    return edges