"""confounding_tester.py

A lightweight, dependency‑minimal implementation of **conditional directed
information**–based tests for confounding, plus an automated evaluation
suite.  The numerical routines borrow conceptual ideas from IDTxl (history
embeddings & permutation tests), dit (plug‑in entropy with Miller–Madow
bias correction) and pyphi (exhaustive state enumeration) but are coded
here from scratch using **only NumPy and the Python standard library**.

--------------------------------------------------------------------------
Public API
--------------------------------------------------------------------------
    conditional_directed_information(X, Y, Z=None, *, max_lag=1, n_perm=1000,
                     alpha=0.05, n_bins=8) -> str
        Statistical decision about whether Y confounds the causal effect
        of X and whether a candidate Z removes that confounding.

    run_evaluation(n_runs=200, n_perm=400, alpha=0.05) -> None
        Reproduces four canonical scenarios (direct, confounded, mixed,
        null) and prints confusion‑matrix style metrics.

To use from the command line:
    >>> python confounding_tester.py  # runs evaluation suite

--------------------------------------------------------------------------
Author   : Your Name
Created  : 2025‑04‑21
License  : MIT
"""
from __future__ import annotations

import math
import random
import itertools
from typing import Iterable, Tuple, List, Sequence, Dict

import numpy as np

###############################################################################
# Internal helpers: discretisation & probability models
###############################################################################

def _discretize(data: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile discretise *data* into `n_bins` symbols 0…n_bins‑1.

    Args:
        data: 1‑D array of floats.
        n_bins: Number of discrete symbols.

    Returns:
        1‑D `int64` array of bin indices.
    """
    if data.ndim != 1:
        data = data.ravel()
    # Use numpy.percentile for robust quantile edges
    q_edges = np.percentile(data, np.linspace(0, 100, n_bins + 1))
    # Ensure unique edges to satisfy numpy.digitize
    q_edges[0] -= 1e-12
    symbols = np.digitize(data, q_edges[1:-1])  # returns 0…n_bins‑1
    return symbols.astype(np.int64)


def _miller_madow_entropy(counts: np.ndarray) -> float:
    """Miller–Madow bias‑corrected entropy (base‑2)."""
    counts = counts[counts > 0]
    n = counts.sum()
    probs = counts / n
    h_mle = -(probs * np.log2(probs)).sum()
    k = len(counts)
    correction = (k - 1) / (2 * n * math.log(2))
    return h_mle + correction


def _joint_counts(arrays: Sequence[np.ndarray]) -> Tuple[np.ndarray, Dict[str, int]]:
    """Return joint frequency counts and map each state to linear index."""
    stacked = np.stack(arrays, axis=1)  # shape (N, d)
    # Use structured dtype to allow numpy.unique on rows
    dtype = np.dtype([("f" + str(i), stacked.dtype) for i in range(stacked.shape[1])])
    structured = stacked.view(dtype).squeeze(axis=1)
    unique, inverse, counts = np.unique(structured, return_inverse=True, return_counts=True)
    return counts, {"state_index": inverse}  # inverse maps each row to state id


def _entropy_of(arrays: Sequence[np.ndarray]) -> float:
    counts, _ = _joint_counts(arrays)
    return _miller_madow_entropy(counts)


def _mutual_information(X: np.ndarray, Y: np.ndarray) -> float:
    """Plug‑in MI with Miller–Madow correction."""
    return _entropy_of([X]) + _entropy_of([Y]) - _entropy_of([X, Y])


def _conditional_mutual_information(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
    """CMI I(X;Y|Z) via entropic expansion."""
    h_xz = _entropy_of([X, Z])
    h_yz = _entropy_of([Y, Z])
    h_xyz = _entropy_of([X, Y, Z])
    h_z = _entropy_of([Z])
    return h_xz + h_yz - h_xyz - h_z

###############################################################################
# Directed information estimates (single‑step, discrete)
###############################################################################

def _directed_information(X: np.ndarray, Y: np.ndarray) -> float:
    """Estimate DI = I(Y -> X) using Raginsky’s expression for two scalars.

    We compute:
        I(Y -> X) = H(X|do(Y)) - H(X|Y)
    but in practice we only need KL divergence between P(X|Y) and P(X).
    For a single time step and discrete symbols this collapses to:
        I = I(X;Y)  (because do(Y) removes edges into Y).
    """
    return _mutual_information(X, Y)


def _conditional_directed_information(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
    """Estimate CDI = I(Y -> X | Z) for discrete variables.

    For one‑step scalars CDI equals the conditional mutual information I(X;Y|Z).
    """
    return _conditional_mutual_information(X, Y, Z)

###############################################################################
# Time‑series embedding and permutation surrogate
###############################################################################

def _create_embeddings(series: np.ndarray, max_lag: int) -> List[np.ndarray]:
    """Return a list of lagged versions  [s(t‑1), …, s(t‑max_lag)]."""
    embeds: List[np.ndarray] = []
    for lag in range(1, max_lag + 1):
        embeds.append(series[max_lag - lag: -(lag)])
    return embeds


def _future(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Return the target vector s(t) aligned with embeddings."""
    return series[max_lag:]


def _block_shuffle(series: np.ndarray, block_len: int, rng: random.Random) -> np.ndarray:
    """Permutation surrogate preserving local autocorrelation up to *block_len*.

    Splits the series into non‑overlapping blocks of length *block_len* and
    shuffles the order of the blocks.
    """
    n_blocks = len(series) // block_len
    trimmed = series[: n_blocks * block_len]
    blocks = trimmed.reshape(n_blocks, block_len)
    rng.shuffle(blocks)
    return blocks.reshape(-1)

###############################################################################
# Public function: confounding test
###############################################################################

def conditional_directed_information(
    X: Sequence[float] | np.ndarray,
    Y: Sequence[float] | np.ndarray,
    Z: Sequence[float] | np.ndarray | None = None,
    *,
    max_lag: int = 1,
    n_perm: int = 1000,
    alpha: float = 0.05,
    n_bins: int = 8,
    rng: random.Random | None = None,
) -> str:
    """Test whether *Y* confounds the causal influence of *X* and if *Z* removes it.

    The method discretises the inputs, builds simple fixed‑lag embeddings, and
    computes DI and CDI with plug‑in entropy.  Significance is assessed via
    block‑shuffle surrogates.

    Args:
        X: 1‑D iterable of samples of the *cause* variable.
        Y: 1‑D iterable of samples of the *putative confounder*.
        Z: Optional iterable of samples of an *adjustment* variable.
        max_lag: Number of past steps to include in history embeddings.
        n_perm: Number of permutation surrogates.
        alpha: Significance level.
        n_bins: Number of quantile bins for discretisation.
        rng: Optional `random.Random` instance (for reproducibility).

    Returns:
        Textual decision following four cases:
            • "no_dependence"
            • "confounded"              (DI sig., no Z provided)
            • "blocked_by_Z"            (DI sig., CDI not sig.)
            • "direct_or_unblocked"     (both DI & CDI sig.)
    """
    rng = rng or random.Random()
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Z is not None:
        Z_arr = np.asarray(Z, dtype=float)
    else:
        Z_arr = None

    if not (len(X) == len(Y) and (Z_arr is None or len(Z_arr) == len(X))):
        raise ValueError("All series must share the same length.")

    # Discretise
    X_sym = _discretize(X, n_bins)
    Y_sym = _discretize(Y, n_bins)
    if Z_arr is not None:
        Z_sym = _discretize(Z_arr, n_bins)

    # Build embeddings (simple fixed lag for demonstration)
    emb_Y = _create_embeddings(Y_sym, max_lag)
    emb_X = _create_embeddings(X_sym, max_lag)
    X_future = _future(X_sym, max_lag)

    # Flatten embedding matrices
    if emb_Y:
        Y_hist = np.stack(emb_Y, axis=1)
        X_hist = np.stack(emb_X, axis=1)
    else:
        Y_hist = Y_sym[:-max_lag]
        X_hist = X_sym[:-max_lag]

    # Align arrays
    assert len(X_future) == len(Y_hist) == len(X_hist)
    if Z_arr is not None:
        Z_cur = Z_sym[max_lag:]
        assert len(Z_cur) == len(X_future)

    # Mutual information style vectors (flatten histories to tuples)
    def _to_state_id(mat: np.ndarray) -> np.ndarray:
        if mat.ndim == 1:
            return mat
        # Convert each row to a unique base‑n representation
        base = n_bins
        coeff = base ** np.arange(mat.shape[1])
        return (mat * coeff).sum(axis=1)

    Y_state = _to_state_id(Y_hist)
    X_state = _to_state_id(X_hist)

    # DI uses Y_history as predictor, X_future as target, conditioning on X_history
    DI_obs = _conditional_mutual_information(
        X_future,  # X
        Y_state,   # Y
        X_state,   # Z (conditioning)
    )

    # Surrogate distribution
    block = max_lag + 1
    DI_sur = []
    for _ in range(n_perm):
        Y_perm = _block_shuffle(Y_state, block, rng)
        DI_sur.append(
            _conditional_mutual_information(X_future, Y_perm, X_state)
        )
    p_DI = np.mean(np.array(DI_sur) >= DI_obs)

    if p_DI >= alpha:
        return "no_dependence"
    if Z_arr is None:
        return "confounded"

    # Conditional DI with Z
    CDI_obs = _conditional_mutual_information(
        X_future, Y_state, _to_state_id(np.column_stack([X_state, Z_cur]))
    )
    CDI_sur = []
    Z_block = block  # shuffle jointly with same block size
    for _ in range(n_perm):
        Y_perm = _block_shuffle(Y_state, block, rng)
        CDI_sur.append(
            _conditional_mutual_information(
                X_future,
                Y_perm,
                _to_state_id(np.column_stack([X_state, Z_cur]))
            )
        )
    p_CDI = np.mean(np.array(CDI_sur) >= CDI_obs)
    if p_CDI >= alpha:
        return "blocked_by_Z"
    return "direct_or_unblocked"

###############################################################################
# Evaluation suite – four canonical scenarios
###############################################################################

def _scenario_direct_only(T: int, a: float, *, rng: np.random.Generator):
    Y = rng.normal(size=T)
    X = np.zeros(T)
    for t in range(1, T):
        X[t] = a * Y[t - 1] + rng.normal()
    return X, Y, None


def _scenario_confounded(T: int, b: float, c: float, *, rng: np.random.Generator):
    Z = rng.normal(size=T)
    Y = np.zeros(T)
    X = np.zeros(T)
    for t in range(1, T):
        Y[t] = b * Z[t - 1] + rng.normal()
        X[t] = c * Z[t - 1] + rng.normal()
    return X, Y, Z


def _scenario_mixed(T: int, a: float, b: float, c: float, *, rng: np.random.Generator):
    Z = rng.normal(size=T)
    Y = np.zeros(T)
    X = np.zeros(T)
    for t in range(1, T):
        Y[t] = b * Z[t - 1] + rng.normal()
        X[t] = a * Y[t - 1] + c * Z[t - 1] + rng.normal()
    return X, Y, Z


def _scenario_null(T: int, *, rng: np.random.Generator):
    X = rng.normal(size=T)
    Y = rng.normal(size=T)
    return X, Y, None


def run_evaluation(n_runs: int = 200, n_perm: int = 400, alpha: float = 0.05) -> None:
    """Monte‑Carlo evaluation over four scenarios and confusion metrics."""
    rng = np.random.default_rng(seed=42)
    scenarios = {
        "direct": _scenario_direct_only,
        "confounded": _scenario_confounded,
        "mixed": _scenario_mixed,
        "null": _scenario_null,
    }
    counts = {name: {label: 0 for label in ("no_dependence", "confounded",
                                            "blocked_by_Z", "direct_or_unblocked")}
              for name in scenarios}

    for name, generator in scenarios.items():
        for _ in range(n_runs):
            if name == "direct":
                X, Y, Z = generator(800, a=0.9, rng=rng)
            elif name == "confounded":
                X, Y, Z = generator(800, b=0.9, c=0.9, rng=rng)
            elif name == "mixed":
                X, Y, Z = generator(800, a=0.7, b=0.5, c=0.5, rng=rng)
            else:  # null
                X, Y, Z = generator(800, rng=rng)

            decision = conditional_directed_information(
                X, Y, Z,
                max_lag=1,
                n_perm=n_perm,
                alpha=alpha,
                n_bins=8,
                rng=random.Random(rng.integers(1 << 30)),
            )
            counts[name][decision] += 1

    # Pretty print metrics
    for name, tally in counts.items():
        total = sum(tally.values())
        print(f"\nScenario: {name} ({total} runs)")
        for decision, k in tally.items():
            print(f"  {decision:18s}: {k:4d}  ({k / total:4.1%})")

###############################################################################
# Command‑line entry point
###############################################################################

if __name__ == "__main__":
    run_evaluation()