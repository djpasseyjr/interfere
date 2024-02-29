"""Tests for benchmarking tools."""

from interfere.benchmarking import (
    DirectionalChangeBinary, RootMeanStandardizedSquaredError
)
import numpy as np


def test_rmsse():
    X_train = np.random.rand(20, 4)
    X_true = np.random.rand(10, 4)
    X_false = np.zeros((10, 4))
    X_pred_good = X_true + 0.01 * np.random.randn(10, 4)
    intervention_idxs = np.array([0])

    rmsse = RootMeanStandardizedSquaredError()
    x_false_err = rmsse(X_train, X_true, X_false, intervention_idxs)
    x_true_err = rmsse(X_train, X_true, X_pred_good, intervention_idxs)
    assert x_false_err > x_true_err


def test_directional():
    X_train = np.random.rand(20, 10)
    X_true = np.random.rand(10, 10)
    X_false = np.zeros((10, 10))
    X_pred_good = X_true
    intervention_idxs = np.array([0])

    dir_change = DirectionalChangeBinary()
    x_false_err = dir_change(X_train, X_true, X_false, intervention_idxs)
    x_true_err = dir_change(X_train, X_true, X_pred_good, intervention_idxs)
    assert x_false_err < x_true_err

