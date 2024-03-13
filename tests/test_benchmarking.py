"""Tests for benchmarking tools."""

from interfere.benchmarking import (
    DirectionalChangeBinary,
    RootMeanStandardizedSquaredError,
    TTestDirectionalChangeAccuracy
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


def test_ttest_directional():

    ttest_dir_acc = TTestDirectionalChangeAccuracy()
    rng = np.random.default_rng(11)
    dim = 5
    nsamp = 300
    p_cut = 0.05

    # Check for X means bigger than Y means
    X = rng.random((nsamp, dim)) + 0.33
    Y = rng.random((nsamp, dim))
    estim_chng = ttest_dir_acc.directional_change(X, Y, p_cut)
    assert np.all(estim_chng == 1)

    # Check for X means smaller than Y means
    X = rng.random((nsamp, dim)) - 0.33
    Y = rng.random((nsamp, dim))
    estim_chng = ttest_dir_acc.directional_change(X, Y, p_cut)
    assert np.all(estim_chng == -1)

    # Check for X means same as Y means
    X = rng.random((nsamp, dim))
    Y = rng.random((nsamp, dim))
    estim_chng = ttest_dir_acc.directional_change(X, Y, p_cut)
    assert np.all(estim_chng == 0)

    # Check for mix of bigger, smaller and the same
    X = rng.random((nsamp, dim))
    Y = rng.random((nsamp, dim))
    X[:, :2] += 0.3
    X[:, 2:6] -= 0.3

    estim_chng = ttest_dir_acc.directional_change(X, Y, p_cut)
    assert np.all(estim_chng[:2] == 1)
    assert np.all(estim_chng[2:6] == -1)
    assert np.all(estim_chng[6:] == 0)