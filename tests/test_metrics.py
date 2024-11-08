"""Tests for metrics."""

from interfere.metrics import (
    DirectionalChangeBinary,
    RootMeanStandardizedSquaredError,
    TTestDirectionalChangeAccuracy,
    ValidPredictionTime,
    RootMeanSquaredScaledErrorOverAvgMethod
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


def test_vpt_all_idxs():
    vpt = ValidPredictionTime()
    X_true = np.random.rand(10, 2)
    assert vpt(None, X_true, X_true, []) == 10
    assert vpt(None, X_true, X_true, [0]) == 10


def test_vpt_middle_idxs():
    vpt = ValidPredictionTime()
    X_true = np.random.rand(10, 2)
    X_pred = X_true.copy()
    X_pred[5:, :] += 1
    assert vpt(None, X_true, X_pred, []) == 5
    assert vpt(None, X_true, X_pred, [0]) == 5


def test_vpt_first_idx():
    vpt = ValidPredictionTime()
    X_true = np.random.rand(10, 2)
    X_pred = X_true.copy()
    X_pred += 1
    assert vpt(None, X_true, X_pred, []) == 0
    assert vpt(None, X_true, X_pred, [0]) == 0


def test_rmsse_over_avg():
    X_train = np.random.rand(20, 4)
    X_true = np.random.rand(10, 4)
    X_false = np.zeros((10, 4))
    X_pred_good = X_true + 0.01 * np.random.randn(10, 4)
    intervention_idxs = np.array([0])

    rmsse_over_avg = RootMeanSquaredScaledErrorOverAvgMethod()
    x_false_err = rmsse_over_avg(X_train, X_true, X_false, intervention_idxs)
    x_true_err = rmsse_over_avg(X_train, X_true, X_pred_good, intervention_idxs)
    assert x_false_err > x_true_err

    X_mean_pred = np.vstack([
        np.mean(X_train, axis=0) for i in range(X_true.shape[0])])
    assert 1.0 == rmsse_over_avg(X_train, X_true, X_mean_pred, intervention_idxs)