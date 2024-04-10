"""Tests for the functions in utils.py"""

import interfere
import numpy as np


def test_sktime_transforms():

    x = np.arange(0, 100)
    ts = [x / 1000, x / 10, x * 100]
    
    for t in ts:
        X = np.random.rand(100, 5)

        # Forecasting horizon
        fh = interfere.utils.to_forecasting_horizon(t)
        assert np.allclose(t, interfere.utils.fh_to_seconds(fh))

        # Time series
        X_skt = interfere.utils.to_sktime_time_series(t, X)
        t_inter, X_inter = interfere.utils.to_interfere_time_series(X_skt)
        assert np.allclose(t, t_inter)
        assert np.allclose(X, X_inter)




