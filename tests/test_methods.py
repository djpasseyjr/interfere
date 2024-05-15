import interfere
from interfere.methods import (
    BaseInferenceMethod, ResComp, AverageMethod
)
import numpy as np

SEED = 11

ALL_METHODS = [ResComp, AverageMethod]


def VARIMA_timeseries(dim=4, lags=3, noise_lags=2, tsteps=200):
    # Initialize a VARMA model
    rng = np.random.default_rng(SEED)
    phis = [0.5 * (rng.random((dim, dim)) - 0.5) for i in range(lags)]
    thetas = [0.5 * (rng.random((dim, dim)) - 0.5) for i in range(noise_lags)]
    sigma = rng.random((dim, dim))
    sigma += sigma.T
    model = interfere.dynamics.VARMA_Dynamics(phis, [], sigma)

    # Generate a time series
    t = np.arange(tsteps)
    x0 = rng.random((dim, lags))
    X = model.simulate(x0, t, rng=rng)

    n_do = 100
    intervention = interfere.PerfectIntervention(1, -0.5)
    historic_times = t[:n_do]
    forecast_times = t[n_do:]
    X_historic = X[:n_do, :]
    x0_do = X_historic[-1, :]

    X_do = model.simulate(x0_do, forecast_times, intervention, rng=rng)

    return X_historic, historic_times, X_do, forecast_times, intervention


def standard_inference_method_checks(method_type: BaseInferenceMethod):
    """Ensures that a method has all of the necessary functionality.
    """
    (X_historic, historic_times, X_do, 
     forecast_times, intervention) = VARIMA_timeseries()

    # Access test parameters.
    method_params = method_type.get_test_params()
    param_grid = method_type.get_test_param_grid()

    # Initialize method and tune.
    endo, exog = intervention.split_exogeneous(X_historic)

    best_method, gs_results = interfere.benchmarking.grid_search(
        method_type, method_params, param_grid, endo, historic_times, exog)
    
    X_do_pred = best_method.simulate(
        forecast_times,
        X_historic,
        intervention,
        historic_times,
        # TODO: Add rng to grid search.
    )

    assert X_do_pred.shape == X_do.shape
    

def test_average_method():
    standard_inference_method_checks(interfere.methods.AverageMethod)


def test_rescomp():
    standard_inference_method_checks(interfere.methods.ResComp)