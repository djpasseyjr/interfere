from typing import Any, Dict, Type

import interfere
from interfere.methods import BaseInferenceMethod
from interfere.methods.nixtla_methods.nixtla_adapter import to_nixtla_df
import numpy as np
from pandas import DataFrame
import pytest
import scipy.stats


SEED = 11
PRED_LEN = 10


def VARIMA_timeseries(dim=4, lags=3, noise_lags=2, tsteps=100, n_do=20):
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

    intervention = interfere.PerfectIntervention([0, 1], [-0.5, -0.5])
    historic_times = t[:-n_do]
    forecast_times = t[-n_do:]
    X_historic = X[:-n_do, :]
    X0_do = X_historic[-lags:, :]

    # TODO: Fix simulate times -> states mapping w historic times.
    sim_times = np.hstack([historic_times[-lags:], forecast_times])
    X_do = model.simulate(
        X0_do,
        sim_times,
        intervention,
        rng=rng
    )
    X_do = X_do[lags:, :]

    return X_historic, historic_times, X_do, forecast_times, intervention

def belozyorov_timeseries(tsteps=100, n_do=20):
    rng = np.random.default_rng(SEED)
    lags = 1
    dim = 3
    model = interfere.dynamics.Belozyorov3DQuad(
        mu=1.81, sigma=0.05, measurement_noise_std = 0.01 * np.ones(dim),
    )

    # Generate a time series
    t = np.linspace(0, 1, tsteps)
    x0 = rng.random(dim)
    X = model.simulate(x0, t, rng=rng)

    intervention = interfere.PerfectIntervention(0, 5.0)
    historic_times = t[:-n_do]
    forecast_times = t[-n_do:]
    X_historic = X[:-n_do, :]
    X0_do = X_historic[-lags, :]

    # TODO: Fix simulate times -> states mapping w historic times.
    sim_times = np.hstack([historic_times[-lags:], forecast_times])
    X_do = model.simulate(
        X0_do,
        sim_times,
        intervention,
        rng=rng
    )
    X_do = X_do[1:, :]

    return X_historic, historic_times, X_do, forecast_times, intervention


def fit_predict_checks(
        method_type: Type[BaseInferenceMethod],
        X_historic: np.ndarray,
        historic_times: np.ndarray,
        X_do: np.ndarray, 
        forecast_times: np.ndarray,
        intervention: interfere.interventions.ExogIntervention
):
    """Checks that fit and predict work for all combos of hyper parameters.
    """
    # Access test parameters.
    method_params = method_type.get_test_params()

    # Create time series combonations.
    historic_endog, historic_exog = intervention.split_exogeneous(X_historic)
    endo_true, exog = intervention.split_exogeneous(X_do)
    forecast_times = forecast_times[:PRED_LEN]
    exog = exog[:PRED_LEN, :]

    # Test fit with and without exog.
    method = method_type(**method_params)
    method.fit(historic_endog, historic_times, historic_exog)

    assert method.is_fit

    method = method_type(**method_params)
    method.fit(historic_endog, historic_times, None)

    assert method.is_fit

    # Test simulate with exog.
    method.fit(historic_endog, historic_times, historic_exog)
    X_do_pred = method.simulate(
            forecast_times,
            X_historic,
            intervention,
            historic_times,
        )

    assert X_do_pred.shape == (PRED_LEN, X_do.shape[1])

    # Simulate without exog
    method.fit(X_historic, historic_times)
    X_do_pred = method.simulate(
            forecast_times,
            X_historic,
            interfere.interventions.IdentityIntervention(),
            historic_times,
        )

    assert X_do_pred.shape == (PRED_LEN, X_do.shape[1])

    method.fit(X_historic, historic_times)
    X_do_pred = method.simulate(
            forecast_times,
            X_historic,
            None,
            historic_times,
        )

    assert X_do_pred.shape == (PRED_LEN, X_do.shape[1])

    # Test fit and predict with different combinations of args.
    arg_combos = [
        (forecast_times, historic_endog, exog, historic_times, historic_exog),
        (forecast_times, historic_endog, None, historic_times, None),
    ]

    # Initialize method fit to data and predict for each combo of params.
    for args in arg_combos:
        ft, he, ex, ht, hex = args
        method = method_type(**method_params)
        method.fit(endog_states=he, t=ht, exog_states=hex)

        assert method.is_fit

        endo_pred = method.predict(
            forecast_times=ft,
            historic_endog=he,
            exog=ex,
            historic_times=ht,
            historic_exog=hex,
        )

        assert endo_pred.shape[1] == endo_true.shape[1]
        assert endo_pred.shape[0] == PRED_LEN

    # Test that prediction_max clips correctly
    method = method_type(**method_params)
    method.fit(historic_endog, historic_times, historic_exog)
    endo_pred = method.predict(
        forecast_times, historic_endog, exog, 
        historic_exog, historic_times, prediction_max=3.0
    )
    
    assert np.all(endo_pred) <= 3.0


def grid_search_checks(
        method_type: Type[BaseInferenceMethod],
        X_historic: np.ndarray,
        historic_times: np.ndarray,
        X_do: np.ndarray, 
        forecast_times: np.ndarray,
        intervention: interfere.interventions.ExogIntervention
):
    # Access test parameters.
    method_params = method_type.get_test_params()
    param_grid = method_type.get_test_param_grid()

    # Initialize method and tune.
    historic_endog, historic_exog = intervention.split_exogeneous(X_historic)
    
    # With exogeneous.
    _, gs_results = interfere.benchmarking.grid_search(
        method_type, method_params, param_grid, historic_endog, historic_times, historic_exog, refit=1)
    
    grid_search_assertions(gs_results, param_grid)

    # Without exogeneous.
    _, gs_results = interfere.benchmarking.grid_search(
        method_type, method_params, param_grid, historic_endog, historic_times, None, refit=1)
    
    grid_search_assertions(gs_results, param_grid)
    

def grid_search_assertions(
    gs_results: DataFrame,
    param_grid: Dict[str, Any]
):
    
    # Make sure that the grid is evaluated for every combo
    assert len(gs_results) == np.prod([len(v) for v in param_grid.values()])
    
    # Make sure that at least 3 non NA mean square errors exist.
    gs_results = gs_results.dropna()
    assert len(gs_results) > 3
    
    # Test that grid search produces a minimum that is much lower than the other
    # combinations. I facilitate this by providing obviously bad hyper
    # parameters to the test param grid (method_type.get_test_param_grid())
    best_mse = gs_results.mean_squared_error.min()
    best_mse_idx = gs_results.mean_squared_error.argmin()
    other_scores = gs_results.drop(
        gs_results.index[best_mse_idx],
        axis=0
    ).mean_squared_error

    # Fit a normal distribution to all scores except the best one.
    prob_of_lowest_score = scipy.stats.norm(
        np.mean(other_scores),
        np.std(other_scores)
    ).cdf(best_mse)


    # Assert that the best score is unlikely to come from the distribution of
    # other scores. This ensures that a clear minimum exists, and this test only
    # passes if you purposely provide TERRRIBLE hyper parameters to the test
    # param grid along with reasonable ones.
    if (prob_of_lowest_score > 0.3) or np.isnan(prob_of_lowest_score):
        print(gs_results[["params", "mean_squared_error"]])
    assert prob_of_lowest_score < 0.3


def check_exogeneous_effect(
        method_type: Type[BaseInferenceMethod],
):
    """Tests that a method can recognize when exogeneous signals influence
    outcome."""
    dim = 3

    params = dict(
        model_type=interfere.dynamics.coupled_map_1dlattice_spatiotemp_intermit1,
        model_params={
            "dim": dim,
            "sigma": 0.0,
            "measurement_noise_std": 0.05 * np.ones(dim)},
        intervention_type=interfere.PerfectIntervention,
        intervention_params={"intervened_idxs": 0, "constants": 0.5},
        initial_conds=[0.01 * np.ones(dim)],
        start_time=0, end_time=100, dt=1,
        rng = np.random.default_rng(SEED)
    )

    Xs, X_dos, t = interfere.generate_counterfactual_forecasts(**params)
    X, X_do = Xs[0], X_dos[0]

    n_do, _ = X_do.shape
    X_historic, historic_times = X[:-n_do, :], t[:-n_do]
    forecast_times = t[-n_do:]

    intervention = params["intervention_type"](**params["intervention_params"])
    method = method_type(**method_type.get_test_params())

    endo, exog = intervention.split_exogeneous(X_historic)
    method.fit(endo, historic_times, exog)

    X_do_pred = method.simulate(
        forecast_times,
        X_historic,
        intervention,
        historic_times,
    )

    mse_intervened = np.mean((X_do_pred - X_do) ** 2) ** 0.5
    mse_no_intervention = np.mean((X_do_pred - X[-n_do:, :]) ** 2) ** 0.5
    assert mse_intervened < mse_no_intervention


def forecast_intervention_check(
        method_type: Type[BaseInferenceMethod],
        X_historic: np.ndarray,
        historic_times: np.ndarray,
        X_do: np.ndarray, 
        forecast_times: np.ndarray,
        intervention: interfere.interventions.ExogIntervention
):
    # Number of predictions to simulate
    num_sims = 3

    # Access test parameters.
    method_params = method_type.get_test_params()
    param_grid = method_type.get_test_param_grid()

    X_do_preds, best_params = interfere.benchmarking.forecast_intervention(
        X=X_historic,
        time_points=historic_times,
        forecast_times=forecast_times, 
        intervention=intervention,
        method_type=method_type,
        method_params=method_params,
        method_param_grid=param_grid,
        num_intervention_sims=num_sims,
        best_params=None,
        rng=np.random.default_rng(SEED)
    )
    
    assert len(X_do_preds) == num_sims
    assert np.all([X_do.shape == X_do_preds[i].shape for i in range(num_sims)])
    assert isinstance(best_params, dict)


def standard_inference_method_checks(method_type: BaseInferenceMethod):
    """Ensures that a method has all of the necessary functionality.
    """
    # Tests each for discrete and continuous time
    fit_predict_checks(method_type, *VARIMA_timeseries())
    fit_predict_checks(method_type, *belozyorov_timeseries())

    grid_search_checks(method_type, *VARIMA_timeseries())
    grid_search_checks(method_type, *belozyorov_timeseries())

    forecast_intervention_check(method_type, *VARIMA_timeseries())
    forecast_intervention_check(method_type, *belozyorov_timeseries())
    
    check_exogeneous_effect(method_type)
    

def test_nixtla_converter():

    n = 100
    n_endog = 3
    n_exog = 2

    endog = np.random.rand(n, n_endog) 
    exog = np.random.rand(n, n_exog)
    t = np.arange(n)

    default_endog_names = [f"x{i}" for i in range(n_endog)]
    default_exog_names = [f"u{i}" for i in range(n_exog)]

    test_exog_names = [f"x{i}" for i in range(n_exog)]

    with pytest.raises(ValueError):
        nf_data = to_nixtla_df(t)

    with pytest.raises(ValueError):
        nf_data = to_nixtla_df(
            t, exog_state_ids=test_exog_names)

    test_unique_ids = ["a1", "a2", "a3"]
    nf_data = to_nixtla_df(t, unique_ids=test_unique_ids)

    assert nf_data.shape == (n_endog * n, 2)
    assert all([i in test_unique_ids for i in nf_data.unique_id.unique()])
    assert len(test_unique_ids) == len(nf_data.unique_id.unique())


    nf_data = to_nixtla_df(
        t, exog_states=exog, unique_ids=test_unique_ids)
    
    assert nf_data.shape == (n_endog * n, 2 + n_exog)
    assert all([id in test_unique_ids for id in nf_data.unique_id.unique()])
    assert len(test_unique_ids) == len(nf_data.unique_id.unique())
    assert all([i in nf_data.columns for i in default_exog_names])

    nf_data = to_nixtla_df(
        t, exog_states=exog, endog_states=endog, exog_state_ids=test_exog_names)
    
    assert nf_data.shape == (n_endog * n, 3 + n_exog)
    assert all([id in default_endog_names for id in nf_data.unique_id.unique()])
    assert len(default_endog_names) == len(nf_data.unique_id.unique())
    assert all([i in nf_data.columns for i in test_exog_names])

    nf_data = to_nixtla_df(t, endog_states=endog)
    
    assert nf_data.shape == (n_endog * n, 3)
    assert all([id in default_endog_names for id in nf_data.unique_id.unique()])
    assert len(default_endog_names) == len(nf_data.unique_id.unique())


def test_average_method():
    fit_predict_checks(
        interfere.methods.AverageMethod,
        *VARIMA_timeseries()
    )


def test_var():
    standard_inference_method_checks(interfere.methods.VAR)


def test_rescomp():
    standard_inference_method_checks(interfere.methods.ResComp)


def test_sindy():
    standard_inference_method_checks(interfere.methods.SINDY)


def test_lstm():
    standard_inference_method_checks(interfere.methods.LSTM)


def test_autoarima():
    standard_inference_method_checks(interfere.methods.AutoARIMA)


def test_ltsf():
    standard_inference_method_checks(interfere.methods.LTSFLinearForecaster)