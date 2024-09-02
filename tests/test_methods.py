import re
from typing import Any, Dict, Type

import interfere
from interfere.methods import BaseInferenceMethod
from interfere.methods.nixtla_methods.nixtla_adapter import to_nixtla_df
from interfere.benchmarking import generate_counterfactual_forecasts
import numpy as np
from pandas import DataFrame
import pytest
import scipy.stats


SEED = 11
PRED_LEN = 10


def VARIMA_timeseries(dim=5, lags=3, noise_lags=2, tsteps=100, n_do=20):
    # Initialize a VARMA model
    rng = np.random.default_rng(SEED)
    phis = [0.5 * (rng.random((dim, dim)) - 0.5) for i in range(lags)]
    thetas = [0.5 * (rng.random((dim, dim)) - 0.5) for i in range(noise_lags)]
    sigma = rng.random((dim, dim))
    sigma += sigma.T
    model = interfere.dynamics.VARMA_Dynamics(phis, thetas, sigma)

    max_lags = max(lags, noise_lags)
    # Generate a time series
    t = np.arange(tsteps)
    prior_states = rng.random((max_lags, dim))
    X = model.simulate(t, prior_states, rng=rng)

    intervention = interfere.PerfectIntervention([0, 1], [-0.5, -0.5])
    historic_t = t[:-n_do]
    historic_X = X[:-n_do, :]
    t_do = t[(-n_do - 1):]

    X_do = model.simulate(
        t_do,
        historic_X,
        prior_t=historic_t,
        intervention=intervention,
        rng=rng
    )
    X_do = X_do[lags:, :]

    return historic_X, historic_t, X_do, t_do, intervention

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
    X = model.simulate(t, x0, rng=rng)

    intervention = interfere.PerfectIntervention(0, 5.0)
    historic_times = t[:-n_do]
    X_historic = X[:-n_do, :]
    forecast_times = t[(-n_do - 1):]
    X0_do = X_historic[-lags, :]

    X_do = model.simulate(
        forecast_times,
        X0_do,
        intervention=intervention,
        rng=rng
    )
    X_do = X_do[1:, :]

    return X_historic, historic_times, X_do, forecast_times, intervention


def fit_predict_checks(
        method_type: Type[BaseInferenceMethod],
        X_historic: np.ndarray,
        prior_t: np.ndarray,
        X_do: np.ndarray, 
        forecast_times: np.ndarray,
        intervention: interfere.interventions.ExogIntervention
):
    """Checks that fit and predict work for all combos of hyper parameters.
    """
    # Access test parameters.
    method_params = method_type.get_test_params()

    # Create time series combonations.
    prior_endog_states, prior_exog_states = intervention.split_exogeneous(X_historic)
    endo_true, exog = intervention.split_exogeneous(X_do)
    forecast_times = forecast_times[:PRED_LEN]
    exog = exog[:PRED_LEN, :]

    # Test fit with and without exog.
    method = method_type(**method_params)
    method.fit(prior_t, prior_endog_states, prior_exog_states)

    assert method.is_fit

    method = method_type(**method_params)
    method.fit(prior_t, prior_endog_states, None)

    assert method.is_fit

    # Test simulate with exog.
    method.fit(prior_t, prior_endog_states,  prior_exog_states)
    X_do_pred = method.simulate(
            t=forecast_times,
            prior_states=X_historic,
            prior_t=prior_t,
            intervention=intervention,
        )

    assert X_do_pred.shape == (PRED_LEN, X_do.shape[1])

    # Simulate without exog
    method.fit(prior_t, X_historic)
    X_do_pred = method.simulate(
        t=forecast_times,
        prior_states=X_historic,
        prior_t=prior_t,
        intervention=interfere.interventions.IdentityIntervention(),
    )

    assert X_do_pred.shape == (PRED_LEN, X_do.shape[1])

    method.fit(prior_t, X_historic)
    X_do_pred = method.simulate(
            t=forecast_times,
            prior_states=X_historic,
            prior_t=prior_t,
        )

    assert X_do_pred.shape == (PRED_LEN, X_do.shape[1])

    # Test fit and predict with different combinations of args.
    arg_combos = [
        (forecast_times, prior_endog_states, exog, prior_t, prior_exog_states),
        (forecast_times, prior_endog_states, None, prior_t, None),
    ]

    # Initialize method fit to data and predict for each combo of params.
    for args in arg_combos:
        ft, pe, ex, pt, pex = args
        method = method_type(**method_params)
        method.fit(t=pt, endog_states=pe, exog_states=pex)

        assert method.is_fit

        endo_pred = method.predict(
            t=ft,
            prior_endog_states=pe,
            prior_exog_states=pex,
            prior_t=pt,
            prediction_exog=ex,
        )

        assert endo_pred.shape[1] == endo_true.shape[1]
        assert endo_pred.shape[0] == PRED_LEN

    # Test that prediction_max clips correctly
    method = method_type(**method_params)
    method.fit(prior_t, prior_endog_states, prior_exog_states)

    prediction_max = max(np.max(prior_endog_states), np.max(prior_exog_states))
    endo_pred = method.predict(
        t=forecast_times,
        prior_endog_states=prior_endog_states,
        prior_exog_states=prior_exog_states,
        prior_t=prior_t,
        prediction_exog=exog,
        prediction_max=prediction_max
    )
    
    assert np.all(endo_pred) <= prediction_max

def predict_error_checks(method_type):
    """Checks that predict raises appropriate errors."""

    method_params = method_type.get_test_params()

    # Warning and exception tests.
    t = np.arange(10)
    prior_endog_states = np.random.rand(9, 2)
    prior_exog_states = np.random.rand(9, 3)
    prediction_exog = np.random.rand(len(t), 3)
    true_prior_t = np.arange(-8, 1)
    method = method_type(**method_params)
    method.fit(true_prior_t, prior_endog_states)

    # Test that predict requires monotonic time arrays.
    with pytest.raises(ValueError, match=(
        f"Time points passed to the {str(type(method).__name__)}.predict "
        "`t` argument must be strictly increasing."
    )):
        method.predict(np.random.rand(10), prior_endog_states)

    # Test that predict warns about inferring prior_t.
    with pytest.warns(UserWarning, match=(
        "Inferring additional `prior_t` values. Assuming `prior_t` has"
        " the same equally spaced timestep size as `t`"
    )):
        method.predict(t, prior_endog_states[-1,:], prior_t=t[0:1])

    # Test that predict requires equally spaced t to infer prior_t.
    with pytest.raises(ValueError, match=(
        "The `prior_t` argument not provided"
        " AND `t` is not equally spaced. Cannot infer "
        " `prior_t`. Either pass it explicitly or provide "
        " equally spaced time `t`."
    )):
        method.predict(np.hstack([
            t, [t[-1] + np.pi]]), prior_endog_states[-1,:], prior_t=None)

    # Test that predict requires last entry of prior_t to equal first entry of
    # t. 
    bad_prior_t = np.random.rand(1)
    with pytest.raises(ValueError, match=re.escape(
        f"For {str(type(method).__name__)}.predict, the last prior time, "
        f"`prior_t[-1]`={bad_prior_t[-1]} must equal the first simulation "
        f"time t[0]={t[0]}."
    )):
        method.predict(t, prior_endog_states[-1,:], prior_t=bad_prior_t)

    # Test that predict requires the same number of entries in prior_t and
    # prior_endog_states.
    p = 3
    num_prior_times = 2
    bad_prior_t = true_prior_t[-num_prior_times:].copy()
    bad_prior_t[0] -= np.pi

    with pytest.raises(ValueError, match=re.escape(
        f"{str(type(method).__name__)}.predict was passed {p} "
        "prior_endog_states but there are only "
        f"{num_prior_times} entries in `prior_t`."
    )):
        method.predict(t, prior_endog_states[-p:,: ], prior_t=bad_prior_t)

    # Test that predict requires monotonic time arrays.
    with pytest.raises(ValueError, match=re.escape(
        f"Prior time points passed to {str(type(method).__name__)}."
        "predict must be strictly increasing."
    )):
        bad_prior_t = np.random.rand(len(true_prior_t))
        bad_prior_t[-1] = t[0]
        method.predict(t, prior_endog_states, prior_t=bad_prior_t)
        
    # Test that predict raises warning when not enough historic data is passed.

    method = method_type(**method_params)
    method.fit(true_prior_t, prior_endog_states, prior_exog_states)
    # Change the window size method. This is a bit hacky and could cause
    # problems in the future. The real issue here is that window size is not
    # an attribute and therefore we can't change it. It is computed from the
    # internal method parameters which we have no knowledge of at this scope.
    # Therefore this hack is used instead of reconfiguring the test API and
    # requring each method to provide parameters that lead to a window size
    # greater than 2. Additionally a window size smaller than 2 will throw an
    # error and some methods don't have parameters that generate a window size
    # bigger than 2. So a hack is where we are at for now. The point is just to
    # ensure that the warning is raised correctly.
    w_old = method.get_window_size()
    method.get_window_size = lambda : w_old + 1


    with pytest.warns(UserWarning, match=str(type(method).__name__) + " has window size "
        f"{w_old + 1} but only recieved {w_old} "
        "endog observations. Augmenting historic edogenous "
        "observations with zeros."
    ):
        method.predict(
            t,
            prior_endog_states=prior_endog_states[-w_old:, :],
            prior_exog_states=prior_exog_states[-w_old:, :],
            prior_t=true_prior_t[-w_old:], 
            prediction_exog=prediction_exog,
            prediction_max=3.0
        )

    with pytest.warns(UserWarning, match=str(type(method).__name__) + " has window size"
        f" {w_old + 1} but only recieved {w_old} exog observations. "
        "Augmenting historic exogenous observations with zeros."
    ):
        method.predict(
            t=t,
            prior_endog_states=prior_endog_states[-w_old:, :],
            prior_exog_states=prior_exog_states[-w_old:, :],
            prior_t=true_prior_t[-w_old:],
            prediction_exog=prediction_exog, 
            prediction_max=3.0
        )

    # Test method.predict time array warnings and exceptions.
    X = np.random.rand(10, 2)
    t = np.arange(10)
    true_prior_t = np.arange(-9, 1)
    method.fit(true_prior_t, X)

    with pytest.warns(UserWarning, match=(
            "Inferring additional `prior_t` values. Assuming `prior_t` has"
            " the same equally spaced timestep size as `t`"
    )):
        method.predict(t, X[-w_old:], prior_t=true_prior_t[-w_old:])
    
    # Test that predict requires equally spaced t to infer prior_t.
    with pytest.raises(ValueError, match=re.escape(
        f"{str(type(method).__name__)}.predict augmented "
        "`prior_endog_states` with zeros but `prior_t` was not "
        "equally spaced so it was not possible to infer "
        "additional prior times. \n\nTo solve, pass at least "
        f"({w_old + 1}) previous time values or use "
        "equally spaced `prior_t`."
    )):
        bad_prior_t = true_prior_t[-w_old:].copy()
        bad_prior_t[0] -= np.pi
        method.predict(t, X[-w_old:], prior_t=bad_prior_t)
    # Clean up method so that bad window size doesn't break things.
    method = None

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
    prior_endog_states, prior_exog_states = intervention.split_exogeneous(X_historic)
    
    # With exogeneous.
    _, gs_results = interfere.benchmarking.grid_search(
        method_type,
        method_params,
        param_grid,
        t=historic_times,
        endog_states=prior_endog_states,
        exog_states=prior_exog_states,
        refit=1
    )
    
    grid_search_assertions(gs_results, param_grid)

    # Without exogeneous.
    _, gs_results = interfere.benchmarking.grid_search(
        method_type,
        method_params,
        param_grid,
        t=historic_times,
        endog_states=prior_endog_states,
        refit=1
    )
    
    grid_search_assertions(gs_results, param_grid)
    

def grid_search_assertions(
    gs_results: DataFrame,
    param_grid: Dict[str, Any],
    prob_threshold = 0.3,
    mse_diff_cutoff = 20
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
    worst_mse = gs_results.mean_squared_error.max()

    best_mse_idx = gs_results.mean_squared_error.argmin()
    other_scores = gs_results.drop(
        gs_results.index[best_mse_idx],
        axis=0
    ).mean_squared_error

    # Fit a normal distribution to all scores except the best one.
    lowest_score_prob = scipy.stats.norm(
        np.mean(other_scores),
        np.std(other_scores)
    ).cdf(best_mse)


    # Assert that the best score is unlikely to come from the distribution of
    # other scores. This ensures that a clear minimum exists, and this test only
    # passes if you purposely provide TERRRIBLE hyper parameters to the test
    # param grid along with reasonable ones.
    if (lowest_score_prob >= prob_threshold) or np.isnan(lowest_score_prob):
        print(gs_results[["params", "mean_squared_error"]])
    prob_is_low =  lowest_score_prob < prob_threshold
    diff_is_big = worst_mse - best_mse > mse_diff_cutoff
    assert prob_is_low or diff_is_big


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

    Xs, X_dos, t = generate_counterfactual_forecasts(**params)
    X, X_do = Xs[0], X_dos[0]

    n_do, _ = X_do.shape
    X_historic, historic_times = X[:(-n_do + 1), :], t[:(-n_do + 1)]
    forecast_times = t[-n_do:]

    intervention = params["intervention_type"](**params["intervention_params"])
    method = method_type(**method_type.get_test_params())

    endo, exog = intervention.split_exogeneous(X_historic)
    method.fit(historic_times, endo, exog)

    X_do_pred = method.simulate(
        t=forecast_times,
        prior_states=X_historic,
        prior_t=historic_times,
        intervention=intervention
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
    assert np.all([
        (len(forecast_times), X_do.shape[1]) == X_do_preds[i].shape 
        for i in range(num_sims)
    ])
    assert isinstance(best_params, dict)


def standard_inference_method_checks(method_type: BaseInferenceMethod):
    """Ensures that a method has all of the necessary functionality.
    """
    # Tests each for discrete and continuous time
    fit_predict_checks(method_type, *VARIMA_timeseries())
    fit_predict_checks(method_type, *belozyorov_timeseries())
    predict_error_checks(method_type)

    grid_search_checks(method_type, *VARIMA_timeseries())
    grid_search_checks(method_type, *belozyorov_timeseries())

    forecast_intervention_check(method_type, *VARIMA_timeseries())
    forecast_intervention_check(method_type, *belozyorov_timeseries())
    
    # check_exogeneous_effect(method_type)
    

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
    predict_error_checks(interfere.methods.AverageMethod)


def test_var():
    standard_inference_method_checks(interfere.methods.VAR)


def test_rescomp():
    standard_inference_method_checks(interfere.methods.ResComp)


def test_sindy():
    standard_inference_method_checks(interfere.methods.SINDY)


def test_lstm():
    method_type = interfere.methods.LSTM
    fit_predict_checks(method_type, *VARIMA_timeseries())
    fit_predict_checks(method_type, *belozyorov_timeseries())
    # predict_error_checks(method_type)

    grid_search_checks(method_type, *VARIMA_timeseries())
    grid_search_checks(method_type, *belozyorov_timeseries())

    forecast_intervention_check(method_type, *VARIMA_timeseries())
    forecast_intervention_check(method_type, *belozyorov_timeseries())

def test_autoarima():
    standard_inference_method_checks(interfere.methods.AutoARIMA)


def test_ltsf():
    standard_inference_method_checks(interfere.methods.LTSFLinearForecaster)
