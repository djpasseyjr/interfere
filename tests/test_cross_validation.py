from unittest.mock import Mock
from typing import Callable

import numpy as np
import optuna
import pytest

from interfere._methods.deep_learning import LTSF
import interfere.methods
from interfere.metrics import rmse
from interfere import CrossValObjective

SEED = 12
RNG = np.random.default_rng(SEED)

DEFAULT_CV_ARGS = {
    "method_type": interfere.methods.VAR,
    "data": RNG.random((100, 3)),
    "times": np.arange(100),
    "train_window_percent": 0.6,
    "num_folds": 3,
    "raise_errors": True,
}

# List of methods to test.
METHODS = [
    interfere.methods.ARIMA,
    interfere.methods.AverageMethod,
    interfere.methods.ResComp,
    interfere.methods.SINDy, 
    LTSF,
    interfere.methods.VAR,
    interfere.methods.LSTM,
    interfere.methods.NHITS,
]    


def test_make_train_window_idxs():
    """Tests the make_train_window_idxs method."""
    cv = CrossValObjective(**DEFAULT_CV_ARGS)
    train_window_idxs = cv._make_train_window_idxs(
        num_train_obs=10,
        num_val_chunk_obs=5,
        num_val_chunks=3,
    )
    assert len(train_window_idxs) == 4
    assert train_window_idxs[0] == (0, 10)
    assert train_window_idxs[1] == (5, 15)
    assert train_window_idxs[2] == (10, 20)
    assert train_window_idxs[3] == (15, 25)

    train_window_idxs = cv._make_train_window_idxs(
        num_train_obs=100,
        num_val_chunk_obs=20,
        num_val_chunks=1,
    )
    assert len(train_window_idxs) == 2
    assert train_window_idxs[0] == (0, 100)
    assert train_window_idxs[1] == (20, 120)

    train_window_idxs = cv._make_train_window_idxs(
        num_train_obs=2,
        num_val_chunk_obs=20,
        num_val_chunks=3,
    )
    assert len(train_window_idxs) == 4
    assert train_window_idxs[0] == (0, 2)
    assert train_window_idxs[1] == (20, 22)
    assert train_window_idxs[2] == (40, 42)
    assert train_window_idxs[3] == (60, 62)


def test_make_val_chunk_idxs_forecast():
    """Tests the make_val_chunk_idxs method when val_scheme == 'forecast'."""
    cv = CrossValObjective(**DEFAULT_CV_ARGS)
    val_chunk_idxs = cv._make_val_chunk_idxs(
        num_train_obs=10,
        num_val_chunk_obs=5,
        num_val_chunks=3,
        val_scheme="forecast",
    )

    assert len(val_chunk_idxs) == 4
    assert val_chunk_idxs[0] == [(10, 15)]
    assert val_chunk_idxs[1] == [(15, 20)]
    assert val_chunk_idxs[2] == [(20, 25)]
    assert val_chunk_idxs[3] == []

    val_chunk_idxs = cv._make_val_chunk_idxs(
        num_train_obs=100,
        num_val_chunk_obs=20,
        num_val_chunks=1,
        val_scheme="forecast",
    )

    assert len(val_chunk_idxs) == 2
    assert val_chunk_idxs[0] == [(100, 120)]
    assert val_chunk_idxs[1] == []


    val_chunk_idxs = cv._make_val_chunk_idxs(
        num_train_obs=2,
        num_val_chunk_obs=20,
        num_val_chunks=3,
        val_scheme="forecast",
    )
    assert len(val_chunk_idxs) == 4
    assert val_chunk_idxs[0] == [(2, 22)]
    assert val_chunk_idxs[1] == [(22, 42)]
    assert val_chunk_idxs[2] == [(42, 62)]
    assert val_chunk_idxs[3] == []


def test_make_val_chunk_idxs_last():
    """Tests the make_val_chunk_idxs method when val_scheme == 'last'."""
    cv = CrossValObjective(**DEFAULT_CV_ARGS)

    val_chunk_idxs = cv._make_val_chunk_idxs(
        num_train_obs=10,
        num_val_chunk_obs=5,
        num_val_chunks=3,
        val_scheme="last",
    )

    assert len(val_chunk_idxs) == 4
    assert val_chunk_idxs[0] == [(20, 25)]
    assert val_chunk_idxs[1] == [(20, 25)]
    assert val_chunk_idxs[2] == []
    assert val_chunk_idxs[3] == []

    with pytest.raises(ValueError):
        val_chunk_idxs = cv._make_val_chunk_idxs(
            num_train_obs=100,
            num_val_chunk_obs=20,
            num_val_chunks=1,
            val_scheme="last",
        )

    val_chunk_idxs = cv._make_val_chunk_idxs(
        num_train_obs=2,
        num_val_chunk_obs=20,
        num_val_chunks=3,
        val_scheme="last",
    )

    assert len(val_chunk_idxs) == 4
    assert val_chunk_idxs[0] == [(42, 62)]
    assert val_chunk_idxs[1] == [(42, 62)]
    assert val_chunk_idxs[2] == []
    assert val_chunk_idxs[3] == []


def test_make_val_chunk_idxs_all():
    """Tests the make_val_chunk_idxs method when val_scheme == 'all'."""
    cv = CrossValObjective(**DEFAULT_CV_ARGS)

    val_chunk_idxs = cv._make_val_chunk_idxs(
        num_train_obs=10,
        num_val_chunk_obs=5,
        num_val_chunks=3,
        val_scheme="all",
    )
    assert len(val_chunk_idxs) == 4
    assert val_chunk_idxs[0] == [(10, 15), (15, 20), (20, 25)]
    assert val_chunk_idxs[1] == [(0, 5), (15, 20), (20, 25)]
    assert val_chunk_idxs[2] == [(0, 5), (5, 10), (20, 25)]
    assert val_chunk_idxs[3] == [(0, 5), (5, 10), (10, 15)]

    val_chunk_idxs = cv._make_val_chunk_idxs(
        num_train_obs=100,
        num_val_chunk_obs=20,
        num_val_chunks=1,
        val_scheme="all",
    )
    assert len(val_chunk_idxs) == 2
    assert val_chunk_idxs[0] == [(100, 120)]
    assert val_chunk_idxs[1] == [(0, 20)]

    val_chunk_idxs = cv._make_val_chunk_idxs(
        num_train_obs=2,
        num_val_chunk_obs=20,
        num_val_chunks=3,
        val_scheme="all",
    )
    assert len(val_chunk_idxs) == 4
    assert val_chunk_idxs[0] == [(2, 22), (22, 42), (42, 62)]
    assert val_chunk_idxs[1] == [(0, 20), (22, 42), (42, 62)]
    assert val_chunk_idxs[2] == [(0, 20), (20, 40), (42, 62)]
    assert val_chunk_idxs[3] == [(0, 20), (20, 40), (40, 60)]


def test_make_val_prior_state_idxs_forecast():
    """Tests the make_val_prior_state_idxs method when val_scheme == 'forecast'."""
    cv = CrossValObjective(**DEFAULT_CV_ARGS)
    val_prior_state_start, val_prior_state_end = cv._make_val_prior_state_idxs(
        val_chunk_start=10,
        num_val_prior_states=5,
        val_scheme="forecast"
    )
    assert val_prior_state_start == 6
    assert val_prior_state_end == 11

    val_prior_state_start, val_prior_state_end = cv._make_val_prior_state_idxs(
        val_chunk_start=100,
        num_val_prior_states=20,
        val_scheme="forecast"
    )
    assert val_prior_state_start == 81
    assert val_prior_state_end == 101


def test_make_val_prior_state_idxs_last_or_all():
    """Tests the make_val_prior_state_idxs when val_scheme is 'last' or 'all'.
    """
    cv = CrossValObjective(**DEFAULT_CV_ARGS)
    val_prior_state_start, val_prior_state_end = cv._make_val_prior_state_idxs(
        val_chunk_start=10,
        num_val_prior_states=5,
        val_scheme="last"
    )
    assert val_prior_state_start == 11
    assert val_prior_state_end == 16

    val_prior_state_start, val_prior_state_end = cv._make_val_prior_state_idxs(
        val_chunk_start=100,
        num_val_prior_states=20,
        val_scheme="last"
    )

    assert val_prior_state_start == 101
    assert val_prior_state_end == 121


def test_make_val_idxs_forecast():
    """Tests the _make_val_idxs method for the forecast scheme."""
    cv = CrossValObjective(**DEFAULT_CV_ARGS)

    val_start_idx, val_end_idx = cv._make_val_idxs(
        val_chunk_start=10,
        val_chunk_end=15,
        num_val_prior_states=5,
        val_scheme="forecast",
    )

    assert val_start_idx == 10
    assert val_end_idx == 15

    val_start_idx, val_end_idx = cv._make_val_idxs(
        val_chunk_start=100,
        val_chunk_end=120,
        num_val_prior_states=20,
        val_scheme="forecast",
    )
    assert val_start_idx == 100
    assert val_end_idx == 120


def test_make_val_idxs_last_or_all():
    """Tests the _make_val_idxs method for the last or all schemes.
    """
    cv = CrossValObjective(**DEFAULT_CV_ARGS)
    val_start_idx, val_end_idx = cv._make_val_idxs(
        val_chunk_start=10,
        val_chunk_end=20,
        num_val_prior_states=5,
        val_scheme="last",
    )

    assert val_start_idx == 15
    assert val_end_idx == 20

    val_start_idx, val_end_idx = cv._make_val_idxs(
        val_chunk_start=100,
        val_chunk_end=140,
        num_val_prior_states=20,
        val_scheme="last",
    )
    assert val_start_idx == 120
    assert val_end_idx == 140


def test_cvr_cv_init():
    """Tests the __init__ method of the CrossValObjective class."""
    cv = CrossValObjective(**DEFAULT_CV_ARGS)
    assert cv.method_type == interfere.methods.VAR
    assert cv.data.shape == (100, 3)
    assert cv.times.shape == (100,)
    assert cv.train_window_percent == 0.6
    assert cv.num_folds == 3
    assert cv.val_scheme == "forecast"
    assert cv.num_val_prior_states == 10
    assert isinstance(cv.metric, Callable)
    assert cv.metric_direction == "minimize"
    assert cv.hyperparam_func == interfere.methods.VAR._get_optuna_params
    assert cv.store_preds == True
    assert cv.intervention == interfere.PerfectIntervention([], [])
    assert cv.trial_results == {}
    assert cv.num_obs == 100
    assert cv.num_val_chunks == 2
    assert cv.num_train_obs == 60
    assert cv.num_val_chunk_obs == 20
    assert cv.cv_descr == (
        "Control V.S. Response Cross Validation Objective"
        "\n\tValidation scheme: forecast"
        "\n\tNumber of folds: 3"
        "\n\tNumber of training observations: 60"
        "\n\tNumber of validation chunks: 2"
        "\n\tNumber of observations per validation chunk: 20"
        "\n\tNumber of validation prior states: 10"
    )
    
    assert cv.train_window_idxs == [
        (0, 60),
        (20, 80),
        (40, 100)
    ]
    assert cv.val_chunk_idxs == [
        [(60, 80)],
        [(80, 100)],
        [],
    ]


def test_cvr_cv_init_raises():
    """Tests the __init__ method of the CrossValObjective class with invalid
    arguments.
    """
    with pytest.raises(ValueError, match="validation chunk"):
        cv = CrossValObjective(**{
            **DEFAULT_CV_ARGS,
            "num_folds": 41
        })

    with pytest.raises(ValueError, match="observations in the training set"):
        cv = CrossValObjective(**{
            **DEFAULT_CV_ARGS,
            "train_window_percent": 0.01
        })

    with pytest.raises(ValueError, match="number of folds"):
        cv = CrossValObjective(**{
            **DEFAULT_CV_ARGS,
            "num_folds": 1
        })

    with pytest.raises(ValueError, match="validation scheme"):
        cv = CrossValObjective(**{
            **DEFAULT_CV_ARGS,
            "val_scheme": "invalid"
        })

    with pytest.raises(ValueError, match="metric direction"):
        cv = CrossValObjective(**{
            **DEFAULT_CV_ARGS,
            "metric_direction": "invalid"
        })

    with pytest.raises(ValueError, match="val_scheme=='forecast'"):
        cv = CrossValObjective(**{
            **DEFAULT_CV_ARGS,
            "val_scheme": "forecast",
            "train_window_percent": 0.09
        })

    with pytest.raises(ValueError, match="val_scheme is 'last' or 'all'"):
        cv = CrossValObjective(**{
            **DEFAULT_CV_ARGS,
            "val_scheme": "last",
            "num_val_prior_states": 21
        })


@pytest.mark.parametrize("val_scheme", ["forecast", "last", "all"])
@pytest.mark.parametrize("exog_idxs", [None, [0]])
def test_cvr_cv_call(val_scheme: str, exog_idxs: list[int]):
    """Tests the __call__ method of the CrossValObjective class.

    Args:
        val_scheme: The validation scheme to use. One of ["forecast",
            "last", "all].
        exog_idxs (list[int]): An list of exogenous indexs. Can be None.
    """
    # Build a mock method type.
    mock_method = Mock()
    predict_side_effect = lambda *args, **kwargs: RNG.random(
        (len(args[0]), kwargs["prior_endog_states"].shape[1])
    )
    mock_method.predict = Mock(side_effect=predict_side_effect)
    mock_method.fit = Mock()
    mock_method.get_window_size = Mock(return_value=5)

    mock_method_type = Mock(return_value=mock_method)

    cv = CrossValObjective(**{
        **DEFAULT_CV_ARGS,
        "method_type": mock_method_type,
        "hyperparam_func": Mock(return_value={}),
        "store_preds": True,
        "val_scheme": val_scheme,
        "exog_idxs": exog_idxs
    })

    # Make a mock trial.
    trial = Mock()
    trial.number = 1

    score = cv(trial)

    # Raise an error 

    assert isinstance(score, float) and (score >= 0) and (not np.isnan(score)), (
        "Invalid score."
        f"\n\tScore: {score}"
        f"\n\tFold scores: {cv.trial_results[trial.number]['scores']}"
    )

    assert cv.trial_results.get(trial.number, None) is not None, (
        "Trial results not stored"
    )

    # Check that hyper parameter and method type are called once each.
    cv.hyperparam_func.assert_called_once_with(
        trial, max_lags=10, max_horizon=50)
    mock_method_type.assert_called_once_with()

    # Check that fit is called the correct number of times.
    val_folds = [fold for fold in cv.val_chunk_idxs if fold != []]
    assert mock_method.fit.call_count == len(val_folds), (
        "The fit method should be called once for each validation fold."
        f"\n\tCall count: {mock_method.fit.call_count}"
        f"\n\tValidation chunk indexs: {cv.val_chunk_idxs}"
    )

    # Check that the predict method is called the correct number of times.
    val_chunks = [idx for fold in cv.val_chunk_idxs for idx in fold]
    assert mock_method.predict.call_count == len(val_chunks), (
        "The predict method should be called once for each validation "
        "chunk."
    )

    # Check that the cross validator stores the correct number of results.
    for key in ["preds", "errors", "scores"]:
        assert len(cv.trial_results[trial.number][key]) == len(val_chunks), (
            f"The number of saved {key} should be equal to the number of "
            "validation chunks."
        )

    # Check that the target validation sets correspond to the correct
    # section of data.
    for i, (val_start, val_end) in enumerate(
        cv.trial_results[trial.number]["val_idxs"]):
        target = cv.trial_results[trial.number]["targets"][i]
        val_data = cv.data[val_start:val_end, :]
        assert np.all(
            cv.data[val_start:val_end, :] == target), (
                "The target validation set does not correspond to the "
                "correct section of data."
                f"\n\tScheme: {cv.val_scheme}"
                f'\n\tVal idxs: {cv.trial_results[trial.number]["val_idxs"]}'
                f"\n\tValidation start index: {val_start}"
                f"\n\tValidation end index: {val_end}"
                f"\n\tData shape: {val_data.shape}"
                f"\n\tTarget shape: {target.shape}"
                f"\n\tData: {val_data}"
                f"\n\tTarget: {target}"
            )

    # Check that there are no errors.
    errors = np.array(cv.trial_results[trial.number]["errors"])
    assert np.all(errors == None), (
        f"There should be no errors. Errors = {errors}"
    )


@pytest.mark.parametrize("val_scheme", ["forecast", "last", "all"])
@pytest.mark.parametrize("exog_idxs", [None, [0]])
def test_cvr_cv_call_no_store_preds(val_scheme: str,  exog_idxs: list[int]):
    """Tests the __call__ method of the CrossValObjective class when
    store_preds is False.

    Args:
        val_scheme: The validation scheme to use. One of ["forecast",
            "last", "all].
        exog_idxs (list[int]): An list of exogenous indexs. Can be None.
    """
    cv = CrossValObjective(**{
        **DEFAULT_CV_ARGS,
        "store_preds": False,
        "val_scheme": val_scheme,
        "hyperparam_func": Mock(return_value={}),
        "exog_idxs": exog_idxs
    })

    # Make a mock trial.
    trial = Mock()
    trial.number = 1

    # Validation sets
    val_chunks = [idx for fold in cv.val_chunk_idxs for idx in fold]

    # Call objective. 
    score = cv(trial)

    # Check that score is valid.
    assert isinstance(score, float) and (score >= 0) and (not np.isnan(score)), (
        "Invalid score."
        f"\n\tScore: {score}"
        f"\n\tFold scores: {cv.trial_results[trial.number]['scores']}"
    )

    # Check that predictions and targets are not stored.
    for key in cv.trial_results[trial.number].keys():
        if key in ["preds", "targets"]:
            assert cv.trial_results[trial.number][key] == [], (
                "When store_preds is False, cv.trial_results[trial.number]"
                f"[{key}] should be empty."
                "\n\tcv.trial_results[trial.number][key]} = "
                f"{cv.trial_results[trial.number][key]}"
            )

        else:
            assert len(cv.trial_results[trial.number][key]) == len(val_chunks), (
                "The number of saved {key} should be equal to the number of "
                "validation chunks."
            )

    # Check that there are no errors.
    errors = np.array(cv.trial_results[trial.number]["errors"])
    assert np.all(errors == None), (
        f"There should be no errors. Errors = {errors}"
    )


def test_cvr_cv_call_raises():
    """Tests the __call__ method of the CrossValObjective class with invalid
    arguments.
    """
    mock_method = Mock()
    mock_method.predict = Mock(side_effect=lambda *a, **kw: 1/0)
    mock_method.fit = Mock()
    mock_method.get_window_size = Mock(return_value=5)

    mock_method_type = Mock(return_value=mock_method)

    cv = CrossValObjective(**{
        **DEFAULT_CV_ARGS,
        "method_type": mock_method_type,
        "store_preds": False,
        "hyperparam_func": Mock(return_value={}),
        "raise_errors": False,
    })

    # Make a mock trial.
    trial = Mock()
    trial.number = 1

    # Call objective. 
    score = cv(trial)

    assert np.isnan(score), f"Score should be NaN: score = {score}"

    # Check that errors were stored.
    errors = np.array(cv.trial_results[trial.number]["errors"])
    assert np.all(errors != None), (
        f"CV should store errors. Errors = {errors}"
    )


@pytest.mark.parametrize("method_type", METHODS)
@pytest.mark.parametrize("val_scheme", ["forecast", "last", "all"])
def test_cvr_cv_call_methods(
    method_type: interfere.ForecastMethod,
    val_scheme: str
):
    """Tests that the cv objective works for each method.

    Args:
        method_type (intergere.ForecastMethod): An interfere method.
        val_scheme (str): The validation scheme to use. One of ["forecast",
            "last", "all].
    """
    cv = CrossValObjective(**{
        **DEFAULT_CV_ARGS,
        "data": RNG.random((600, 3)),
        "times": np.arange(600),
        "train_window_percent": 0.5,
        "num_folds": 4,
        "num_val_prior_states": 5,
        "method_type": method_type,
        "store_preds": False,
        "val_scheme": val_scheme,
        "raise_errors": True,
    })

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(cv, n_trials=1)
    score = study.best_value

    # Make sure that the method can initialize using the best parameters.
    ps = study.best_params
    best_method = method_type(**ps)

    # Score must not be NaN.
    assert isinstance(score, float)