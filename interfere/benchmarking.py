from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
)
from warnings import warn

import numpy as np
import pandas as pd

from .base import DynamicModel, ForecastMethod, DEFAULT_RANGE
from .interventions import ExogIntervention
from .utils import copy_doc


def generate_counterfactual_forecasts(
    model_type: Optional[Type[DynamicModel]] = None,
    model_params: Optional[Dict[str, Any]] = None,
    intervention_type: Optional[
        Callable[[np.ndarray, float], np.ndarray]] = None,
    intervention_params: Optional[Dict[str, Any]] = None,
    initial_conds: Optional[Iterable[np.ndarray]] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    dt: Optional[float] = None,
    train_per: Optional[float] = 0.8,
    reps: Optional[int] = 1,
    rng: np.random.mtrand.RandomState = DEFAULT_RANGE,
):
    """Generates trajectories and corresponding counterfactual trajectories.

    Args:
        model (Type[DynamicModel]): The type of the dynamic model to simulate
            with and without interventions.
        model_params (Dict[str, Any]): The initialization parameters of the    
            dynamic model.
        intervention_type (Type[Intervention]): The type of the intervention to
            apply.
        intervention_params (Dict[str, Any]): The initialization parameters of
            the intervention.
        initials_conds (List[np.ndarray]): A list of initial conditions to use
            to run the simulations.
        start_time (float): The time to start each simulation.
        end_time (float): The time to end each simulation.
        dt (float): The timestep size for the numerical solver.
        train_per (float): Percent of time points to use as training data. Must
            be between zero and one.
        rng: A numpy random state for reproducibility. (Uses numpy's mtrand 
            random number generator by default.)
            
    Returns:
        observation: An list of arrays where the ith array represents a
            realization of a trajectory of the dynamic model when the initial
            condition is initial_condition_iter[i]. The ith array has dimensions
            (n_i, m) where  `n_i = len(time_points_iter[i])` and m is the
            dimensionality of the system.
        counterfactual: A list of arrays corresponding exactly to observations
            except that the supplied intervention was applied.  
        time_points: The time points corresponding to each row of a particular 
            observation. 
    """
    model = model_type(**model_params)
    intervention = intervention_type(**intervention_params)
    time_points = np.arange(start_time, end_time + dt, dt)

    observations = [
        model.simulate(
            t=time_points,
            prior_states=ic,
            intervention=None,
            rng=rng,
        )
        for ic in initial_conds
    ]

    # Collect intervention points and time scales.
    counterf_init_cond = []
    for X in observations:
        test_idx = int(len(time_points) * train_per)
        counterf_init_cond.append(X[test_idx, :])

    # Simulate the intervention.
    counterfactual_forecasts = [
        model.simulate(
            t=time_points[test_idx:],
            prior_states=ic,
            intervention=intervention,
            rng=rng,
        )
        for ic in counterf_init_cond
    ]
    return observations, counterfactual_forecasts, time_points


def forecast_intervention(
    X: np.ndarray,
    time_points: np.ndarray,
    forecast_times: np.ndarray,
    intervention: ExogIntervention,
    method_type: Type,
    method_params: Dict[str, Any],
    method_param_grid: Dict[str, Any],
    num_intervention_sims: int,
    best_params: Optional[Dict["str", Any]] = None,
    rng: np.random.RandomState = DEFAULT_RANGE
) -> Tuple[List[float], Dict[str, Any]]:
    """Tunes, fits and forecasts the effect of an intervention with a method.

    Args:
        X (np.ndarray): An (m, n) matrix that is interpreted to be a  
            realization of an n dimensional stochastic multivariate timeseries.
        time_points (np.ndarray): 1D array of time points corresponding to the
            rows of X.
        forecast_times (np.ndarray): 1D array of time points corresponding to
            the desired prediction times.
        intervention (interfere.Intervention): The type of the intervention to
            apply.
        method_type (Type[ForecastMethod]): The method to be
            used for prediction.
        method_params (dict): A dictionary of default parameters for the method.
        method_param_grid (dict): The parameter grid for a sklearn grid search.
        num_intervention_preds (int): The number of interventions to simulate
            with the method. Used for noisy methods.
        best_params (dict): An optional dictionary of the top hyper parameters.
        rng: An optional numpy random state for reproducibility. (Uses numpy's 
            mtrand random number generator by default.)

    Returns:
        X_do_preds: A list of `num_intervention_preds` numpy arrays that are   
            each are an attempt to predict how the system will behave next in
            response to the intervention.
        best_params: A Dict of the best parameters found by the grid search.
    """

    prior_endog_states, prior_exog_states = intervention.split_exog(X)

    if best_params is None:

        _, gs_results = grid_search(
            method_type,
            method_params,
            method_param_grid,
            t=time_points,
            endog_states=prior_endog_states,
            exog_states=prior_exog_states,
            # Uses a moving window of train on 20% of the data, predict next 20%
            initial_train_window_percent = 0.2,
            predict_percent = 0.2,
            # Refits the forecaster every time. This must be set to 1 in order
            # to use statsforecast models bc they store historic exog during
            # fit. 
            refit=1
        )

        best_params = gs_results.params.iloc[
            gs_results.mean_squared_error.argmin()
        ]

    # Combine best params with default params. (Items in best_params will
    # overwrite items in method_params if there are duplicates here.)
    sim_params = {**method_params, **best_params}

    # Simulate intervention
    method = method_type(**sim_params)
    method.fit(
        t=time_points,
        endog_states=prior_endog_states,
        exog_states=prior_exog_states
    )
    
    X_do_preds = [
        method.simulate(
            forecast_times,
            prior_states=X,
            prior_t=time_points,
            intervention=intervention,
            rng=rng
        )
        for i in range(num_intervention_sims)
    ]
    return X_do_preds, sim_params