from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
)
from warnings import warn

import numpy as np
import pandas as pd
from skforecast.model_selection_multiseries import grid_search_forecaster_multiseries

from .base import DynamicModel, DEFAULT_RANGE
from .interventions import ExogIntervention
from .utils import copy_doc
from .methods import BaseInferenceMethod


##################################
# Skforecast Grid Search Adapter #
##################################


class ForecasterAutoregMultiSeriesCustom:
    """Custom class to interface with the skforecast grid search method.
    
    Because of the difficulty getting hyper parameter tuning for time series
    models to work with sklearn (not enough features) and sktime
    (over-engineered), the interfere package uses a small hack to make custom
    methods compatible with the skforecast API.

    The skforecast package does not have a base API and their grid search
    functions only with for their own defined types. By naming our adapter class
    the same name as their defined types, we are able to bypass their type
    checker and insert optimize any interfere method by wrapping it in this
    custom class.

    This class does not inherit any methods from skforecast classes, and this is
    done to restrict the manner that functions can operate on internal
    attributes.
    
    Still, yes, this is a hack. I've been careful and looked through their code
    for ways the dummy class attributes initialized here interact with grid
    search, and believe they have no major impact, but there is always a risk of
    a bug. (Hence, this is accompanied with a suite of tests.)
    """


    def __init__(
        self,
        method_type: Type[BaseInferenceMethod],
        method_params: Dict[str, Any],
        **kwargs
    ):
        self.method = method_type(**method_params, **kwargs)

        # Sets the number of previous observations the method needs in order to
        # make a prediction.
        self.window_size = self.method.get_window_size()

        # These attributes are dummy variables which are needed to maintain 
        # compatibility with skforecast API.
        self.dropna_from_series = None
        self.regressor = "None"
        self.level = None


    def fit(
        self,
        series: pd.DataFrame = None,
        exog: pd.DataFrame = None,
        store_last_window = None,
        suppress_warnings = None,
        store_in_sample_residuals = None,
    ):
        """Adapter that maps skforecast fit API to 
        interfere.BaseInferenceMethod.fit API.

        Args:
            series: A pandas dataframe containing the endogenous states with
                time values as the index.
            exog: A pandas dataframe contianing the exogenous states with
                time values as the index.
            store_last_window: NOT USED - INCLUDED FOR COMPATABILITY
            suppress_warnings: NOT USED - INCLUDED FOR COMPATABILITY
            store_in_sample_residuals: NOT USED - INCLUDED FOR COMPATABILITY
        """
        t = series.index.values

        # Store the endog and exog data.
        self._series = series
        self._exog = exog

        exog_states = None
        if exog is not None:
            exog_states = exog.values

        return self.method.fit(
            t=t,
            endog_states=series.values,
            exog_states=exog_states
        )
    

    def predict(
        self,
        steps: int,           
        levels,
        last_window: pd.DataFrame,
        exog: pd.DataFrame,
        suppress_warnings,
    ):
        """Maps skforecast predict() to interfere.BaseInferenceMethod.predict().
        
        Args:
            steps: The number of timesteps forward to predict.
            levels: NOT USED - INCLUDED FOR COMPATABILITY.
            last_window: A DataFrame of previously simulated (or observed)
                states.
            exog: A DataFrame of exogenous signals corresponding to each step.
            suppress_warnings: NOT USED - INCLUDED FOR COMPATABILITY.

        Returns:
            A dataframe where rows are observations, columns are variables and
            the index is floating point time values.
        """
        if len(last_window) < 2:
            raise ValueError("For compatibility with skforecast, all interfere" " methods must provide at least two previous observations and have"
            " a time window size of two or more so that the timestep can be "
            "determined from the passed historic time series.")
        
        prior_t = last_window.index.values
        prior_endog_states = last_window.values
        prediction_exog = None
        prior_exog_states = None

        if exog is not None:
            # If the prediction involves exogeneous data build exog
            # arrays for interfere methods.
            prediction_exog = exog.values

            # Build historic exogeneous array. TODO: Pass true exog states.
            if set(prior_t).issubset(self._exog.index):        
                prior_exog_states = np.vstack([
                    self._exog.loc[ti].values
                    for ti in prior_t
                ])
            else:
                warn("Missing exogeneous data detected in grid gearch."
                    " Replacing with np.inf values.")
                # Using np.inf bypasses the neuralforecast missing value checker
                # but will throw an error if these values are used in computation.
                # It is unlikely that models will use historic exogenous signals
                # for prediction and so this is an acceptable solution for now.
                prior_exog_states = np.full(
                    (len(prior_t), exog.shape[1]), np.inf)
            
        # Compute timestep size.
        dt = prior_t[1] - prior_t[0]

        if not np.all(np.isclose(np.diff(prior_t), dt)):
                raise ValueError(
                    "`last_window` must have equally spaced time values.")
            
        forecast_times = np.arange(0, steps) * dt + prior_t[-1]


        endog_pred = self.method.predict(
            t=forecast_times,
            prior_endog_states=prior_endog_states,
            prior_exog_states=prior_exog_states,
            prior_t=prior_t,
            prediction_exog=prediction_exog,
            rng=DEFAULT_RANGE, # TODO: Figure out randomness.
        )
        preds = pd.DataFrame(endog_pred)
        preds.set_index(forecast_times)
        return preds
    

    def set_params(self, params):
        self.method.set_params(**params)
        self.window_size = self.method.get_window_size()


    
def grid_search(
    method_type: Type[BaseInferenceMethod],
    method_params: Dict[str, Any],
    param_grid: Dict[str, List[Any]],
    t: np.ndarray,
    endog_states: np.ndarray,
    exog_states: Optional[np.ndarray] = None,
    initial_train_window_percent: float = 0.2,
    predict_percent: float = 0.2,
    refit: Union[bool, int] = False
) -> Tuple[BaseInferenceMethod, pd.DataFrame]:
    """Tunes hyperparameters using skforecast.

    Args:
        method_type (Type[BaseInferenceMethod]): The method to be
            used for prediction.
        method_params (dict): A dictionary of default parameters for the method.
        param_grid (dict): The parameter grid for a skforecast grid search.
        t (ndarray): An array of time points with shape (m,).
        endog_states (ndarray): An array of endogenous signals with shape 
            (m, p). Rows are observations and columns are variables. Rows
            correspond to the times in `t`. Optional.
        exog_states (ndarray): An array of exogenous signals. with shape (m, k).
            Rows are observations and columns are variables. Rows correspond to
            the times in `t`. 
        initial_train_window_percent (float): A number between zero and one that
            denotes the percentage of endogenous states that will be used for the first moving window.
        predict_percent (float): A number between zero and one that denotes the 
            percentage of endogenous states that will be predicted for each
            sliding window.
        refit (bool): Whether to re-fit the forecaster in each iteration. If
            refit is an integer, the Forecaster will be trained every that
            number of iterations. 

    Returns:
        best_method: An initialization of the given method with the best       
            parameters.
        gs_results: A dataframe of grid search results.
    """
    # Transform interfere time series to skforecast format.
    endog_skf = pd.DataFrame(endog_states)
    endog_skf = endog_skf.set_index(t)

    if exog_states is not None:
        exog_skf = pd.DataFrame(exog_states)
        exog_skf = exog_skf.set_index(t)
    
    else:
        exog_skf = None

    # Initialize the interfere <-> skforecast adapter.
    # This is a hacky way to use skforecast for other predictive methods.
    # Scroll up to the definition of the ForecasterAutoregMultiSeriesCustom 
    # class for more details.
    adapter = ForecasterAutoregMultiSeriesCustom(method_type, method_params)

    n_obs = len(endog_skf)
    initial_train_size = int(initial_train_window_percent * n_obs)
    steps = int(predict_percent * n_obs)

    # Run the grid search.
    gs_results = grid_search_forecaster_multiseries(
        adapter,
        endog_skf,
        param_grid,
        exog=exog_skf,
        steps=steps,
        refit=refit,

        # This is an important argument and it connects to 
        # ForecasterAutoregMultiSeriesCustom.window_size.
        initial_train_size=initial_train_size, 

        metric="mean_squared_error",
        verbose=False,
        n_jobs="auto",

        # Do not touch these. Grid search will break.
        allow_incomplete_fold=False,
        levels=None,
    )

    # Return the best method and results.
    best_method = adapter.method
    return best_method, gs_results


###############################################################################
# Counterfactual Forecasts
###############################################################################

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
        method_type (Type[BaseInferenceMethod]): The method to be
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

    prior_endog_states, prior_exog_states = intervention.split_exogeneous(X)

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