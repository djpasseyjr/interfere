from abc import ABC
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
)
from warnings import warn

import numpy as np
import pandas as pd
import scipy as sp
from skforecast.model_selection_multiseries import grid_search_forecaster_multiseries
from sktime.performance_metrics.forecasting import mean_squared_scaled_error

from .base import (
    generate_counterfactual_dynamics,
    generate_counterfactual_forecasts,
    DEFAULT_RANGE
)
from .interventions import ExogIntervention
from .utils import copy_doc
from .methods import BaseInferenceMethod
from .methods.neuralforecast_methods import NeuralForecastAdapter


class CounterfactualForecastingMetric(ABC):

    def __init__(self, name):
        """Initializes a metric for counterfactual forecasting.
        """
        self.name = name


    def drop_intervention_cols(self, intervention_idxs: Iterable[int], *Xs):
        """Remove intervention columns for each array in `args`
        
        Args:
            intervention_ids (Iterable[int]): A list of the indexes of columns
                that contain the exogeneous intervention.
            Xs (Iterable[np.ndarray]): An iterable containing numpy arrays    
                with dimension (m_i, n). They should all have the same number of
                columns but can have a variable number of rows.

            Returns:
                Xs_response (Iterable[np.ndarray]): Every array in `Xs` with the
                    columns corresponding to the indexes in `intervention_idxs`
                    removed.  
        """

        # Check that all arrays have the same number of columns. 
        if  len(set([X.shape[1] for X in Xs])) != 1:
            raise ValueError(
                "All input arrays must have the same number of columns.")
        
        return  [np.delete(X, intervention_idxs, axis=1) for X in Xs]

    
    def __call__(self, X, X_do, X_do_pred, intervention_idxs):
        """Scores the ability to forecast the counterfactual.

        Args:
            X (np.ndarray): An (m, n) matrix that is interpreted to be a  
                realization of an n dimensional stochastic multivariate
                timeseries sampled at m points
            X_do (np.ndarray): A (m, n) maxtix. The ground truth 
                counterfactual, what X would be if the intervention was applied.
            X_do_pred (np.ndarray):  A (m, n) maxtix. The PREDICTED 
                counterfactual, what X would be if the intervention was applied.
            intervention_idxs (List[int]): Which columns of X, X_do, X_do_pred.
                received the intervention.
            t (np.ndarray): A, (m,) array of time points associated with the
                time series.

        Returns:
            score (float): A scalar score.
        """
        raise NotImplementedError


class DirectionalChangeBinary(CounterfactualForecastingMetric):

    def __init__(self):
        super().__init__("Directional Change (Increase or decrease)")

    @copy_doc(CounterfactualForecastingMetric.__call__)
    def __call__(self, X, X_do, X_do_pred, intervention_idxs, **kwargs):
        
        # Drop intervention column
        X_resp, X_do_resp, pred_X_do_resp = self.drop_intervention_cols(
            intervention_idxs, *[X, X_do, X_do_pred])
            
        # Compute time average
        X_avg = np.mean(X_resp, axis=0)
        X_do_avg = np.mean(X_do_resp, axis=0)
        pred_X_do_avg = np.mean(pred_X_do_resp, axis=0)

        # Compute sign of the difference
        sign_of_true_diff = (X_do_avg - X_avg) > 0
        sign_of_pred_diff = (pred_X_do_avg - X_avg) > 0

        # Return number of signals correct
        acc = np.mean(sign_of_true_diff == sign_of_pred_diff)
        return acc
    

class TTestDirectionalChangeAccuracy(CounterfactualForecastingMetric):

    def __init__(self):
        super().__init__("T-Test Directional Change")


    @copy_doc(CounterfactualForecastingMetric.__call__)
    def __call__(self, X, X_do, X_do_pred, intervention_idxs, p_val_cut=0.01):
        """Measures if the forecast correctly predicts the change in the mean
        value of the time series in response to the intervention. 

        The direction of change and whether a change can be inferred is computed
        via a t-test.

        Args:
            X (np.ndarray): An (m, n) matrix that is interpreted to be a  
                realization of an n dimensional stochastic multivariate
                timeseries sampled at m points
            X_do (np.ndarray): A (k, n) maxtix. The ground truth 
                counterfactual, what X would be if the intervention was applied.
            X_do_pred (np.ndarray):  A (k, n) maxtix. The PREDICTED 
                counterfactual, what X would be if the intervention was applied.
            intervention_idxs (List[int]): Which columns of X, X_do, X_do_pred.
                received the intervention.
            t (np.ndarray): A, (m,) array of time points associated with the
                time series.
            p_val_cut (float): The cutoff for statistical significance.

        Returns:
            score (float): A scalar score.
        """
        
        # Drop intervention columns and get response columns.
        X_resp, X_do_resp, pred_X_do_resp = self.drop_intervention_cols(
            intervention_idxs, *[X, X_do, X_do_pred])
        
        true_direct_chg = self.directional_change(X_resp, X_do_resp, p_val_cut)
        pred_direct_chg = self.directional_change(
            X_resp, pred_X_do_resp, p_val_cut)
        
        return np.mean(true_direct_chg == pred_direct_chg)
        
    
    def directional_change(self, X: np.ndarray, Y: np.ndarray, p_val_cut):
        """Return sign of the difference in mean across all columns of X and Y.

        Args:
            X (np.ndarray): A (m x n) array.
            Y (np.ndarray): A (k x n) array.
            p_value_cut (float): The cutoff for statistical significance.
            
        Returns:
            estimated_change (np.ndarray): A 1d array with length equal to the
                number of columns in X and Y. Each entry of `estimated_change`
                can take on one of three values, 1, -1, or 0. If the ith entry
                of `estimated_change` is 1, then the mean of X[:, i] is greater
                than the mean of Y[:, i] (positive t-statistic). A -1 denotes
                that the mean of X[:, i] is less than the mean of Y[:, i]
                (negative t-statistic) and a 0 means that no statistically
                significant change was detected given the p-value cutoff for the t-test.
        """
        true_ttest_result = sp.stats.ttest_ind(X, Y, axis=0, equal_var=False)
        
        # Extract t-statistic
        estimated_change = true_ttest_result.statistic

        # Zero out where p-value is above the cutoff
        estimated_change *= (true_ttest_result.pvalue < p_val_cut)

        # Keep only whether the change in mean was positive, negative or no
        # change (zero)
        estimated_change[estimated_change < 0] = -1
        estimated_change[estimated_change > 0] = 1

        return estimated_change
        
            
class RootMeanStandardizedSquaredError(CounterfactualForecastingMetric):

    def __init__(self):
        super().__init__("Room Mean Standardized Squared Error")


    @copy_doc(CounterfactualForecastingMetric.__call__)
    def __call__(self, X, X_do, X_do_pred, intervention_idxs, **kwargs):
        X_resp, X_do_resp, pred_X_do_resp = self.drop_intervention_cols(
            intervention_idxs, *[X, X_do, X_do_pred])
        return mean_squared_scaled_error(
            X_do_resp, pred_X_do_resp, y_train=X_resp)


class ValidPredictionTime(CounterfactualForecastingMetric):

    def __init__(self):
        super().__init__("Valid Prediction Time")
        self.eps_max = 0.5


    @copy_doc(CounterfactualForecastingMetric.__call__)
    def __call__(self, X, X_do, X_do_pred, intervention_idxs, **kwargs):

        eps_max = kwargs.get("eps_max", self.eps_max)

        X_do_resp, pred_X_do_resp = self.drop_intervention_cols(
            intervention_idxs, X_do, X_do_pred)
        
        # Compute infinity norm of error for each point in time
        inf_norm_err = np.max(np.abs(X_do_resp - pred_X_do_resp), axis=1)
        idxs, = (inf_norm_err > eps_max).nonzero()

        if len(idxs) == 0:
            return len(inf_norm_err)
        
        vpt = idxs.min()
        return vpt
        

    


def directional_accuracy(X, X_do, pred_X_do, intervention_idx):
    # TODO Fix this docstring
    """Predict the correct directional change of each non intervened signal in
    response to the intervention relative to the counterfactual, non-intervened
    system.

    For example, if a system has three variables, $x(t)$, $y(t)$ and $z(t)$ and
    we intervene on $x(t)$ to produce $y_\text{do}(t)$ and $z_\text{do}(t)$, we
    evaluate the method's ability to estimate the sign of the difference in the
    time average of the signals. E.g.
    $$
    \frac{1}{T}\int_0^T  y_\text{do}(t) dt - \frac{1}{T}\int_0^T y(t)dt 
    $$
    """
    # Drop intervention column
    X_resp = np.delete(X, intervention_idx, axis=1)
    X_do_resp = np.delete(X_do, intervention_idx, axis=1)
    pred_X_do_resp = np.delete(pred_X_do, intervention_idx, axis=1)

    # Compute time average
    X_avg = np.mean(X_resp, axis=0)
    X_do_avg = np.mean(X_do_resp, axis=0)
    pred_X_do_avg = np.mean(pred_X_do_resp, axis=0)

    # Compute sign of the difference
    sign_of_true_diff = (X_do_avg - X_avg) > 0
    sign_of_pred_diff = (pred_X_do_avg - X_avg) > 0

    # Return number of signals correct
    acc = np.mean(sign_of_true_diff == sign_of_pred_diff)
    return acc


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
            endog_states=series.values,
            t=t,
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
        
        historic_times = last_window.index.values
        historic_endog = last_window.values
        exog_states = None
        historic_exog = None

        if exog is not None:
            # If the prediction involves exogeneous data build exog
            # arrays for interfere methods.
            exog_states = exog.values

            # Build historic exogeneous array. TODO: Pass true exog states.
            if set(historic_times).issubset(self._exog.index):        
                historic_exog = np.vstack([
                    self._exog.loc[ti].values
                    for ti in historic_times
                ])
            else:
                warn("Missing exogeneous data detected in grid gearch."
                    " Replacing with np.inf values.")
                # Using np.inf bypasses the neuralforecast missing value checker
                # but will throw an error if these values are used in computation.
                # It is unlikely that models will use historic exogenous signals
                # for prediction and so this is an acceptable solution for now.
                historic_exog = np.full(
                    (len(historic_times), exog.shape[1]), np.inf)
            
        # Compute timestep size.
        dt = historic_times[1] - historic_times[0]

        if not np.all(np.isclose(np.diff(historic_times), dt)):
                raise ValueError(
                    "`last_window` must have equally spaced time values.")
            
        forecast_times = np.arange(0, steps) * dt + historic_times[-1]


        endog_pred = self.method.predict(
            forecast_times,
            historic_endog,
            exog_states,
            historic_exog=historic_exog,
            historic_times=historic_times,
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
    endog_states: np.ndarray,
    t: np.ndarray,
    exog_states: Optional[np.ndarray] = None,
    initial_train_window_percent: float = 0.2,
    predict_percent: float = 0.2,
    refit: Union[bool, int] = False
) -> Tuple[BaseInferenceMethod, pd.DataFrame]:
    """Tunes hyperparameters using skforecast.

    Args:
        method_type (sktime.forecasting.base.BaseForecaster): The method to be
            used for prediction.
        method_params (dict): A dictionary of default parameters for the method.
        param_grid (dict): The parameter grid for a skforecast grid search.
        endog_states: An (m, n) array of endogenous signals. Sometimes
            called Y. Rows are observations and columns are variables. Each
            row corresponds to the times in `t`.
        t: An (m,) array of time points.
        exog_states: An (m, k) array of exogenous signals. Sometimes called
            X. Rows are observations and columns are variables. Each row 
            corresponds to the times in `t`.
        initial_train_window_percent: A number between zero and one that denotes
            the percentage of endogenous states that will be used for the first
            moving window.
        predict_percent: A number between zero and one that denotes the 
            percentage of endogenous states that will be predicted for each
            sliding window.
        refit:
            Whether to re-fit the forecaster in each iteration. If refit is an integer, the Forecaster will be trained every that number of iterations.

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
# Historic Counterfactuals.
###############################################################################

def score_counterfactual_extrapolation_method(
    X: np.ndarray,
    X_do: np.ndarray,
    time_points: np.ndarray,
    intervention: ExogIntervention,
    method_type: Type,
    method_params: Dict[str, Any],
    method_param_grid: Dict[str, Any],
    num_intervention_sims: int,
    score_function: Callable,
    score_function_args: Dict[str, Any]
) -> Tuple[List[float], Dict[str, Any]]:
    """Scores a method's ability to extrapolate about interventions.

    Args:
        X (np.ndarray): An (m, n) matrix that is interpreted to be a  
            realization of an n dimensional stochastic multivariate timeseries.
        X_do (np.ndarray): The ground truth counterfactual, what X would look
            like, if the intervention had been applied.
        time_point (np.ndarray): 1D array of time points corresponding to the
            rows of X and X_do.
        intervention (interfere.Intervention): The type of the intervention to
            apply.
        method_type (sktime.forecasting.base.BaseForecaster): The method to be
            used for prediction.
        method_params (dict): A dictionary of default parameters for the method.
        method_param_grid (dict): The parameter grid for a sklearn grid search.
        num_intervention_preds (int): The number of interventions to simulate
            with the method. Used for noisy methods.
        score_function (Callable): Must be able to accept the followin
            combinations of parameters: 
                score_function(X, X_do, X_do_pred, **score_function_args)
            and return a single float.
        score_function_args: A dictionary of keywords args for the score
            function.

    Returns:
        scores: A list with num_intervention_preds entries containing the score for each
        intervention simulation.
        best_params: A dictionary of the best hyper parameters found.
    """

    # Perform hyper parameter optimization to find optimal parameters.
    best_params = tune_method(
        method_type, method_params, method_param_grid, X, time_points)

    # Combine best params with default params. (Items in best_params will
    # overwrite items in method_params if there are duplicates here.)
    sim_params = {**method_params, **best_params}

    # Simulate intervention
    X_do_predictions = [
        method_type(**sim_params).simulate_counterfactual(
            X,    
            time_points,
            intervention, 
        )
        for i in range(num_intervention_sims)
    ]

    # Score the responses predicted by the mehtod
    scores = [
        score_function(X, X_do, X_do_pred, **score_function_args)
        for X_do_pred in X_do_predictions
    ]

    return scores, best_params, X_do_predictions


def counterfactual_extrapolation_benchmark(
    gen_cntftl_args: Dict[str, Any],  score_cntftl_method_args: Dict[str, Any]):
    """Scores if inference method can simulate a counterfactual scenario.

    This asks the question: "What would have happened in the past if an
    intervention had taken place?". How would this differ from what was
    observed. It tests the ability of the inference method to extapolate
    outside the bounds of what was observed.

    Args:
        gen_cntftl_args (Dict[str, Any]): Contains all arguments for the
            function `interfere.base.generate_counterfactual_dynamics`.
        score_cntftl_method_args (Dict[str, Any]): Contains the following
            arguments for the function 
            `interfere.benchmarking.score_counterfactual_extrapolation_method`
            * method_type: Type,
            * method_params: Dict[str, Any],
            * method_param_grid: Dict[str, Any],
            * num_intervention_sims: int,
            * score_function: Callable,
            * score_function_args: Dict[str, Any]
    """
    obs, cntrfactuals = generate_counterfactual_dynamics(**gen_cntftl_args)
    all_scores = []
    all_best_ps = []
    all_X_do_preds = []

    intervention=gen_cntftl_args["intervention_type"](
            **gen_cntftl_args["intervention_params"])
    
    for i in range(len(obs)):
        score, best_ps, X_do_pred = score_counterfactual_extrapolation_method(
            obs[i],
            cntrfactuals[i],
            gen_cntftl_args["time_points_iter"][i],
            intervention,
            **score_cntftl_method_args
        )
        all_scores.append(score)
        all_best_ps.append(best_ps)
        all_X_do_preds.append(X_do_pred)

    return all_scores, all_best_ps, all_X_do_preds


###############################################################################
# Counterfactual Forecasts
###############################################################################

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
        method_type (sktime.forecasting.base.BaseForecaster): The method to be
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

    historic_endog, historic_exog = intervention.split_exogeneous(X)

    hist_exog_shape = None
    if historic_exog is not None:
        hist_exog_shape = historic_exog.shape

    print(f""""
          
Grid search split:
          
historic_endog: {historic_endog.shape}
historic_exog: {hist_exog_shape}
""")

    if best_params is None:

        _, gs_results = grid_search(
            method_type, method_params, method_param_grid,
            historic_endog, time_points, historic_exog,
            # Uses a moving window of train on 20% of the data, predict next 20%
            initial_train_window_percent = 0.2,
            predict_percent = 0.2,
            # Refits the forecaster every time. This must be set to 1 in order
            # to use statsforecast models bc they store historic exog during
            # fit. 
            refit=1
        )

        best_params = gs_results.params[gs_results.mean_squared_error.argmin()]

    # Combine best params with default params. (Items in best_params will
    # overwrite items in method_params if there are duplicates here.)
    sim_params = {**method_params, **best_params}

    # Simulate intervention
    method = method_type(**sim_params)
    method.fit(historic_endog, time_points, historic_exog)
    
    X_do_preds = [
        method.simulate(
            forecast_times,
            historic_states=X,
            intervention=intervention,
            historic_times=time_points,
            rng=rng
        )
        for i in range(num_intervention_sims)
    ]
    return X_do_preds, sim_params


def score_counterfactual_forecast_method(
    X: np.ndarray,
    X_do_forecast: np.ndarray,
    time_points: np.ndarray,
    intervention: ExogIntervention,
    method_type: Type,
    method_params: Dict[str, Any],
    method_param_grid: Dict[str, Any],
    num_intervention_sims: int,
    score_function: Callable,
    score_function_args: Dict[str, Any]
) -> Tuple[List[float], Dict[str, Any]]:
    """Scores a method's ability to extrapolate about interventions.

    Args:
        X (np.ndarray): An (m, n) matrix that is interpreted to be a  
            realization of an n dimensional stochastic multivariate timeseries.
        X_do_forecast (np.ndarray): An (p, n) maxtix. The ground truth 
            counterfactual, what the last p rows of X would be if the
            intervention was applied starting at time `time_points[-p]`.
        time_points (np.ndarray): 1D array of time points corresponding to the
            rows of X. The last p entries correspond to the rows of
            X_do_forecast.
        intervention (interfere.Intervention): The type of the intervention to
            apply.
        method_type (sktime.forecasting.base.BaseForecaster): The method to be
            used for prediction.
        method_params (dict): A dictionary of default parameters for the method.

        method_param_grid (dict): The parameter grid for a sklearn grid search.
        num_intervention_preds (int): The number of interventions to simulate
            with the method. Used for noisy methods.
        score_function (Callable): Must be able to accept the followin
            combinations of parameters: 
                score_function(X, X_do, X_do_pred, **score_function_args)
            and return a single float.
        score_function_args: A dictionary of keywords args for the score
            function.

    Returns:
        scores: A list with num_intervention_preds entries containing the score for each
        intervention simulation.
        best_params: A dictionary of the best hyper parameters found.
    """
    # Simulate intervention
    X_do_forecast_predictions, best_params = forecast_intervention(
        X,
        X_do_forecast,
        time_points,
        intervention,
        method_type,
        method_params,
        method_param_grid,
        num_intervention_sims
    )
    
    p = X_do_forecast.shape[0]
    X_forecast = X[-p:, :]

    # Score the responses predicted by the method
    scores = [
        score_function(
            X_forecast, X_do_forecast, X_do_forecast_pred, **score_function_args)
        for X_do_forecast_pred in X_do_forecast_predictions
    ]

    return scores, best_params, X_do_forecast_predictions
    

def counterfactual_forecast_benchmark(
        gen_cntftl_args: Dict[str, Any], score_cntftl_method_args: Dict[str, Any]):
    """Scores if inference method can simulate a counterfactual forecast.

    This asks the question: "What will happened next if an
    intervention is applied?". How will this differ from what would have
    happened without the intervention. This tests the ability of the inference
    method to extapolate correctly about probable futures.

    Args:
        gen_cntftl_args (Dict[str, Any]): Contains all arguments for the
            function `interfere.base.generate_counterfactual_dynamics`.
        score_cntftl_method_args (Dict[str, Any]): Contains the following
            arguments for the function 
            `interfere.benchmarking.score_counterfactual_forecast_method`
            * method_type: Type,
            * method_params: Dict[str, Any],
            * method_param_grid: Dict[str, Any],
            * num_intervention_sims: int,
            * score_function: Callable,
            * score_function_args: Dict[str, Any]
    """
    obs, cntftl_forecasts, time_points = generate_counterfactual_forecasts(**gen_cntftl_args)
    all_scores = []
    all_best_ps = []
    all_X_do_preds = []

    intervention=gen_cntftl_args["intervention_type"](
            **gen_cntftl_args["intervention_params"])
    
    for i in range(len(obs)):
        score, best_ps, X_do_pred = score_counterfactual_forecast_method(
            obs[i],
            cntftl_forecasts[i],
            time_points,
            intervention,
            **score_cntftl_method_args
        )
        all_scores.append(score)
        all_best_ps.append(best_ps)
        all_X_do_preds.append(X_do_pred)

    return all_scores, all_best_ps, all_X_do_preds