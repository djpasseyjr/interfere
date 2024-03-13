from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type
from abc import ABC

import numpy as np
import scipy as sp
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sktime.performance_metrics.forecasting import mean_squared_scaled_error

from .interventions import ExogIntervention
from .base import (
    generate_counterfactual_dynamics, generate_counterfactual_forecasts
)
from .utils import copy_doc


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


def tune_method(method_type, method_params, method_param_grid, X, time_points):
    """Tune hyperparameters using sklearn."""
    grid_search = GridSearchCV(
        method_type(**method_params),
        method_param_grid,
        cv=TimeSeriesSplit(n_splits=5)
    )
    grid_search.fit(X)
    return grid_search.best_params_
    
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
    X_do_forecast: np.ndarray,
    time_points: np.ndarray,
    intervention: ExogIntervention,
    method_type: Type,
    method_params: Dict[str, Any],
    method_param_grid: Dict[str, Any],
    num_intervention_sims: int,
    best_params: Optional[Dict["str", Any]] = None,
) -> Tuple[List[float], Dict[str, Any]]:
    """Tunes, fits and forecasts the effect of an intervention with a method.

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
        method_param_grid (dict): The parameter grid for a sklearn grid search.
        num_intervention_preds (int): The number of interventions to simulate
            with the method. Used for noisy methods.
        best_params (dict): An optional dictionary of the top hyper parameters.
    """
    p = X_do_forecast.shape[0]
    historic_times = time_points[:-p]
    forecast_times = time_points[-p:]
    X_historic = X[:-p, :]

    if best_params is None:
        # Perform hyper parameter optimization to find optimal parameters.
        best_params = tune_method(
            method_type, method_params, method_param_grid,
            X_historic, historic_times
        )

    # Combine best params with default params. (Items in best_params will
    # overwrite items in method_params if there are duplicates here.)
    sim_params = {**method_params, **best_params}

    # Simulate intervention
    X_do_forecast_predictions = [
        method_type(**sim_params).counterfactual_forecast(
            X_historic,
            historic_times,
            forecast_times,
            intervention, 
        )
        for i in range(num_intervention_sims)
    ]
    return X_do_forecast_predictions, sim_params

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