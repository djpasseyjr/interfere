from typing import Any, Callable, Dict, List, Tuple, Type
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

from .interventions import ExogIntervention
from .base import generate_counterfactual_dynamics


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
    

def score_extrapolation_method(
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


def benchmark(dynamic_model_args, score_method_args):
    obs, cntrfactuals = generate_counterfactual_dynamics(**dynamic_model_args)
    all_scores = []
    all_best_ps = []
    all_X_do_preds = []

    intervention=dynamic_model_args["intervention_type"](
            **dynamic_model_args["intervention_params"])
    
    for i in range(len(obs)):
        score, best_ps, X_do_pred = score_extrapolation_method(
            obs[i],
            cntrfactuals[i],
            dynamic_model_args["time_points_iter"][i],
            intervention,
            **score_method_args
        )
        all_scores.append(score)
        all_best_ps.append(best_ps)
        all_X_do_preds.append(X_do_pred)

    return all_scores, all_best_ps, all_X_do_preds