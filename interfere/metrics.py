from abc import ABC
from typing import Iterable

import numpy as np
import scipy as sp
from sktime.performance_metrics.forecasting import mean_squared_scaled_error

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

    
    def __call__(self,
        X: np.ndarray,
        X_do: np.ndarray,
        X_do_pred:np.ndarray,
        intervention_idxs: Iterable[int],
        **kwargs
    ):
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


        Returns:
            score (float): A scalar score.
        """
        raise NotImplementedError


class DirectionalChangeBinary(CounterfactualForecastingMetric):

    def __init__(self):
        super().__init__("Directional Change (Increase or decrease)")

    @copy_doc(CounterfactualForecastingMetric.__call__)
    def __call__(self,
        X: np.ndarray,
        X_do: np.ndarray,
        X_do_pred:np.ndarray,
        intervention_idxs: Iterable[int],
        **kwargs
    ):
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
    def __call__(self,
        X: np.ndarray,
        X_do: np.ndarray,
        X_do_pred:np.ndarray,
        intervention_idxs: Iterable[int],
        p_val_cut=0.01
    ):
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
    def __call__(self,
        X: np.ndarray,
        X_do: np.ndarray,
        X_do_pred:np.ndarray,
        intervention_idxs: Iterable[int],
        **kwargs
    ):
        X_resp, X_do_resp, pred_X_do_resp = self.drop_intervention_cols(
            intervention_idxs, *[X, X_do, X_do_pred])
        return mean_squared_scaled_error(
            X_do_resp, pred_X_do_resp, y_train=X_resp)


class ValidPredictionTime(CounterfactualForecastingMetric):

    def __init__(self):
        """Initializes a vaid prediction time metric. Returns the index where
        the absolute difference between the target and the predicted is greater
        than a threshold.
        """
        super().__init__("Valid Prediction Time")
        self.eps_max = 0.5


    @copy_doc(CounterfactualForecastingMetric.__call__)
    def __call__(self,
        X: np.ndarray,
        X_do: np.ndarray,
        X_do_pred:np.ndarray,
        intervention_idxs: Iterable[int],
        **kwargs
    ):
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