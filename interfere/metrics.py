from abc import ABC
from typing import Iterable, Optional, Union

import numpy as np
import scipy as sp
from .interventions import ExogIntervention
from .utils import copy_doc

EPSILON = 1e-10


class CounterfactualForecastingMetric(ABC):

    def __init__(self, name):
        """Initializes a metric for counterfactual forecasting.
        """
        self.name = name


    def drop_intervention_cols(
            self,
            intervention_idxs: Union[ExogIntervention, Iterable[int]],
            *Xs
        ):
        """Remove intervention columns for each array in `args`
        
        Args:
            intervention_idxs (Union[ExogIntervention, Iterable[int]]): An 
                intervention or a list of the indexes of columns that contain
                the exogeneous intervention states.
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
        
        if intervention_idxs is None:
            intervention_idxs = []

        if isinstance(intervention_idxs, ExogIntervention):
            intervention_idxs = intervention_idxs.iv_idxs
        
        return  [np.delete(X, intervention_idxs, axis=1) for X in Xs]

    
    def __call__(self,
        X: np.ndarray,
        X_do: np.ndarray,
        X_do_pred:np.ndarray,
        intervention_idxs: Optional[Iterable[int]] = None,
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
        super().__init__("BDC")

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
        super().__init__("DC")


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
        estimated_change[estimated_change < 0] = -1.0
        estimated_change[estimated_change > 0] = 1.0

        # Get rid of negative zeros.
        estimated_change += 0.0
        return estimated_change
        
            
class RootMeanStandardizedSquaredError(CounterfactualForecastingMetric):

    def __init__(self):
        super().__init__("RMSSE")


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
        return rmsse(
            X_do_resp, pred_X_do_resp)


class ValidPredictionTime(CounterfactualForecastingMetric):

    def __init__(self):
        """Initializes a vaid prediction time metric. Returns the index where
        the absolute difference between the target and the predicted is greater
        than a threshold.
        """
        super().__init__("VPT")
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


class RootMeanSquaredScaledError(CounterfactualForecastingMetric):

    def __init__(self):
        super().__init__("RMSSE")


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
        return rmsse(
            X_do_resp, pred_X_do_resp)


class RootMeanSquaredScaledErrorOverAvgMethod(CounterfactualForecastingMetric):
    def __init__(self):
        """Computes RMSSE(actual, predicted) / RMSSE(actual, mean(training))."""
        super().__init__("RMSSE/RMSSE(AVG)")

    @copy_doc(CounterfactualForecastingMetric.__call__)
    def __call__(self,
        X: np.ndarray,
        X_do: np.ndarray,
        X_do_pred:np.ndarray,
        intervention_idxs: Iterable[int],
        **kwargs
    ):
        rmsse_cntr_metric = RootMeanSquaredScaledError()

        err = rmsse_cntr_metric(
            X, X_do, X_do_pred, intervention_idxs, **kwargs
        )
        X_means = np.vstack([np.mean(X, axis=0) for i in range(X_do.shape[0])])
        avg_err = rmsse_cntr_metric(
            X, X_do, X_means, intervention_idxs, **kwargs
        )
        return err / avg_err
    

def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """
    Percentage error

    Note: result is NOT multiplied by 100
    """
    return _error(actual, predicted) / (actual + EPSILON)


def _naive_forecasting(actual: np.ndarray, seasonality: int = 1):
    """ Naive forecasting method which just repeats previous samples """
    return actual[:-seasonality]


def _relative_error(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Relative Error """
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        if not isinstance(benchmark, int):
            seasonality = 1
        else:
            seasonality = benchmark
        return _error(actual[seasonality:], predicted[seasonality:]) /\
               (_error(actual[seasonality:], _naive_forecasting(actual, seasonality)) + EPSILON)

    return _error(actual, predicted) / (_error(actual, benchmark) + EPSILON)


def _bounded_relative_error(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Bounded Relative Error """
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        if not isinstance(benchmark, int):
            seasonality = 1
        else:
            seasonality = benchmark

        abs_err = np.abs(_error(actual[seasonality:], predicted[seasonality:]))
        abs_err_bench = np.abs(_error(actual[seasonality:], _naive_forecasting(actual, seasonality)))
    else:
        abs_err = np.abs(_error(actual, predicted))
        abs_err_bench = np.abs(_error(actual, benchmark))

    return abs_err / (abs_err + abs_err_bench + EPSILON)


def _geometric_mean(a, axis=0, dtype=None):
    """ Geometric mean """
    if not isinstance(a, np.ndarray):  # if not an ndarray object attempt to convert it
        log_a = np.log(np.array(a, dtype=dtype))
    elif dtype:  # Must change the default dtype allowing array type
        if isinstance(a, np.ma.MaskedArray):
            log_a = np.log(np.ma.asarray(a, dtype=dtype))
        else:
            log_a = np.log(np.asarray(a, dtype=dtype))
    else:
        log_a = np.log(a)
    return np.exp(log_a.mean(axis=axis))


def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))


def nrmse(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Root Mean Squared Error """
    return rmse(actual, predicted) / (actual.max() - actual.min())


def me(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Error """
    return np.mean(_error(actual, predicted))


def mae(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Error """
    return np.mean(np.abs(_error(actual, predicted)))


def mad(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Deviation """
    error = _error(actual, predicted)
    return np.mean(np.abs(error - np.mean(error)))


def gmae(actual: np.ndarray, predicted: np.ndarray):
    """ Geometric Mean Absolute Error """
    return _geometric_mean(np.abs(_error(actual, predicted)))


def mdae(actual: np.ndarray, predicted: np.ndarray):
    """ Median Absolute Error """
    return np.median(np.abs(_error(actual, predicted)))


def mpe(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Percentage Error """
    return np.mean(_percentage_error(actual, predicted))


def mape(actual: np.ndarray, predicted: np.ndarray):
    """
    Mean Absolute Percentage Error

    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0

    Note: result is NOT multiplied by 100
    """
    return np.mean(np.abs(_percentage_error(actual, predicted)))


def mdape(actual: np.ndarray, predicted: np.ndarray):
    """
    Median Absolute Percentage Error

    Note: result is NOT multiplied by 100
    """
    return np.median(np.abs(_percentage_error(actual, predicted)))


def smape(actual: np.ndarray, predicted: np.ndarray):
    """
    Symmetric Mean Absolute Percentage Error

    Note: result is NOT multiplied by 100
    """
    return np.mean(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + EPSILON))


def smdape(actual: np.ndarray, predicted: np.ndarray):
    """
    Symmetric Median Absolute Percentage Error

    Note: result is NOT multiplied by 100
    """
    return np.median(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + EPSILON))


def maape(actual: np.ndarray, predicted: np.ndarray):
    """
    Mean Arctangent Absolute Percentage Error

    Note: result is NOT multiplied by 100
    """
    return np.mean(np.arctan(np.abs((actual - predicted) / (actual + EPSILON))))


def mase(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    """
    Mean Absolute Scaled Error

    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    return mae(actual, predicted) / mae(actual[seasonality:], _naive_forecasting(actual, seasonality))


def std_ae(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Absolute Error """
    __mae = mae(actual, predicted)
    return np.sqrt(np.sum(np.square(_error(actual, predicted) - __mae))/(len(actual) - 1))


def std_ape(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Absolute Percentage Error """
    __mape = mape(actual, predicted)
    return np.sqrt(np.sum(np.square(_percentage_error(actual, predicted) - __mape))/(len(actual) - 1))


def rmspe(actual: np.ndarray, predicted: np.ndarray):
    """
    Root Mean Squared Percentage Error

    Note: result is NOT multiplied by 100
    """
    return np.sqrt(np.mean(np.square(_percentage_error(actual, predicted))))


def rmdspe(actual: np.ndarray, predicted: np.ndarray):
    """
    Root Median Squared Percentage Error

    Note: result is NOT multiplied by 100
    """
    return np.sqrt(np.median(np.square(_percentage_error(actual, predicted))))


def rmsse(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    """ Root Mean Squared Scaled Error """
    q = np.abs(_error(actual, predicted)) / mae(actual[seasonality:], _naive_forecasting(actual, seasonality))
    return np.sqrt(np.mean(np.square(q)))


def inrse(actual: np.ndarray, predicted: np.ndarray):
    """ Integral Normalized Root Squared Error """
    return np.sqrt(np.sum(np.square(_error(actual, predicted))) / np.sum(np.square(actual - np.mean(actual))))


def rrse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Relative Squared Error """
    return np.sqrt(np.sum(np.square(actual - predicted)) / np.sum(np.square(actual - np.mean(actual))))


def mre(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Mean Relative Error """
    return np.mean(_relative_error(actual, predicted, benchmark))


def rae(actual: np.ndarray, predicted: np.ndarray):
    """ Relative Absolute Error (aka Approximation Error) """
    return np.sum(np.abs(actual - predicted)) / (np.sum(np.abs(actual - np.mean(actual))) + EPSILON)


def mrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Mean Relative Absolute Error """
    return np.mean(np.abs(_relative_error(actual, predicted, benchmark)))


def mdrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Median Relative Absolute Error """
    return np.median(np.abs(_relative_error(actual, predicted, benchmark)))


def gmrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Geometric Mean Relative Absolute Error """
    return _geometric_mean(np.abs(_relative_error(actual, predicted, benchmark)))


def mbrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Mean Bounded Relative Absolute Error """
    return np.mean(_bounded_relative_error(actual, predicted, benchmark))


def umbrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Unscaled Mean Bounded Relative Absolute Error """
    __mbrae = mbrae(actual, predicted, benchmark)
    return __mbrae / (1 - __mbrae)


def mda(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Directional Accuracy """
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - actual[:-1])).astype(int))