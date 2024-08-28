"""Base objects for methods. 

Method classes should inheret from sktime.forecasting.base.BaseForecaster
and implement the `sktime` required functions as outlined in the template
below.

In addition, methods must implement the method, `counterfactual_forecast`
in order to be compatible with the  `interfere` API.


The template below outlines the forecaster methods required to integrate with
sktime and sklearn taken. The `CounterfacualForecastMethodTemplate` taken from
`sktime.sktime.extension_templates.forecasting_supersimple` and modified.
"""


from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sktime.forecasting.base import BaseForecaster

from ..interventions import ExogIntervention
from ..base import DEFAULT_RANGE


class CounterfacualForecastMethodTemplate(BaseForecaster):
    """Custom forecaster. todo: write docstring.

    todo: describe your custom forecaster here

    Parameters
    ----------
    parama : int
        descriptive explanation of parama
    paramb : string, optional (default='default')
        descriptive explanation of paramb
    paramc : boolean, optional (default= whether paramb is not the default)
        descriptive explanation of paramc
    and so on
    """

    _tags = {
        "scitype:y": "both",
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
    }

    # todo: add any hyper-parameters and components to constructor
    def __init__(self, parama, paramb="default", paramc=None):
        # todo: write any hyper-parameters to self
        self.parama = parama
        self.paramb = paramb
        self.paramc = paramc
        # IMPORTANT: the self.params should never be overwritten or mutated from now on
        # for handling defaults etc, write to other attributes, e.g., self._parama

        # leave this as is

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.parama etc
        # instead, write to self._parama, self._newparam (starting with _)

    # todo: implement this, mandatory
    def _fit(self, y, X=None, fh=None):
        """Fit reservoir computer to training data.

        Private sktime BaseForecaster._fit containing the core logic, called
        from BaseForecaster.fit.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Args:
            y (np.ndarray): A time series. Rows are samples, columns are variables.
            fh (ForecastingHorizon): The forecasting horizon with the steps ahead to
                predict. 
            X (np.ndarray): Exogeneous time series to fit to.

        Returns:
            self : reference to self
        """
        # any model parameters should be written to attributes ending in "_"
        #  attributes set by the constructor must not be overwritten
        #
        # todo:
        # insert logic here
        # self.fitted_model_param_ = sthsth
        #
        return self

        # IMPORTANT: avoid side effects to y, X, fh
        #
        # Note: when interfacing a model that has fit, with parameters
        #   that are not data (y, X) or forecasting-horizon-like,
        #   but model parameters, *don't* add as arguments to fit, but treat as follows:
        #   1. pass to constructor,  2. write to self in constructor,
        #   3. read from self in _fit,  4. pass to interfaced_model.fit in _fit


    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.DataFrame
            Point predictions
        """
        # todo
        # to get fitted model params set in fit, do this:
        #
        # fitted_model_param = self.fitted_model_param_

        # todo: add logic to compute values
        # values = sthsthsth

        # then return as pd.DataFrame
        # below code guarantees the right row and column index
        #
        # row_idx = fh.to_absolute_index(self.cutoff)
        # col_idx = self._y.index
        #
        # y_pred = pd.DataFrame(values, index=row_ind, columns=col_idx)

        # IMPORTANT: avoid side effects to X, fh


    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """

        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries.
        # Testing parameter choice should cover internal cases well.
        #   for "simple" extension, ignore the parameter_set argument.
        #
        # this method can, if required, use:
        #   class properties (e.g., inherited); parent class test case
        #   imported objects such as estimators from sktime or sklearn
        # important: all such imports should be *inside get_test_params*, not at the top
        #            since imports are used only at testing time
        #
        # A good parameter set should primarily satisfy two criteria,
        #   1. Chosen set of parameters should have a low testing time,
        #      ideally in the magnitude of few seconds for the entire test suite.
        #       This is vital for the cases where default values result in
        #       "big" models which not only increases test time but also
        #       run into the risk of test workers crashing.
        #   2. There should be a minimum two such parameter sets with different
        #      sets of values to ensure a wide range of code coverage is provided.
        #
        # example 1: specify params as dictionary
        # any number of params can be specified
        # params = {"est": value0, "parama": value1, "paramb": value2}
        #
        # example 2: specify params as list of dictionary
        # note: Only first dictionary will be used by create_test_instance
        # params = [{"est": value1, "parama": value2},
        #           {"est": value3, "parama": value4}]
        #
        # return params


    def counterfactual_forecast(
        self,
        X_historic: np.ndarray,
        historic_times: np.ndarray,
        forecast_times: np.ndarray,
        intervention: ExogIntervention
    ):
        """Makes a forecast in the prescence of an intervention.

        TODO: Move this to ..benchmarking.py 

        Args:
            X_historic (np.array): Historic data.
            historic_times (np.ndarray): Historic time points
            forecast_times (np.ndarray): Time points to forecast.
            intervention (ExogIntervention): The intervention to apply at each
                future time point.
        """
        raise NotImplementedError()
    

    def simulate(
        self,
        initial_condition: np.ndarray,
        time_points: np.ndarray,
        exog_X: Optional[np.ndarray] = None,
        rng: np.random.RandomState = DEFAULT_RANGE
    ):
        """Runs a simulation of the dynamics of a fitted forcasting method.

        Args:
            initial_condition: A (n,) array of the initial condition
                of the dynamic model.
            time_points: A (m,) array of the time points where the dynamic 
                model will be simulated.
            exog_X: An optional (m, k) matrix of exogenous inputs. The m rows
                are observations and k columns are variables.
            rng: A numpy random state for reproducibility. (Uses numpy's mtrand 
                random number generator by default.)

        Returns:
            X_sim: A (m, n) array of containing a multivariate time series. The
            rows are observations correstponding to entries in `time_points` and
            the columns correspod to the endogenous cariables in the forecasting method.
        """
        raise NotImplementedError()
    
    
    def get_test_params():
        """Returns a dict of parameters for testing."""
        raise NotImplementedError()
    

    def get_test_param_grid():
        """Returns a small grid for testing hyper parameter optimization."""
        raise NotImplementedError()