"""The base class for methods for intervention response prediction.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ..base import DEFAULT_RANGE
from ..interventions import ExogIntervention


class BaseInferenceMethod(BaseEstimator):
    """Base class for interfere inference methods."""

    # Class attribute to determine if the method was fit to data.
    is_fit = False


    def simulate(
        self,
        t: np.ndarray,
        prior_states: np.ndarray,
        prior_t: Optional[np.ndarray] = None,
        intervention:  Optional[ExogIntervention] = None,
        rng: np.random.mtrand.RandomState = DEFAULT_RANGE
    ) -> np.ndarray:
        """Simulates a the intervention response with a fitted method.

        Args:
            t: A (m,) array of times to simulate.
            prior_states: A (p, n + k) array of endogenous and exogenous
                signals. Rows are observations corresponding to
                `t` and columns are variables. The k exogenous
                variable indexes are contained in `intervention.intervened_idxs`
            intervention: An interfere.ExogIntervention.
            prior_t: Optional (p,) array of times corresponding to
                historic observations. Defaults to equally spaced points
                immediately prior to `t`. Assumes that the last
                row of `prior_states` corresponds to the initial condition,
                the state of the system at time `t = t[0]`.
            rng: Numpy random state.

        Returns:
            simulated_states: A (m, n + k) array of simulated exogenous and
                endogenous signals. The k exogenous variable indexes are
                contained in `intervention.intervened_idxs`.
        """
        if intervention is not None:
            (prior_endog_states, 
             prior_exog_states) = intervention.split_exogeneous(prior_states)
            prediction_exog = intervention.eval_at_times(t)

        else:
            prior_endog_states = prior_states
            prior_exog_states = None
            prediction_exog = None

        endo_pred = self.predict(
            t,
            prior_endog_states,
            prior_exog_states=prior_exog_states,
            prior_t=prior_t,
            prediction_exog=prediction_exog,
            rng=rng
        )

        simulated_states = endo_pred

        # Optionally intervention to combine exogeneous and endogenous.
        if intervention is not None:
            simulated_states = intervention.combine_exogeneous(
                endo_pred, prediction_exog)
    
        return simulated_states


# TODO Change order of t and endog. Consider new name for t.
    def fit(
        self,
        t: np.ndarray,
        endog_states: np.ndarray,
        exog_states: np.ndarray = None,
    ):
        """Fits the method using the passed data.
        
        Args:
            t: An (m,) array of time points.
            endog_states: An (m, n) array of endogenous signals. Rows are observations and columns are variables. Each
                row corresponds to the times in `t`.
            exog_states: An (m, k) array of exogenous signals. Rows are
                observations and columns are variables. Each row corresponds to
                the times in `t`.
        """
        # Make sure no Pandas DataFrames are passed in.
        if any([
            isinstance(x, pd.DataFrame) 
            for x in [endog_states, t, exog_states]
        ]):
            raise ValueError("Interfere inference methods do not accept " "DataFrames. Use DataFrame.values and DataFrame.index")
        

        self.is_fit = True
        return self._fit(t, endog_states, exog_states)
    

    def predict(
        self,
        t: np.ndarray,
        prior_endog_states: np.ndarray,
        prior_exog_states: Optional[np.ndarray] = None,
        prior_t: Optional[np.ndarray] = None,
        prediction_exog: Optional[np.ndarray] = None,
        prediction_max: float = 1e9,
        rng: np.random.RandomState = DEFAULT_RANGE,
    ) -> np.ndarray:
        """Runs a simulation of the dynamics of a fitted forcasting method.

       Note: Must call `self.fit(...)` before calling `self.predict`.

        Args:
            t (ndarray): An array of the time points with shape (m,) for the
                method to simulate.
            prior_endog_states (ndarray): Aa array of historic observations of
                the n ENDOGENOUS signals with shape  (p, n). Rows represent 
                observations and columns represent variables This is used as the
                initial condition data or lagged initial conditions. It is NOT
                used to fit the method. If `prior_t` is not provided and
                `t` contains equally spaced points, the `prior_endog_states` are
                assumed to  have occured at equally spaced points prior to `t`.
                Additionally, the last row of `prior_endog_states` must
                be an observation that  occured at the time `t[0]`. When
                `prior_t` is provided it is assumed that the rows of
                `prior_endog_states` were observed at the times contained in
                `prior_t`.
            prior_exog_states: An optional array of historic observations of the
                k EXOGENOUS signals with shape (p, k). Rows contain observations
                and columns contain variables. This is used for the
                initial condition data and lag information. It is NOT used to
                fit the method. If `prior_t` is not provided and
                `t` contains equally spaced points, the `prior_exog_states` are
                assumed to  have occured at equally spaced points prior to `t`.
                Additionally, the last row of `prior_enxog_states` must
                be an observation that  occured at the time `t[0]`. When `prior_t` is provided it is assumed
                that the rows of `prior_endog_states` were observed at the times
                contained in `prior_t`.
            prior_t (ndarray): An optional array with shape (p,) of times
                corresponding to the rows of `prior_endog_states` and `prior_exog_states`. If `prior_t` is not provided and `t`
                contains equally spaced points, then `prior_t` is assumed
                to contain occured at equally spaced points prior to `t`
            prediction_exog: An optional (m, k) array of exogenous signals
                corresponding to the times in `t`. Rows are observations and
                columns are variables.
            prediction_max: A threshold for predicted endogeneous values
                to prevent overflow in predictions. All predictions larger in
                magnitude will be set equal to `prediction_max`.
            rng: An optional numpy random state for reproducibility. (Uses 
                numpy's mtrand random number generator by default.)

        Returns:
            X_sim: A (m, n) array of containing a multivariate time series. The
            rows are observations correstponding to entries in `time_points` and
            the columns correspod to the endogenous variables in the forecasting method.
        """
        if not self.is_fit:
            raise ValueError("Call self.fit(...) before self.predict(...).")
        
        if any([
            isinstance(x, pd.DataFrame) 
            for x in [t, prior_endog_states, prior_exog_states, prior_t, prediction_exog]
        ]):
            raise ValueError("Interfere inference methods do not accept " "DataFrames. Use DataFrame.values and DataFrame.index")
        
        if np.any(prior_endog_states > prediction_max):
            raise ValueError(
                f"Historic endogenous contains values ({np.max(prior_endog_states)}"
                " that are larger than" 
                f" `prediction_max = {prediction_max}`, the "
                "prediction threshold."
                "Increase `prediction_max` in order to simulate these values."
            )
        
        # Reshape prior_endog_states if it was 1D.
        if len(prior_endog_states.shape) == 1:
            prior_endog_states = np.reshape(prior_endog_states, 1, -1)

        # Gather array shapes
        p, k = prior_endog_states.shape
        (m,) = t.shape 

        if len(t) < 2:
            raise ValueError("Since the first timestep is assumed to be the " "current time, and correspond to the last row of `prior_endog_states`," " the `t` must have at least two time values.")
        
        # Create prior_t assuming equal time step size.
        if prior_t is None:
            dt = t[1] - t[0]

            # Check for equally spaced forecast time points.
            if not np.all(np.isclose(np.diff(t), dt)):
                raise ValueError("The `prior_t` argument not provided"
                " AND `t` are equally spaced. Cannot infer "
                " `prior_t`. Either pass it explicitly or provide "
                " equally spaced time `t`.")
            
            prior_t = np.arange(-p, 1) * dt + t[0]


        # Check shape of exogenous signals.
        if prediction_exog is not None:
            m_exog, k_exog = prediction_exog.shape
            if m_exog != m:
                raise ValueError(f"Number of exogenous observations ({m_exog})"
                f" does not match the number of t ({m}).")

        # Compare window size of forecaster with amount of historic data.
        w = self.get_window_size()

        if w > p:
            warn(str(type(self)) + f" has window size {w} but only recieved {p}"
                 " endog observations. Augmenting historic edogenous "
                 "observations with zeros."
            )

            prior_endog_states = np.vstack([
                np.zeros((w - p, k)),
                prior_endog_states
            ])
            
        # Check shape of historic exogenous signals.
        if prior_exog_states is not None:
            p_hexog, k_hexog = prior_exog_states.shape

            if p_hexog != p:
                raise ValueError("Arguments `prior_endog_states` and "
                "`prior_exog_states` must have the same number of rows.")
            
            if prediction_exog is not None:
                if k_hexog != k_exog:
                    raise ValueError("The `prior_exog_states` and `exog` arguments"
                    " must have the same number of columns.")
                
            if w > p_hexog:
                warn(str(type(self)) + f" has window size {w} but only recieved"
                     f" {p_hexog} exog observations. Augmenting historic "
                     "exogenous observations with zeros.")

                prior_exog_states = np.vstack([
                    np.zeros((w - p_hexog, k_hexog)),
                    prior_exog_states
                ])


        endog_pred = self._predict(
            t=t,
            prior_endog_states=prior_endog_states,
            prior_exog_states=prior_exog_states,
            prior_t=prior_t,
            prediction_exog=prediction_exog,
            rng=rng
        )

        # Clip predictions at the max prediction.
        endog_pred[np.abs(endog_pred) > prediction_max] = prediction_max

        return endog_pred
    

    def get_window_size(self) -> int:
        """Returns number of previous observations model requires in order to
        make a prediction.
        
        For example, an autoregressive model with four lags needs the four
        previous timesteps in order to make a prediction, but an ODE only needs
        the current observed state.

        This function only exists to maintain flexible compatibility with the
        hyper parameter optimizer. However, at least two
        previous observations are needed in order for the hyper parameter
        optimizer to determine the timestep size. If no previous observations
        are known, simply pad with zeros and supply the appropriate times.

        Because at least two are needed, the default behavior is to require two
        previous observations.
        
        If your model needs more previous observations, overwrite this function.
        The optimizer calls this function after initialization, so the number of
        previous observations needed can depend on interal attributes:

        E.x.
            `return max(self.lags)`
        """
        return 2


    @abstractmethod
    def _fit(
        self,
        t: np.ndarray,
        endog_states: np.ndarray,
        exog_states: np.ndarray = None,
    ):
        """Fits the method using the passed data.
        
        Args:
            t (ndarray): An array of time points with shape (m,).
            endog_states (ndarray): An array of endogenous signals with shape
                (m, n). Rows are  observations and columns are variables. Each
                row corresponds to the times in `t`.
            exog_states (ndarray): An array of exogenous signals with shape  
                (m, k). Rows are observations and columns are variables. Each
                row corresponds to the times in `t`.
        """
        raise NotImplementedError()
    

    @abstractmethod
    def _predict(
        self,
        t: np.ndarray,
        prior_endog_states: np.ndarray,
        prior_exog_states: Optional[np.ndarray] = None,
        prior_t: Optional[np.ndarray] = None,
        prediction_exog: Optional[np.ndarray] = None,
        rng: np.random.RandomState = DEFAULT_RANGE,
    ) -> np.ndarray:
        """Runs a simulation of the dynamics of a fitted forcasting method.

       Note: Must call `self.fit(...)` before calling `self.predict`.

        Args:
            t (ndarray): An array of the time points with shape (m,) for the
                method to simulate.
            prior_endog_states (ndarray): Aa array of historic observations of
                the n endogeneous signals with shape (p, n). Rows represent 
                observations and columns represent variables This is used as the
                initial condition data or time lag information. It is not
                used to fit the method. IMPORTANT: It is assumed that the last
                row in this array was observed at time `t[0]`.
            prior_exog_states: An optional array of historic observations of the
                k exogenous signals with shape (p, k). Rows contain observations
                and columns contain variables. This is used for the
                initial condition data and lag information. It is NOT used to
                fit the method. IMPORTANT: It is assumed that the last
                row in this array was observed at time `t[0]`.
            prior_t (ndarray): An optional array with shape (p,) of times
                corresponding to the rows of `prior_endog_states` and `prior_exog_states`. If `prior_t` is not provided and `t`
                contains equally spaced points, then `prior_t` is assumed
                to contain occured at equally spaced points prior to `t`
            prediction_exog: An optional (m, k) array of exogenous signals
                corresponding to the times in `t`. Rows are observations and
                columns are variables.
            prediction_max: A threshold for predicted endogeneous values
                to prevent overflow in predictions. All predictions larger in
                magnitude will be set equal to `prediction_max`.
            rng: An optional numpy random state for reproducibility. (Uses 
                numpy's mtrand random number generator by default.)

        Returns:
            X_sim: A (m, n) array of containing a multivariate time series. The
            rows are observations correstponding to entries in `time_points` and
            the columns correspod to the endogenous variables in the forecasting
            method.
            
        Notes:
            When `prior_t` is provided it is assumed that the rows of
            `prior_endog_states` and `prior_exog_states` were observed at the
            times contained in `prior_t`.If `prior_t` is not provided and `t`
            contains equally spaced points, then `prior_endog_states` and
            `prior_exog_states` are assumed to have occured at equally spaced
            times before `t[0]`. Additionally, the last rows of
            `prior_endog_states` and `prior_exog_states` must be an observation
            that  occured at the time `t[0]`.
        """
        raise NotImplementedError()
    

    @abstractmethod
    def get_test_param_grid() -> Dict[str, List[Any]]:
        """Returns a dict of hyper parameters for testing grid search.
        
        Should be small and not take a long time to iterate through.
        """
        raise NotImplementedError()
    

    @abstractmethod
    def get_test_params() -> Dict[str, Any]:
        """Returns initialization parameters for testing. 
        
        Should be condusive to fast test cases."""
        raise NotImplementedError