"""The base class for methods for intervention response prediction.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

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
        simulation_times: np.ndarray,
        historic_states: np.ndarray,
        intervention: Optional[ExogIntervention] = None,
        historic_times: Optional[np.ndarray] = None,
        rng: np.random.mtrand.RandomState = DEFAULT_RANGE
    ) -> np.ndarray:
        """Simulates a the intervention response with a fitted method.

        Args:
            simulation_times: A (m,) array of times to simulate.
            historic_states: A (p, n + k) array of endogenous and exogenous
                signals. Rows are observations corresponding to
                `simulation_times` and columns are variables. The k exogenous
                variable indexes are contained in `intervention.intervened_idxs`
            intervention: An interfere.ExogIntervention.
            historic_times: Optional (p,) array of times corresponding to
                historic observations. Defaults to equally spaced points
                immediately prior to `simulation_times`. Assumes that the last
                row of `historic_states` corresponds to the initial condition,
                the state of the system at time `t = simulation_times[0]`.
            rng: Numpy random state.

        Returns:
            simulated_states: A (m, n + k) array of simulated exogenous and
                endogenous signals. The k exogenous variable indexes are
                contained in `intervention.intervened_idxs`.
        """
        if intervention is not None:
            historic_endo, historic_exog = intervention.split_exogeneous(
                historic_states)
            exog = intervention.eval_at_times(simulation_times)

        else:
            historic_endo = historic_states
            historic_exog = None
            exog = None

        endo_pred = self.predict(
            simulation_times,
            historic_endo,
            exog,
            historic_exog,
            historic_times,
            rng
        )

        simulated_states = endo_pred

        # Optionally intervention to combine exogeneous and endogenous.
        if intervention is not None:
            simulated_states = intervention.combine_exogeneous(endo_pred, exog)
    
        return simulated_states


# TODO Change order of t and endog. Consider new name for t.
    def fit(
        self,
        endog_states: np.ndarray,
        t: np.ndarray,
        exog_states: np.ndarray = None,
    ):
        """Fits the method using the passed data.
        
        Args:
            endog_states: An (m, n) array of endogenous signals. Sometimes
                called Y. Rows are observations and columns are variables. Each
                row corresponds to the times in `t`.
            t: An (m,) array of time points.
            exog_states: An (m, k) array of exogenous signals. Sometimes called
                X. Rows are observations and columns are variables. Each row 
                corresponds to the times in `t`.
        """
        # Make sure no Pandas DataFrames are passed in.
        if any([
            isinstance(x, pd.DataFrame) 
            for x in [endog_states, t, exog_states]
        ]):
            raise ValueError("Interfere inference methods do not accept " "DataFrames. Use DataFrame.values and DataFrame.index")
        

        self.is_fit = True
        return self._fit(endog_states, t, exog_states)
    

    def predict(
        self,
        forecast_times: np.ndarray,
        historic_endog: np.ndarray,
        exog: Optional[np.ndarray] = None,
        historic_exog: Optional[np.ndarray] = None,
        historic_times: Optional[np.ndarray] = None,
        rng: np.random.RandomState = DEFAULT_RANGE,
        prediction_max: float = 1e9,
    ) -> np.ndarray:
        """Runs a simulation of the dynamics of a fitted forcasting method.

       Note: Must call `self.fit(...)` before calling `self.predict`.

        Args:
            forecast_times: A (m,) array of the time points for the method to
                simulate.
            historic_endog: A (p, n) array of historic observations of the
                ENDOGENOUS signals. This is used as the initial condition data
                and lagged initial conditions. It is NOT used to fit the method.
                If `historic_times` is not provided and `forecast_times`
                contains equally spaced points, the observations are assumed to
                have occured at equally spaced points prior to `forecast times`.
                Additionally, the last row of `historic_states` is assumed to
                correspond with the first entry in `forecast_times`. Otherwise,
                the rows of this matrix must must correspond to times in
                `historic_times`.
            exog: An optional (m, k) array of exogenous signals corresponding to
                the times in `forecast_times`.
            historic_exog: An optional (p, k) array of historic obs of the
                EXOGENOUS signals.  This is used as the initial condition data
                and lag information. It is not used to fit the method. If 
                `historic_times` is not provided and `forecast_times` contains
                equally spaced points, the observations are assumed to have
                occured at equally spaced points prior to `forecast times`.
                Otherwise, the rows of this matrix must must correspond to
                times in `historic_times`.
            historic_times: An optional (p,) array of times corresponding to 
                the rows of `historic_endog` and `historic_exog`. If 
                `historic_times` is not provided and `forecast_times`
                contains equally spaced points, then `historic_times` is assumed
                to contain occured at equally spaced points prior to `forecast
                times`.
            prediction_max: A threshold for predicted endogeneous values
                to prevent overflow in predictions. All predictions larger in
                magnitude will be set equal to `prediction_max`.
            rng: An optional numpy random state for reproducibility. (Uses 
                numpy's mtrand random number generator by default.)

        Returns:
            X_sim: A (m, n) array of containing a multivariate time series. The
            rows are observations correstponding to entries in `time_points` and
            the columns correspod to the endogenous cariables in the forecasting method.
        """
        if not self.is_fit:
            raise ValueError("Call self.fit(...) before self.predict(...).")
        
        if any([
            isinstance(x, pd.DataFrame) 
            for x in [forecast_times, historic_endog, exog, historic_exog]
        ]):
            raise ValueError("Interfere inference methods do not accept " "DataFrames. Use DataFrame.values and DataFrame.index")
        
        if np.any(historic_endog > prediction_max):
            raise ValueError(
                f"Historic endogenous contains values ({np.max(historic_endog)}"
                " that are larger than" 
                f" `prediction_max = {prediction_max}`, the "
                "prediction threshold."
                "Increase `prediction_max` in order to simulate these values."
            )
        
        # Reshape historic_endog if it was 1D.
        if len(historic_endog.shape) == 1:
            historic_endog = np.reshape(historic_endog, 1, -1)

        # Gather array shapes
        p, _ = historic_endog.shape
        (m,) = forecast_times.shape 

        if len(forecast_times) < 2:
            raise ValueError("Since the first timestep is assumed to be the " "current time, and correspond to the last row of `historic_endog`," " the `forecast_times` must have at least two time values.")
        
        # Create historic_times assuming equal time step size.
        if historic_times is None:
            dt = forecast_times[1] - forecast_times[0]

            # Check for equally spaced forecast time points.
            if not np.all(np.isclose(np.diff(forecast_times), dt)):
                raise ValueError("The `historic_times` argument not provided"
                " AND `forecast_times` are equally spaced. Cannot infer "
                " `historic_times`. Either pass it explicitly or provide "
                " equally spaced time `forecast_times`.")
            
            historic_times = np.arange(-p, 1) * dt + forecast_times[0]


        # Check shape of exogenous signals.
        if exog is not None:
            m_exog, k_exog = exog.shape
            if m_exog != m:
                raise ValueError(f"Number of exogenous observations ({m_exog})"
                f" does not match the number of forecast_times ({m}).")


        # Check shape of historic exogenous signals.
        if historic_exog is not None:
            p_hexog, k_hexog = historic_exog.shape

            if p_hexog != p:
                raise ValueError("Arguments `historic_endog` and "
                "`historic_exog` must have the same number of rows.")
            
            if exog is not None:
                if k_hexog != k_exog:
                    raise ValueError("The `historic_exog` and `exog` arguments"
                    " must have the same number of columns.")


        endog_pred = self._predict(
            forecast_times=forecast_times,
            historic_endog=historic_endog,
            exog=exog,
            historic_exog=historic_exog,
            historic_times=historic_times,
            rng=rng
        )

        endog_pred[np.abs(endog_pred) > prediction_max] = prediction_max

        return endog_pred
    

    def get_window_size(self):
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
        endog_states: np.ndarray,
        t: np.ndarray,
        exog_states: Optional[np.ndarray] = None,
    ):
        """Fits the method using the passed data.
        
        Args:
            endog_states: An (m, n) array of endogenous signals. Sometimes
                called Y. Rows are observations and columns are variables. Each
                row corresponds to the times in `t`.
            t: An (m,) array of time points.
            exog_states: An (m, k) array of exogenous signals. Sometimes called
                X. Rows are observations and columns are variables. Each row 
                corresponds to the times in `t`.
        """
        raise NotImplementedError()
    

    @abstractmethod
    def _predict(
        self,
        forecast_times: np.ndarray,
        historic_endog: np.ndarray,
        historic_times: np.ndarray,
        exog: Optional[np.ndarray] = None,
        historic_exog: Optional[np.ndarray] = None,
        rng: np.random.RandomState = DEFAULT_RANGE,
    ) -> np.ndarray:
        """Runs a simulation of the dynamics of a fitted forcasting method.

       Note: Must call `self.fit(...)` before calling `self.predict`.

        Args:
            forecast_times: A (m,) array of the time points for the method to
                simulate.
            historic_endog: A (p, n) array of historic observations of the
                ENDOGENOUS signals. This is used as the initial condition data
                and lagged initial conditions. It is NOT used to fit the method.
            exog: An optional (m, k) array of exogenous signals corresponding to
                the times in `forecast_times`.
            historic_exog: An optional (p, k) array of historic obs of the
                EXOGENOUS signals.  This is used as the initial condition data
                and lag information. It is not used to fit the method. If 
                `historic_times` is not provided and `forecast_times` contains
                equally spaced points, the observations are assumed to have
                occured at equally spaced points prior to `forecast times`.
                Otherwise, the rows of this matrix must must correspond to
                times in `historic_times`.
            historic_times: An optional (p,) array of times corresponding to 
                the rows of `historic_endog` and `historic_exog`.
            rng: An optional numpy random state for reproducibility. (Uses 
                numpy's mtrand random number generator by default.)

        Returns:
            endog_pred: A (m, n) array containing a multivariate time series.
            The rows are observations and the columns are the
            endogenous variables. Each row corresponds directly with the times
            contained in `forecast_times`.
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