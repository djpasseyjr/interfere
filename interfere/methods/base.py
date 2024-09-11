"""The base class for methods for intervention response prediction.
"""
from abc import abstractmethod
from typing import Any, Dict, List, Optional
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
        if len(t.shape) != 1:
            raise ValueError(
                f"The `t` arg to {str(type(self).__name__)}.fit() is not 1D.")
        
        if len(endog_states.shape) != 2:
            raise ValueError(
                f"The `endog_states` arg to {str(type(self).__name__)}.fit() is"
                " not 2D"
            )
        
        m, = t.shape
        m_endog, endog_dim = endog_states.shape
        exog_dim = None

        if m != m_endog:
            raise ValueError(
                f"The arguments `t` and `endog_states` for "
                f"{str(type(self).__name__)}.fit() have incompatible shapes: \n"
                f"\tt.shape = {t.shape}\n"
                f"\tendog_states.shape = {endog_states.shape}"
            )
        
        if exog_states is not None:
            if len(exog_states.shape) != 2:
                raise ValueError(
                    f"The `exog_states` arg to {str(type(self).__name__)}.fit() is"
                    " not 2D"
                )
        
            m_exog, exog_dim = exog_states.shape
            if m != m_exog:

                raise ValueError(
                    f"The arguments `t` and `exog_states` for "
                    f"{str(type(self).__name__)}.fit() have incompatible shapes: \n"
                    f"t.shape = {t.shape}\n"
                    f"exog_states.shape = {exog_states.shape}"
                )

        
        # Make sure no Pandas DataFrames are passed in.
        if any([
            isinstance(x, pd.DataFrame) 
            for x in [endog_states, t, exog_states]
        ]):
            raise ValueError("Interfere inference methods do not accept " "DataFrames. Use DataFrame.values and DataFrame.index")
        
        # Make sure time points are monotonic.
        if np.any(np.diff(t) <= 0):
            raise ValueError(f"Time points passed to {str(type(self).__name__)}.fit must be strictly increasing.")
        
        self.timestep_of_fit = None
        self.endog_dim_of_fit = endog_dim
        self.exog_dim_of_fit = exog_dim

        # Store timestep size if time points are evenly spaced.
        dt = t[1] - t[0]
        if np.allclose(np.diff(t), dt):
            self.timestep_of_fit = dt

        self._fit(t, endog_states, exog_states)
        self.is_fit = True
        return self
    

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
        
        if len(t.shape) != 1:
            raise ValueError(
                f"The `t` arg to {str(type(self).__name__)}.predict is not 1D.")
                
        if any([
            isinstance(x, pd.DataFrame) 
            for x in [t, prior_endog_states, prior_exog_states, prior_t, prediction_exog]
        ]):
            raise ValueError("Interfere inference methods do not accept " "DataFrames. Use DataFrame.values and DataFrame.index")
        
        # Make sure time points are monotonic.
        if np.any(np.diff(t) <= 0):
            raise ValueError(
                f"Time points passed to the {str(type(self).__name__)}.predict "
                "`t` argument must be strictly increasing."
            )
        
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
            prior_endog_states = np.reshape(prior_endog_states, (1, -1))

        # Gather array shapes
        p, k = prior_endog_states.shape
        (m,) = t.shape 

        if len(t) < 2:
            raise ValueError(
                f" For {str(type(self).__name__)}.predict, the first timestep in `t` is assumed to be the current time, and correspond to the last row "
                "of `prior_endog_states`, the `t` argument must have at least "
                "two time values."
            )
        
        if self.exog_dim_of_fit is not None and prediction_exog is None:
            raise ValueError(
                f"{type(self).__name__} was fit to exogenous data but no "
                "exogenous signals were provided to predict().")

        # Check shape of exogenous signals.
        if prediction_exog is not None:
            m_exog, k_exog = prediction_exog.shape
            if m_exog != m:
                raise ValueError(f"Number of exogenous observations ({m_exog})"
                f" does not match the number of t ({m}).")

        # Compare window size of forecaster with amount of historic data.
        w = self.get_window_size()

        if w > p:
            warn(str(type(self).__name__) + f" has window size {w} but only recieved {p}"
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
                warn(str(type(self).__name__) + f" has window size {w} but only recieved"
                     f" {p_hexog} exog observations. Augmenting historic "
                     "exogenous observations with zeros.")

                prior_exog_states = np.vstack([
                    np.zeros((w - p_hexog, k_hexog)),
                    prior_exog_states
                ])

        # Create prior_t assuming equal time step size.
        num_prior_endog, _ = prior_endog_states.shape

        if prior_t is None:
            dt = t[1] - t[0]

            # Check for equally spaced forecast time points.
            if not np.all(np.isclose(np.diff(t), dt)):
                raise ValueError("The `prior_t` argument not provided"
                " AND `t` is not equally spaced. Cannot infer "
                " `prior_t`. Either pass it explicitly or provide "
                " equally spaced time `t`.")
            prior_t = np.arange(-num_prior_endog, 1) * dt + t[0]

        if prior_t[-1] != t[0]:
            raise ValueError(
                f"For {str(type(self).__name__)}.predict, the last prior time, "
                f"`prior_t[-1]`={prior_t[-1]} must equal the first simulation "
                f"time t[0]={t[0]}."
            )

        num_prior_times, = prior_t.shape

        # If the user provided prior_t and it is greater or equal to
        # num_prior_endog, then we trim it to size.
        if num_prior_times > num_prior_endog:
            warn(f"{str(type(self).__name__)}.predict was passed too many"
                 f"({num_prior_times}) prior time points. Using only the last "
                 f"{num_prior_endog} time points.")
            prior_t = prior_t[-num_prior_endog:]

        # If the user did not pass enough prior_t, check for equally spaced time
        # points and infer.
        if num_prior_times < num_prior_endog:

            # Check if equally spaced.
            warn(
                "Inferring additional `prior_t` values. Assuming `prior_t` has"
                " the same equally spaced timestep size as `t`"
            )
            dt = t[1] - t[0]
            if not np.all(np.isclose(np.diff(prior_t), dt)):

                # When prior_endog_states were not augmented with zeros raise a
                # normal value error. 
                if num_prior_endog == p:
                    raise ValueError(
                        f"{str(type(self).__name__)}.predict was passed {p} "
                        "prior_endog_states but there are only "
                        f"{num_prior_times} entries in `prior_t`."
                    )
                
                # When prior_endog_states were augmented with zeros, give
                # instructions on how to augment.
                if num_prior_endog > p:
                    raise ValueError(
                        f"{str(type(self).__name__)}.predict augmented "
                        "`prior_endog_states` with zeros but `prior_t` was not "
                        "equally spaced so it was not possible to infer "
                        "additional prior times. \n\nTo solve, pass at least "
                        f"({num_prior_endog}) previous time values or use "
                        "equally spaced `prior_t`."
                    )
            extra_prior_t = np.arange(
                num_prior_times - num_prior_endog, 0) * dt + prior_t[0]
            prior_t = np.hstack([extra_prior_t, prior_t])

        if np.any(np.diff(prior_t) <= 0):
            raise ValueError(
                f"Prior time points passed to {str(type(self).__name__)}."
                "predict must be strictly increasing."
            )        

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