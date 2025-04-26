"""Dynamic model wrapper for predictive algorithms. 
"""

from typing import Callable, Optional, Union

import numpy as np
import pysindy as ps

from .coupled_map_lattice import coupled_logistic_map
from .quadratic_sdes import Lorenz
from ..base import DynamicModel, DEFAULT_RANGE
from .._methods.vector_autoregression import VAR
from .._methods.sindy import SINDy
from ..base import ForecastMethod
from ..utils import copy_doc


class GenerativeForecaster(DynamicModel):


    def __init__(
        self,
        fitted_method: ForecastMethod,
        sigma: Union[float, np.ndarray] = None,
        measurement_noise_std: np.ndarray = None,
    ):
        """Initializes the GenerativeForecaster.

        Args:
            fitted_method (ForecastMethod): A predictive method which has
                already been fitted to data.
            sigma (float or ndarray): The stochastic noise parameter. Can be a
                float, a 1D matrix or a 2D matrix. Dimension must match
                dimension of model.
            measurement_noise_std (ndarray): None, or a vector with shape (n,)
                where each entry corresponds to the standard deviation of the
                measurement noise for that particular dimension of the dynamic
                model. For example, if the dynamic model had two variables x1
                and x2 and `measurement_noise_std = [1, 10]`, then
                independent gaussian noise with standard deviation 1 and 10
                will be added to x1 and x2 respectively at each point in time.
        """

        self.fitted_method = fitted_method
        if not self.fitted_method.is_fit:
            raise ValueError(
                f"{type(self).__name__}.__init__ requires a fitted method.")
        
        self.timestep = self.fitted_method.timestep_of_fit

        if self.timestep is None:
            raise ValueError(
                f"{type(self).__name__} requires a method that was"
                " fit to evenly spaced time points."
            )

        if self.fitted_method.exog_dim_of_fit is not None:
            ValueError(
                "GenerativeForecaster cannot simulate methods that were fit to "
                "exogenous data."
            )

        dim = self.fitted_method.endog_dim_of_fit

        super().__init__(dim, measurement_noise_std, sigma)


    @copy_doc(DynamicModel.simulate)
    def _simulate(
        self,
        t: np.ndarray,
        prior_states: np.ndarray,
        prior_t: Optional[np.ndarray] = None,
        intervention: Optional[Callable[[np.ndarray, float], np.ndarray]]= None,
        rng: np.random.mtrand.RandomState = DEFAULT_RANGE,
        **kwargs
    ) -> np.ndarray:
        
        num_prior_obs, dim = prior_states.shape
        
        if self.fitted_method.endog_dim_of_fit != dim:
            raise ValueError(
                f"{type(self).__name__}.simulate() was passed prior states with"
                f" shape {prior_states.shape} but the internal forecaster was "
                f"expecting an array with {self.fitted_method.endog_dim_of_fit}"
                " columns."
            )

        for i in range(len(t) - 1):

            new_states = self.fitted_method.predict(
                t[i:(i+2)],
                prior_states,
                prior_t=prior_t,
                prediction_max=np.inf
            )
            next_state = new_states[-1,:]

            # Optionally intervene in model.
            if intervention is not None:
                next_state = intervention(next_state, t[i+1])

            # Add stochastic noise.
            next_state += self.sigma @ rng.normal(size=self.dim)

            prior_states = np.vstack([prior_states, next_state])
            prior_t = np.hstack([prior_t, [t[i+1]]])

        sim_states =  prior_states[-len(t):]

        # Add measurement noise to all but initial condition.
        sim_states[1:, :] = self.add_measurement_noise(
            sim_states[1:, :], rng=rng)
        
        return sim_states
        


def generative_lorenz_VAR_forecaster(
    sigma: Union[float, np.ndarray] = None,
    measurement_noise_std: np.ndarray = None,
    tsteps = 300,
    dt = 0.02,
):
    """Initializes the a GenerativeForecaster with a VAR model that has been
    fit to the Lorenz equations.

        Args:
            sigma (float or ndarray): The stochastic noise parameter. Can be a
                float, a 1D matrix or a 2D matrix. Dimension must match
                dimension of model.
            measurement_noise_std (ndarray): None, or a vector with shape (n,)
                where each entry corresponds to the standard deviation of the
                measurement noise for that particular dimension of the dynamic
                model. For example, if the dynamic model had two variables x1
                and x2 and `measurement_noise_std = [1, 10]`, then
                independent gaussian noise with standard deviation 1 and 10
                will be added to x1 and x2 respectively at each point in time.
            tsteps (int): The number of time steps to simulate.
            dt (float): The timestep size 

    """
    train_t = np.arange(0, tsteps * dt, dt)
    train_prior_states = np.array([0.1, 0.1, 0.1])
    train_states = Lorenz().simulate(
        train_t, train_prior_states)

    method = VAR(maxlags=5)
    method.fit(train_t, train_states)

    return GenerativeForecaster(method,
        sigma=sigma, measurement_noise_std=measurement_noise_std)


def generative_cml_SINDy_forecaster(
    sigma: Optional[Union[float, np.ndarray]] = None,
    measurement_noise_std: Optional[np.ndarray] = None,
    tsteps: int = 300,
):
    """Creates a generative SINDy model that is fit to a coupled map lattice.

    Args:
        sigma (float or ndarray): The stochastic noise parameter. Can be a
            float, a 1D matrix or a 2D matrix. Dimension must match
            dimension of model.
        measurement_noise_std (ndarray): None, or a vector with shape (n,)
            where each entry corresponds to the standard deviation of the
            measurement noise for that particular dimension of the dynamic
            model. For example, if the dynamic model had two variables x1
            and x2 and `measurement_noise_std = [1, 10]`, then
            independent gaussian noise with standard deviation 1 and 10
            will be added to x1 and x2 respectively at each point in time.
        tsteps (int): The number of time steps to simulate.
    """
    train_t = np.arange(0, tsteps)
    dim = 10
    train_prior_states = -0.1 * np.ones(dim)
    rng=np.random.default_rng(11)
    
    train_states = coupled_logistic_map(
        **{
            # Two cycles and isolated node
            "adjacency_matrix": np.array([
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]),
            "eps": 0.9,
            "logistic_param": 3.72,
            "sigma": 0.0,
            "measurement_noise_std": 0.01 * np.ones(10),
    }).simulate(train_t, train_prior_states, rng=rng)

    method = SINDy(**{
        'optimizer__threshold': 2.619572195011037,
        'optimizer__alpha': 0.0001587441595198887,
        'discrete_time': True,
        'feature_library': ps.PolynomialLibrary(),
        'differentiation_method': ps.SINDyDerivative(
            alpha=0.01, kind='trend_filtered', order=1)
    })
    method.fit(train_t, train_states)

    return GenerativeForecaster(method,
        sigma=sigma, measurement_noise_std=measurement_noise_std)
