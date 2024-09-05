"""Dynamic model wrapper for predictive algorithms. 
"""

from typing import Callable, Optional

import numpy as np

from ..base import DynamicModel, DEFAULT_RANGE
from ..methods.base import BaseInferenceMethod
from ..utils import copy_doc


class GenerativeForecaster(DynamicModel):


    def __init__(
        self,
        fitted_method: BaseInferenceMethod,
        sigma: np.ndarray,
        measurement_noise_std: np.ndarray
    ):
        """Initializes the GenerativeForecaster.

        Args:
            fitted_method (BaseInferenceMethod): A predictive method which has
                already been fitted to data.
        """
        # TODO: Add checks to make sure dimension of the model matches the
        # dimension of the method.

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

        super().__init__(dim, measurement_noise_std)


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
        

        for i in range(len(t) - 1):
            new_states = self.fitted_method.predict(
                t[i:(i+2)],
                prior_states,
                prior_t=prior_t,
            )
            next_state = new_states[-1,:]

            # Optionally intervene in model.
            if intervention is not None:
                next_state = intervention(next_state, t[i+1])

            prior_states = np.vstack([prior_states, next_state])
            prior_t = np.hstack([prior_t, [t[i+1]]])

        return prior_states[-len(t):]



        
