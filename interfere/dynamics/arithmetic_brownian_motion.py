from typing import Optional

import numpy as np

from .base import StochasticDifferentialEquation


class ArithmeticBrownianMotion(StochasticDifferentialEquation):

    def __init__(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        measurement_noise_std: Optional[np.ndarray] = None
    ):
        """Initializes n-dimensional arithmetic brownian motion process.

        dX_i =  mu_i dt + sigma_i dW_i

        Args:
            mu (ndarray): A (n,) vector.
            sigma (ndarray): A (n,) vector.
            measurement_noise_std (ndarray): None, or a vector with shape (n,)
                where each entry corresponds to the standard deviation of the
                measurement noise for that particular dimension of the dynamic
                model. For example, ifd4 the dynamic model had two variables x1
                and x2 and `measurement_noise_std = [1, 10]`, then
                independent gaussian noise with standard deviation 1 and 10
                will be added to x1 and x2 respectively at each point in time.  
        """
        # Set dimension and stochasticity
        super().__init__(len(mu), measurement_noise_std, sigma)

        # Input validation
        if mu.shape[0] != self.sigma.shape[0]:
            raise ValueError(
                "Parameters for Arithmetic Brownian motion must have matching dimensions. "
                "Argument shapes: "
                f"\n\tmu = {mu.shape}"
                f"\n\tsigma = {self.sigma.shape}"
            )

        # Assign class attributes
        self.mu = mu

    def drift(self, x: np.ndarray, t: float):
        return self.mu
    
    def noise(self, x: np.ndarray, t: float):
        return np.diag(self.sigma)