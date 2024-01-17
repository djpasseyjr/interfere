from typing import Optional

import numpy as np

from .base import StochasticDifferentialEquation


class GeometricBrownianMotion(StochasticDifferentialEquation):

    def __init__(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        measurement_noise_std: Optional[np.ndarray] = None
    ):
        """Initializes decoupled n-dimensional geometric brownian motion.

        dX_i =  mu_i X_i dt + sigma_i X_i dW_i

        Args:
            mu (ndarray): A (n,) vector.
            sigma (ndarray): A (n,) matrix.
            measurement_noise_std (ndarray): None, or a vector with shape (n,)
                where each entry corresponds to the standard deviation of the
                measurement noise for that particular dimension of the dynamic
                model. For example, ifd4 the dynamic model had two variables x1
                and x2 and `measurement_noise_std = [1, 10]`, then
                independent gaussian noise with standard deviation 1 and 10
                will be added to x1 and x2 respectively at each point in time.  
        """
        # Input validation
        if mu.shape[0] != sigma.shape[0]:
            raise ValueError(
                "Parameters for Arithmetic Brownian motion must have matching dimensions. "
                "Argument shapes: "
                f"\n\tmu = {mu.shape}"
                f"\n\tsigma = {sigma.shape}"
            )
        # Set dimension
        super().__init__(len(mu), measurement_noise_std)
        # Assign class attributes
        self.mu = mu
        self.sigma = sigma

    def drift(self, x: np.ndarray, t: float):
        return self.mu * x
    
    def noise(self, x: np.ndarray, t: float):
        return np.diag(self.sigma) * x