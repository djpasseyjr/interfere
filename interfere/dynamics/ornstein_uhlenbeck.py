from typing import Optional

import numpy as np

from .base import StochasticDifferentialEquation


class OrnsteinUhlenbeck(StochasticDifferentialEquation):

    def __init__(
        self,
        theta: Optional[np.ndarray] = None,
        mu: Optional[np.ndarray] = None,
        sigma: Optional[np.ndarray] = None,
        measurement_noise_std: Optional[np.ndarray] = None
    ):
        """Initializes n-dimensional Ornstein Uhlenbeck process.

        dX = theta(mu - X)dt + sigma dW

        Args:
            theta (ndarray): A (n, n) matrix.
            mu (ndarray): A (n,) vector.
            sigma (ndarray): A (n, n) matrix.
            measurement_noise_std (ndarray): None, or a vector with shape (n,)
                where each entry corresponds to the standard deviation of the
                measurement noise for that particular dimension of the dynamic
                model. For example, if the dynamic model had two variables x1
                and x2 and `measurement_noise_std = [1, 10]`, then
                independent gaussian noise with standard deviation 1 and 10
                will be added to x1 and x2 respectively at each point in time.  
        """
        # Set dimension
        super().__init__(len(mu), measurement_noise_std, sigma)

        # Input validation
        if any([
            mu.shape[0] != theta.shape[0],
            theta.shape[0] != theta.shape[1],
            theta.shape[0] != self.sigma.shape[0],
            self.sigma.shape[1] != mu.shape[0]
        ]):
            raise ValueError(
                "Parameters for OrnsteinUhlenback must have matching "
                "dimensions. Argument shapes: "
                f"\n\ttheta.shape = {theta.shape}"
                f"\n\tmu.shape = {mu.shape}"
                f"\n\tsigma.shape = {sigma.shape}"
            )

        # Assign class attributes
        self.theta = theta
        self.mu = mu

    def drift(self, x: np.ndarray, t: float):
        return self.theta @ (self.mu - x)
    
    def noise(self, x: np.ndarray, t: float):
        return self.sigma