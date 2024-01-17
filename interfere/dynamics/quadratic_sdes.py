from typing import Optional

import numpy as np

from .base import StochasticDifferentialEquation


class Belozyorov3DQuad(StochasticDifferentialEquation):

    def __init__(
        self,
        mu: float = 1.81,
        sigma: float = 0.0,
        measurement_noise_std: Optional[np.ndarray] = None
    ):
        """Initializes a 3D quadratic SDE.

        dx/dt = -2x + 7y^2 + 13z^2 

        dy/dy = mu*x + 7y + 10z - 3xy

        dz/dt = -10y + 7z -3xz

        
        dX = dx/dt(x, y, z) * dt + sigma dW

        dY = dy/dt(x, y, z) * dt + sigma dW
        
        dZ = dz/dt(x, y, z) * dt + sigma dW

        
        Taken from 
            Belozyorov (2015). Exponential-Algebraic Maps and Chaos in 3D
            Autonomous Quadratic Systems. Equation (37).

        Args:
            mu (float): A parameter that can be tuned to generate chaotic
                behavior. Should be in [0, 2.27) and mu = 1.81 should lead
                to chaotic dynamics.
            sigma (float):  Coefficient on Weiner intervals.
            measurement_noise_std (ndarray): None, or a vector with shape (n,)
                where each entry corresponds to the standard deviation of the
                measurement noise for that particular dimension of the dynamic
                model. For example, if the dynamic model had two variables x1
                and x2 and `measurement_noise_std = [1, 10]`, then
                independent gaussian noise with standard deviation 1 and 10
                will be added to x1 and x2 respectively at each point in time.     
        """
        dim = 3
        super().__init__(dim, measurement_noise_std)
        self.mu = mu
        self.sigma = sigma
        self.Sigma = sigma * np.eye(dim)

    def drift(self, X: np.ndarray, t: float):
        x, y, z = X[0], X[1], X[2]
        dxdt = -2 * x + 7 * y ** 2 + 13 * z ** 2
        dydt = self.mu * x + 7 * y + 10 * z - 3 * x * y
        dzdt = -10 * y + 7 * z -3 * x * z
        return np.array([dxdt, dydt, dzdt])
    
    def noise(self, X: np.ndarray, t: float):
        return self.Sigma


class Liping3DQuadFinance(StochasticDifferentialEquation):

    def __init__(
        self,
        sigma: float = 0.0,
        measurement_noise_std: Optional[np.ndarray] = None
    ):
        """Initializes a 3D quadradic chaotic system.

        dy1/dt = y3 + (y2 - 0.3)y1
        dy2/dt = 2 - 0.1y2 - y1^2
        dy3/dt = y1y2 - y1 - 0.1y3

        dY1 = dy1/dt(y) * dt + sigma * y1 * dW
        dY2 = dy1/dt(y) * dt + sigma * y2 * dW
        dY3 = dy1/dt(y) * dt + sigma * y3 * dW

        Taken from :
            Liping (2021). A new financial chaotic model in Atangana-Baleanu
            stochastic fractional differential equations.
        
        A fractional derivative is not used here--the d parameter in the paper
        is assumed to be 1 so the fractional derivative reduces to a normal
        derivative.

        Args:
            sigma (float): Coefficient on the independent Weiner increments.
            measurement_noise_std (ndarray): None, or a vector with shape (n,)
                where each entry corresponds to the standard deviation of the
                measurement noise for that particular dimension of the dynamic
                model. For example, if the dynamic model had two variables x1
                and x2 and `measurement_noise_std = [1, 10]`, then
                independent gaussian noise with standard deviation 1 and 10
                will be added to x1 and x2 respectively at each point in time. 
        """
        dim = 3
        super().__init__(dim, measurement_noise_std)
        self.sigma = sigma

    def drift(self, Y: np.ndarray, t: float):
        y1, y2, y3 = Y[0], Y[1], Y[2]
        dy1dt = y3 + (y2 - 0.3) * y1
        dy2dt = 2 - 0.1 * y2 - y1 ** 2
        dy3dt = y1 * y2 - y1 - 0.1 * y3
        return np.array([dy1dt, dy2dt, dy3dt])
    
    def noise(self, Y: np.ndarray, t: float):
        # Returns an array to rescale the brownian noise:        
        return self.sigma * np.diag(Y)