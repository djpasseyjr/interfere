from typing import Optional, Union

import numpy as np

from .base import StochasticDifferentialEquation
from ..utils import copy_doc


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


    @copy_doc(StochasticDifferentialEquation.drift)
    def drift(self, x: np.ndarray, t: float):
        x, y, z = x[0], x[1], x[2]
        dxdt = -2 * x + 7 * y ** 2 + 13 * z ** 2
        dydt = self.mu * x + 7 * y + 10 * z - 3 * x * y
        dzdt = -10 * y + 7 * z -3 * x * z
        return np.array([dxdt, dydt, dzdt])
    

    @copy_doc(StochasticDifferentialEquation.noise)
    def noise(self, x: np.ndarray, t: float):
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


    @copy_doc(StochasticDifferentialEquation.drift)
    def drift(self, x: np.ndarray, t: float):
        y1, y2, y3 = x[0], x[1], x[2]
        dy1dt = y3 + (y2 - 0.3) * y1
        dy2dt = 2 - 0.1 * y2 - y1 ** 2
        dy3dt = y1 * y2 - y1 - 0.1 * y3
        return np.array([dy1dt, dy2dt, dy3dt])
    

    @copy_doc(StochasticDifferentialEquation.noise)
    def noise(self, x: np.ndarray, t: float):
        # Returns an array to rescale the brownian noise:        
        return self.sigma * np.diag(x)
    

class Lorenz(StochasticDifferentialEquation):


    def __init__(
        self,
        s: float = 10,
        beta: float = 8/3,
        rho: float = 28,
        sigma: Union[float,  np.ndarray] = 0.0,
        measurement_noise_std: Optional[np.ndarray] = None
    ):
        """Initializes a 3D quadratic SDE.

        dx/dt = s * (y - x)

        dy/dy = x * (rho - z) - y

        dz/dt = x * y - beta * z

        This deterministic system is made stochastic by defining

        dx = [dx/dt, dy/dt, dz/dt]^T
        dW = [dw1, dw2, dw3]^T

        so that
        dX = dx * dt + sigma * dW

        where dW is an array of weiner incremements and sigma is a 3x3 matrix.

        
        Args:
            s (float): A parameter of the model.
            beta (float): A parameter of the model.
            rho (float): A parameter of the model.
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


        See: Lorenz, E. (1963) Deterministic Nonperiodic Flow.
        """
        dim = 3
        self.s = s
        self.beta = beta
        self.rho = rho
        super().__init__(dim, measurement_noise_std, sigma)


    @copy_doc(StochasticDifferentialEquation.drift)
    def drift(self, x: np.ndarray, t: float):
        x, y, z = x[0], x[1], x[2]
        dxdt = self.s * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        return np.array([dxdt, dydt, dzdt])
    

    @copy_doc(StochasticDifferentialEquation.noise)
    def noise(self, x: np.ndarray, t: float):
        return self.sigma
    

class Rossler(StochasticDifferentialEquation):


    def __init__(
        self,
        a: float = 0.2,
        b: float = 0.2,
        c: float = 5.7,
        sigma: Union[float, np.ndarray] = 0.0,
        measurement_noise_std: Optional[np.ndarray] = None
    ):
        """Initializes the Rossler system.

        dx/dt = - y - z

        dy/dy = x + a * y

        dz/dt = b + z * (x - c)

        This deterministic system is made stochastic by defining

        dx = [dx/dt, dy/dt, dz/dt]^T
        dW = [dw1, dw2, dw3]^T

        so that
        dX = dx * dt + sigma * dW

        where dW is an array of weiner incremements and sigma is a 3x3 matrix.

        Args:
            a (float): A parameter of the model.
            b (float): A parameter of the model.
            c (float): A parameter of the model.
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

        See: Rössler, O. (1976). Chaotic behavior in simple reaction system.
        """
        self.a = a
        self.b = b
        self.c = c

        super().__init__(3, measurement_noise_std, sigma)


    @copy_doc(StochasticDifferentialEquation.drift)
    def drift(self, x: np.ndarray, t: float):
        x, y, z = x[0], x[1], x[2]
        dxdt = - y - z
        dydt = x + self.a * y
        dzdt = self.b + z * (x - self.c)
        return np.array([dxdt, dydt, dzdt])
    

    @copy_doc(StochasticDifferentialEquation.noise)
    def noise(self, x: np.ndarray, t: float):
        return self.sigma
    

class Thomas(StochasticDifferentialEquation):


    def __init__(
        self,
        b: float = 0.208186,
        sigma: Union[float, np.ndarray] = 0.0,
        measurement_noise_std: Optional[np.ndarray] = None
    ):
        """Initializes Thomas's cyclically symmetric attractor system.

        dx/dt = sin(y) - b * x

        dy/dy = sin(z) - b * y

        dz/dt = sin(x) - b * z

        This deterministic system is made stochastic by defining

        dx = [dx/dt, dy/dt, dz/dt]^T
        dW = [dw1, dw2, dw3]^T

        so that
        dX = dx * dt + sigma * dW

        where dW is an array of weiner incremements and sigma is a 3x3 matrix.

        Args:
            b (float): A parameter of the model corresponding to the amount of
                dissipation in the system. Chaotic for b <= 0.208186.
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

        See: Thomas, R. (1999). Deterministic chaos seen in terms of feedback circuits: Analysis, synthesis, 'labyrinth chaos'
        """
        self.b = b

        super().__init__(3, measurement_noise_std, sigma)


    @copy_doc(StochasticDifferentialEquation.drift)
    def drift(self, x: np.ndarray, t: float):
        x, y, z = x[0], x[1], x[2]
        dxdt = np.sin(y) - self.b * x
        dydt = np.sin(z) - self.b * y
        dzdt = np.sin(x) - self.b * z
        return np.array([dxdt, dydt, dzdt])
    

    @copy_doc(StochasticDifferentialEquation.noise)
    def noise(self, x: np.ndarray, t: float):
        return self.sigma


class MooreSpiegel(StochasticDifferentialEquation):


    def __init__(
        self,
        R: float = 31,
        T: float = 5,
        sigma: Union[float, np.ndarray] = 0.0,
        measurement_noise_std: Optional[np.ndarray] = None
    ):
        """Initializes Moore-Spiegel system.

            dx/dt =y
            dy/dt = z
            dz/dt = -z - (T - R + R x^2)y - Tx

        The deterministic system is made stochastic by defining

        dx = [dx/dt, dy/dt, dz/dt]^T
        dW = [dw1, dw2, dw3]^T

        so that
        dX = dx * dt + sigma * dW

        The Moore–Spiegel system is a nonlinear thermo-mechanical oscillator 
        with displacement x(t). The system describes a fluid element 
        oscillating vertically in a temperature gradient with a linear 
        restoring force.


        Args:
            R (float): A parameter of the model.  Analogous to the Prandtl 
                number times the Rayleigh number
            T (float): A parameter of the model. Analogous to the Prandtl number
                times the Taylor number.
            sigma (float or ndarray): The stochastic noise parameter. Can be a
                float, a 1D matrix or a 2D matrix. Dimension must match
                dimension of model.
            measurement_noise_std (ndarray): None, or a vector with shape (n,)
                where each entry corresponds to the standard deviation of the
                measurement noise for that particular dimension of the dynamic
                model. For example, if the dynamic model had two variables x1
                and x2 and `measurement_noise_std = [1, 10]`, then
                independent gaussian noise with standard deviation 1 and 1
                will be added to x1 and x2 respectively at each point in time. 
        """
        self.R = R
        self.T = T

        super().__init__(3, measurement_noise_std, sigma)
        

    def drift(self, x: np.ndarray, t: float):
        x, y, z = x[0], x[1], x[2]
        dxdt = y
        dydt = z
        dzdt = - z - (self.T - self.R + self.R * x**2) * y - self.T * x
        return np.array([dxdt, dydt, dzdt])


    def noise(self, x: np.ndarray, t: float):
        return self.sigma