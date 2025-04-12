from typing import Optional

from .base import OrdinaryDifferentialEquation, StochasticDifferentialEquation

import numpy as np

class LotkaVoltera(OrdinaryDifferentialEquation):
    """Class for simulating Lotka Voltera dynamics.

    Can be simulated using the parent class `simulate` method.
    """

    def __init__(
        self,
        growth_rates: np.ndarray,
        capacities: np.ndarray,
        interaction_mat: np.ndarray,
        measurement_noise_std: Optional[np.ndarray] = None,
        sigma = None,
    ):
        """Initializes class for simulating Lotka Voltera dynamics.

            dx_i/dt = r_i * x_i * (1 - x_i / k_i +  [A x]_i / k_i)
        
        where r_i and k_i are the growth rates and carrying capacities of
        species i, A is the matrix of interspecies interactions.

        See:
        * Vadillo 2019, "Comparing stochastic Lotka–Volterra predator-prey
        models"
        * https://en.wikipedia.org/wiki/Competitive_Lotka%E2%80%93Volterra_equations        
        * https://github.com/netsiphd/netrd/blob/master/netrd/dynamics/lotka_volterra.py

        Args:
            growth_rates (ndarray): A length n vector of growth rates (r_i's).
            capacities (ndarray): A length n vector of carrying capacities 
                (k_i's).
            interaction_mat: A weighted (n, n) matrix of interspecies
                interactions. (A in the above equation.)
            measurement_noise_std (ndarray): None, or a vector with shape (n,)
                where each entry corresponds to the standard deviation of the
                measurement noise for that particular dimension of the dynamic
                model. For example, if the dynamic model had two variables x1
                and x2 and `measurement_noise_std = [1, 10]`, then
                independent gaussian noise with standard deviation 1 and 10
                will be added to x1 and x2 respectively at each point in time.  
        """
        # Input validation
        if any([
            growth_rates.shape != capacities.shape,
            interaction_mat.shape[0] != interaction_mat.shape[1],
            interaction_mat.shape[1] != capacities.shape[0],
        ]):
            raise ValueError("Parameters for Lotka Voltera must have the same "
                             "dimensions. Argument shapes: "
                             f"\n\tgrowth_rates.shape = {growth_rates.shape}"
                             f"\n\tcapacities.shape = {capacities.shape}"
                             f"\n\tinteraction_mat.shape = {interaction_mat.shape}"
                            )
        
        # Assign parameters.
        dim = len(growth_rates)
        self.growth_rates = growth_rates
        self.capacities = capacities
        self.interaction_mat = interaction_mat

        # Check if a stochastic simulation is not being used.
        if not isinstance(self, LotkaVolteraSDE):

            # Check if the user still supplied a stochastic argument.
            if (sigma is not None) and (sigma != 0):
                raise ValueError(
                    "Type interfere.dynamics.LotkaVoltera cannot be stochastic."
                    " Use interfere.dynamics.LotkaVolteraSDE or set sigma=None "
                    "during initialization. "
                    f"\n\n\tsigma={sigma}"
                )
        
        # Set dimension of the system.
        super().__init__(dim, measurement_noise_std, sigma)

    def dXdt(self, x: np.ndarray, t: Optional[float] = None):
        """Coputes derivative of a generalized Lotka Voltera model.

        dx_i/dt = r_i * x_i * (1 - (x_i +  [A x]_i) / k_i)

        Args:
            x (ndarray): The current state of the system.
            t (float): The current time. Optional because the system is 
                autonomous.

        Returns:
            The derivative of the system at x and t with respect to time.
        """
        return self.growth_rates * x * (
            1 -  (x + self.interaction_mat @ x) / self.capacities
        )
    

class LotkaVolteraSDE(StochasticDifferentialEquation, LotkaVoltera):

    def __init__(
        self,
        growth_rates: np.ndarray,
        capacities: np.ndarray,
        interaction_mat: np.ndarray,
        sigma: float,
        measurement_noise_std: Optional[np.ndarray] = None
    ):
        """Initializes class for simulating Lotka Voltera dynamics.

            dx_i/dt = r_i * x_i * (1 - x_i / k_i +  [A x]_i / k_i) + sigma * x_i
           * dW

        where r_i and k_i are the growth rates and carrying capacities of
        species i, A is the matrix of interspecies interactions and sigma
        is the magnitude of the effect of the Weiner process.

        See:
        * Vadillo 2019, "Comparing stochastic Lotka–Volterra predator-prey
        models"
        * https://en.wikipedia.org/wiki/Competitive_Lotka%E2%80%93Volterra_equations


        Args:
            growth_rates (ndarray): A length n vector of growth rates (r_i's).
            capacities (ndarray): A length n vector of carrying capacities 
                (k_i's).
            interaction_mat: A weighted (n, n) matrix of interspecies
                interactions. (A in the above equation.)
            sigma (float): Coefficient on noise. 
            measurement_noise_std (ndarray): None, or a vector with shape (n,)
                where each entry corresponds to the standard deviation of the
                measurement noise for that particular dimension of the dynamic
                model. For example, if the dynamic model had two variables x1
                and x2 and `measurement_noise_std = [1, 10]`, then
                independent gaussian noise with standard deviation 1 and 10
                will be added to x1 and x2 respectively at each point in time.  
        """
        # The following line actually uses the LotkaVoltera.__init__()
        # function by skipping StochasticDifferentialEquation in this
        # class's multiple resolution order.
        super(StochasticDifferentialEquation, self).__init__(
            growth_rates=growth_rates,
            capacities=capacities,
            interaction_mat=interaction_mat,
            measurement_noise_std=measurement_noise_std,
            sigma=sigma
        )

    def drift(self, x, t):
        return self.dXdt(x, t)
    
    def noise(self, x, t):
        return self.sigma * np.diag(x)

    
