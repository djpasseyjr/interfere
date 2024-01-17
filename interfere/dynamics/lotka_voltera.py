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
        measurement_noise_std: Optional[np.ndarray] = None
    ):
        """Initializes class for simulating Lotka Voltera dynamics.

            dx_i/dt = r_i * x_i * (1 - x_i / k_i +  [A x]_i / k_i)
        
        where r_i and k_i are the growth rates and carrying capacities of
        species i, A is the matrix of interspecies interactions.

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
        self.growth_rates = growth_rates
        self.capacities = capacities
        self.interaction_mat = interaction_mat
        # Set dimension of the system.
        super().__init__(len(growth_rates), measurement_noise_std)

    def dXdt(self, x: np.ndarray, t: Optional[float] = None):
        """Coputes derivative of a generalized Lotka Voltera model.

        dx_i/dt = r_i * x_i * (1 - x_i / k_i +  [A x]_i / k_i)

        Args:
            x (ndarray): The current state of the system.
            t (float): The current time. Optional because the system is 
                autonomous.

        Returns:
            The derivative of the system at x and t with respect to time.
        """
        return self.growth_rates * x * (
            1 - x / self.capacities + self.interaction_mat @ (
                x / self.capacities
            )
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

            dx_i/dt = r_i * x_i * (1 - x_i / k_i +  [A x]_i / k_i) + sigma * dW

        where r_i and k_i are the growth rates and carrying capacities of
        species i, A is the matrix of interspecies interactions and sigma
        is the magnitude of the effect of the Weiner process.


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
            growth_rates, capacities, interaction_mat, measurement_noise_std)
        self.sigma = sigma

    def drift(self, x, t):
        return self.dXdt(x, t)
    
    def noise(self, x, t):
        return self.sigma * np.eye(self.dim)

    
