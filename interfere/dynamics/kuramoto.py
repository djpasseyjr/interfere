"""Contains several variants of the Kuramoto model.

1. Standard Kuramoto
2. Kuramoto-Sakaguchi
3. Stuart-Landau Kuramoto

See S2.3.2 of Cliff et. al. 2023 "Unifying Pairwise..."

For the kuramoto model Cliff et. al. used three coupling schemes (1) all to all
(2) bidirectional list (3) grid four, where each oscilator is connected to four
neighbors.
"""
from typing import Optional, Callable
from warnings import warn

import numpy as np

from .base import StochasticDifferentialEquation, DEFAULT_RANGE
from ..utils import copy_doc


def kuramoto_intervention_wrapper(
        intervention: Callable[[np.ndarray, float], np.ndarray]
    ) -> Callable[[np.ndarray, float], np.ndarray]:
    """Wraps the intervention in arcsin.

    This is done so that the final simulation has the correct intervention
    values. 

    Note: the range of the intervention must be [-1, 1].

    Returns:
        kuramoto_intervention (callable): arcsin(intervention(x, t)).
    """
    
    def kuramoto_intervention(x: np.ndarray, t: float) -> np.ndarray:
        """Wraps intervention in arcsin(x)"""
        x_do = intervention(x, t)
        altered = x_do != x
        if np.any(np.abs(x_do[altered])) > 1:
            raise ValueError("For the kuramoto models, the range of " 
                             " interventions must fall within [-1, 1]")
        
        x_do[altered] = np.arcsin(x_do[altered])
        return x_do
    
    
    return kuramoto_intervention


class Kuramoto(StochasticDifferentialEquation):


    def __init__(
        self,
        omega: np.ndarray,
        K: float,
        adjacency_matrix: np.ndarray,
        sigma: float = 0,
        measurement_noise_std: Optional[np.ndarray] = None
    ):
        """Initializes a Kuramoto SDE with independent noise. 

        dtheta_i = (omega_i + (K/M) sum_j a_{ij} sin(theta_j - theta_i)) dt +
        sigma dW_i

        where M is the number of nodes in the network.

        The model returns sin(theta) to avoid discontinuities in the phase.
        Similarly, the intervention is operates on the phase, but sin(x) is
        applied to every state after the simulation is finished. 

        Args:
            omega (np.ndarray): The  natural frequency of each oscilator.
            K (float): The coupling constant.
            adjacency_matrix (np.ndarray): A matrix containing the connectivity.
            sigma (float): Parameter controlling the standard deiation of     
                system noise.
            measurement_noise_std (ndarray): None, or a vector with shape (n,)
                where each entry corresponds to the standard deviation of the
                measurement noise for that particular dimension of the dynamic
                model. For example, if the dynamic model had two variables x1
                and x2 and measurement_noise_std = [1, 10], then independent
                gaussian noise with standard deviation 1 and 10 will be added to
                x1 and x2 respectively at each point in time. 
        """
        dim = adjacency_matrix.shape[0]
        super().__init__(dim, measurement_noise_std, sigma)
        self.omega = omega
        self.K = K
        self.adjacency_matrix = adjacency_matrix


    @copy_doc(StochasticDifferentialEquation._simulate)
    def _simulate(
        self,
        t: np.ndarray,
        prior_states: np.ndarray,
        prior_t: Optional[np.ndarray] = None,
        intervention: Optional[Callable[[np.ndarray, float], np.ndarray]]= None,
        rng: np.random.mtrand.RandomState = DEFAULT_RANGE,
        dW: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        
        initial_condition = prior_states[-1:, :]

        # Check initial condition.
        if np.any(np.abs(initial_condition) > 1):
            warn("Kuramoto Models require initial conditions in "
                 "the interval (-1, 1). Initial conditions will be thresholded at -1 and 1.")
            initial_condition[initial_condition > 1] = 1
            initial_condition[initial_condition < -1] = -1
            
        # Extract phase of the initial condition.
        theta0 = np.arcsin(initial_condition)

        # Wrap the intervention in arcsin. Its range must be [-1, 1].
        if intervention is not None:
            intervention=kuramoto_intervention_wrapper(intervention)

        # Turn off measurment noise in order to add it after the transformation.
        measurement_noise_std = self.measurement_noise_std
        self.measurement_noise_std = None

        X_do = super()._simulate(
            t,
            theta0,
            prior_t=prior_t,
            intervention=intervention,
            rng=rng,
            dW=dW
        )
        # Return sin of the phase. (This undoes the arcsin transformations.)
        X_do = np.sin(X_do)

        self.measurement_noise_std = measurement_noise_std
        if measurement_noise_std is not None:
            # Don't add noise to initial condition.
            X_do[1:, :] = self.add_measurement_noise(X_do[1:, :], rng)

        return X_do


    def drift(self, theta: np.ndarray, t: float) -> np.ndarray:
        one = np.ones(self.dim)
        prefactor = self.K / self.dim
        theta_j = np.outer(one, theta)
        theta_i = np.outer(theta, one)

        return self.omega + prefactor * (
            self.adjacency_matrix * np.sin(theta_j - theta_i)).dot(one)
        
    def noise(self, theta: np.ndarray, t) -> np.ndarray:
        return self.sigma


class KuramotoSakaguchi(Kuramoto):

    def __init__(
        self,
        omega: np.ndarray,
        K: float,
        adjacency_matrix: np.ndarray,
        phase_frustration: np.ndarray,
        sigma: float = 0,
        measurement_noise_std: Optional[np.ndarray] = None
    ):
        """Initializes a Kuramoto-Sakaguchi SDE with independent noise. 

        dtheta_i = 
            (omega_i + (K/M) sum_j a_{ij} sin(theta_j - theta_i - Z_{ij)}) dt
            + sigma dW_i

        where M is the number of nodes in the network and Z is the phase
        frustration matrix.

        Args:
            omega (np.ndarray): The  natural frequency of each oscilator.
            K (float): The coupling constant.
            adjacency_matrix (np.ndarray): A matrix containing the connectivity.
                The matrix a_{ij} in the equation above.
            phase_frustration (np.ndarray): A matrix containing the phase
                frustration network. Z_{ij} in the equation above.
            sigma (float): Parameter controlling the standard deiation of     
                system noise.
            measurement_noise_std (ndarray): None, or a vector with shape (n,)
                where each entry corresponds to the standard deviation of the
                measurement noise for that particular dimension of the dynamic
                model. For example, if the dynamic model had two variables x1
                and x2 and measurement_noise_std = [1, 10], then independent
                gaussian noise with standard deviation 1 and 10 will be added to
                x1 and x2 respectively at each point in time. 
        """
        self.phase_frustration = phase_frustration
        super().__init__(
            omega, K, adjacency_matrix, sigma, measurement_noise_std)

    def drift(self, theta: np.ndarray, t: float) -> np.ndarray:
        one = np.ones(self.dim)
        prefactor = self.K / self.dim
        theta_j = np.outer(one, theta)
        theta_i = np.outer(theta, one)

        return self.omega + prefactor * (
            self.adjacency_matrix * np.sin(
                theta_j - theta_i - self.phase_frustration)).dot(one)