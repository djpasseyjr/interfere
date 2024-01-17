from typing import Optional, Callable

import numpy as np

from .base import DiscreteTimeDynamics


class CoupledLogisticMaps(DiscreteTimeDynamics):

    def __init__(
        self,
        adjacency_matrix: np.array,
        logistic_param=3.72,
        eps=0.5,
        measurement_noise_std: Optional[np.ndarray] = None
    ):
        """N-dimensional coupled logistic map.
        
        A coupled map lattice where coupling is determined by the passed
        adjacency matrix and the map applied is the logistic map.

        x[n+1]_i = (1 - eps) * f(x[n]_i) + (eps / degree(i)) sum_j Aij f(x[n]_j)

        where f(x) = logistic_param * x * (1 - x).

        Note: This is not a continuous time system.


        Args:
            adjacency_matrix (2D array): The adjacency matrix that defines the
                way the variables are coupled. The entry A[i, j] should contain
                the weight of the link from x_j to x_i.
            logistic_param (float): The logistic map weight parameter   
                r, where the map is f(x) = rx(1-x).
            eps (float): A parameter that controls the relative strenth of    
                coupling. High epsilon means greater connection to neighbors.
            measurement_noise_std (ndarray): None, or a vector with shape (n,)
                where each entry corresponds to the standard deviation of the
                measurement noise for that particular dimension of the dynamic
                model. For example, if the dynamic model had two variables x1
                and x2 and `measurement_noise_std = [1, 10]`, then
                independent gaussian noise with standard deviation 1 and 10
                will be added to x1 and x2 respectively at each point in time.
        """
        self.adjacency_matrix = adjacency_matrix
        self.logistic_param = logistic_param
        self.eps = eps
        self.row_sums = np.sum(self.adjacency_matrix, axis=1)
        super().__init__(self.adjacency_matrix.shape[0], measurement_noise_std)
    
    def logistic_map(self, x):
        return self.logistic_param * x * (1 - x)
    
    def step(self, x: np.ndarray):
        """One step forward in time for a coupled map lattice

        A coupled map lattice where coupling is determined by the passed
        adjacency matrix and the map applied is the logistic map.

        x[n+1]_i = (1 - eps) * f(x[n]_i) + (eps / degree(i)) sum_j Aij f(x[n]_j)

        where f(x) = logistic_param * x * (1 - x).
        """
        x_next = (1 - self.eps) * self.logistic_map(x)
        x_next += self.eps * self.adjacency_matrix @ self.logistic_map(x)
        x_next /= self.row_sums
        return x_next