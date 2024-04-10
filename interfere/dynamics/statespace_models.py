"""State space dynamic models such as vector autoregression or VARIMAX.
"""
from typing import Callable, List, Optional
from warnings import warn

import numpy as np

from ..base import DynamicModel, DEFAULT_RANGE


class VARMA_Dynamics(DynamicModel):

    def __init__(
        self,
        phi_matrices: List[np.ndarray],
        theta_matrices: List[np.ndarray],
        sigma: np.ndarray,
        measurement_noise_std: Optional[np.ndarray] = None
    ):
        """Initializes a vector autoregressive moving average model.

        z_n = a_n + sum_{i=1}^p phi_i z_(n-i) + sum_{i=1}^q theta_i a_{n-i}

        with each a_n ~ MultivariateNormal(0, Sigma).

        See S2.2 of the supplimental material in Cliff et. al 2023, "Unifying pairwise..."

        Args:
            phi_matrices: A list of p (nxn) numpy arrays. The autoregressive
                components of the model. The ith array corresponds to phi_i in the formula above.
            theta_matrices: A list of q (nxn) numpy arrays. The moving average
                component of the model. The ith array corresponds to theta_i in
                the formula above.
            sigma: A (nxn) symmetric positive definite numpy array. The
                covariance of the noise.
            measurement_noise_std (ndarray): None, or a vector with shape (n,)
                where each entry corresponds to the standard deviation of the
                measurement noise for that particular dimension of the dynamic
                model. For example, if the dynamic model had two variables x1
                and x2 and `measurement_noise_std = [1, 10]`, then
                independent gaussian noise with standard deviation 1 and 10
                will be added to x1 and x2 respectively at each point in time.
        """
        phi_dims = [n for phi_i in phi_matrices for n in phi_i.shape]
        theta_dims = [n for theta_i in theta_matrices for n in theta_i.shape]
        sigma_dims = [*sigma.shape]
        n, m = phi_matrices[0].shape
        if not np.all(np.hstack([phi_dims, theta_dims, sigma_dims]) == n):
            raise ValueError("All theta and phi matrices and the sigma matrix"
                             " should be square and have the same dimension.")
        
        self.phi_matrices = phi_matrices
        self.theta_matrices = theta_matrices
        self.sigma = sigma
        super().__init__(n, measurement_noise_std)

    def simulate(
        self,
        initial_condition: np.ndarray,
        time_points: np.ndarray,
        intervention: Optional[Callable[[np.ndarray, float], np.ndarray]]=None,
        rng: np.random.mtrand.RandomState = DEFAULT_RANGE
    ) -> np.ndarray:
        """Simulates the VARMA model with arbitrary interventions.

        Args:
            initial_condition (ndarray): A (p, m) array of the initial
                condition of the dynamic model.
            time_points (ndarray): A (n,) array of the time points where the   
                dynamic model will be simulated. Must be integers
            intervention (callable): A function that accepts (1) a vector of the
                current state of the dynamic model and (2) the current time. It should return a modified state. The function will be used in the
                following way: 
                    
                If the dynamic model without the intervention can be described 
                as
                    x(t+dt) = F(x(t))

                where dt is the timestep size, x(t) is the trajectory, and F is
                the function that uses the current state to compute the state at
                the next timestep. Then the intervention function will be used
                to simulate the system

                    z(t+dt) = F(g(z(t), t), t)
                    x_do(t) = g(z(t), t)

                where x_do is the trajectory of the intervened system and g is 
                the intervention function.
            rng: A numpy random state for reproducibility. (Uses numpy's mtrand 
                random number generator by default.)
        
        Returns:
            X: An (n, m) array containing a realization of the trajectory of 
                the m dimensional system corresponding to the n times in 
                `time_points`. The first p rows contain the initial condition/
                history of the system and count towards n.
        """
        p = len(self.phi_matrices)
        q = len(self.theta_matrices)
        _, m = self.phi_matrices[0].shape

        initial_condition = initial_condition.reshape(-1, m)
        n_initial_obs, _ = initial_condition.shape

        if n_initial_obs < max(p, q):
            warn("Historic timesteps not found in initial condition. Replacing with zeros")
            time_steps_needed = max(p, q) - n_initial_obs
            initial_condition = np.vstack([
                    np.zeros(m) for i in range(time_steps_needed)
                ] + [initial_condition]
            )
        
        
        n = len(time_points)
        X = np.zeros((n, m))

        # Assign initial condition
        X[:max(p, q), :] = initial_condition
        
        # Initialize noise
        noise_vecs = rng.multivariate_normal(
            np.zeros(m), self.sigma, n - p + q)

        # Simulate
        for i in range(n - p):

            # Compute autogregressive component
            x_AR = np.sum([
                phi @ x_i
                for x_i, phi in zip(X[i:(p+i)], self.phi_matrices[::-1])
            ], axis=0)

            # Compute moving average (autocorrelated noise) component
            x_MA = np.sum([
                theta @ a_i
                for a_i, theta in zip(
                    noise_vecs[i:(q+i)], self.theta_matrices[::-1])
            ], axis=0)

            # Combine components and with one stochastic vector
            x_next = noise_vecs[q+i] + x_AR + x_MA

            # Optional intervention
            if intervention is not None:
                x_next = intervention(x_next, time_points[i])

            X[p+i, :] = x_next

        if self.measurement_noise_std is not None:
            X = self.add_measurement_noise(X, rng)
            # Preserve initial conditions.
            X[:max(p, q), :] = initial_condition

        return X

            