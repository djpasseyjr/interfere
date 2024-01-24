from typing import Optional, Callable

import numpy as np

from .base import DiscreteTimeDynamics
from ..base import DEFAULT_RANGE


def logistic_map(x, alpha=3.72):
    return alpha * x * (1 - x)

def quadradic_map(x, alpha=1.45):
    return 1 - alpha * x ** 2

class CoupledMapLattice(DiscreteTimeDynamics):

    def __init__(
        self,
        adjacency_matrix: np.array,
        eps: float = 0.5,
        f: Callable = quadradic_map,
        f_params: tuple =(1.45,),
        tsteps_btw_obs: int = 1,
        measurement_noise_std: Optional[np.ndarray] = None
    ):
        """N-dimensional coupled logistic map.
        
        A coupled map lattice where coupling is determined by the passed
        adjacency matrix.

        x[n+1]_i = (1 - eps) * f(x[n]_i) + (eps / degree(i)) sum_j Aij f(x[n]_j)


        Note: This is not a continuous time system.


        Args:
            adjacency_matrix (2D array): The adjacency matrix that defines the
                way the variables are coupled. The entry A[i, j] should contain
                the weight of the link from x_j to x_i.
            eps (float): A parameter that controls the relative strenth of    
                coupling. High epsilon means greater connection to neighbors.
            f (Callable): A function to be used as f(x) in the equation above.
                Should accept a numpy array followed by an arbitrary number of 
                scalar parameters.
            f_params (tuple): A tuple of floats that will be unpacked and  
                passed to f as so: `f(x, **f_params)`.
            tsteps_per_obs (int): The number of timesteps between observations. 
                The downsample rate. (Interventions will still be applied between observations.)
            measurement_noise_std (ndarray): None, or a vector with shape (n,)
                where each entry corresponds to the standard deviation of the
                measurement noise for that particular dimension of the dynamic
                model. For example, if the dynamic model had two variables x1
                and x2 and `measurement_noise_std = [1, 10]`, then
                independent gaussian noise with standard deviation 1 and 10
                will be added to x1 and x2 respectively at each point in time.
        """
        self.adjacency_matrix = adjacency_matrix
        self.eps = eps
        self.f = f
        self.f_params = f_params
        self.tsteps_btw_obs = tsteps_btw_obs
        self.row_sums = np.sum(self.adjacency_matrix, axis=1)
        super().__init__(self.adjacency_matrix.shape[0], measurement_noise_std)


    
    def step(self, x: np.ndarray):
        """One step forward in time for a coupled map lattice

        A coupled map lattice where coupling is determined by the passed
        adjacency matrix and the map applied is the logistic map.

        x[n+1]_i = (1 - eps) * f(x[n]_i) + (eps / degree(i)) sum_j Aij f(x[n]_j)

        where f(x) = logistic_param * x * (1 - x).
        """
        x_next = (1 - self.eps) * self.f(x, *self.f_params)
        x_next += self.eps * self.adjacency_matrix @ self.f(x, *self.f_params)
        x_next /= self.row_sums
        return x_next
    

    def simulate(
        self,
        initial_condition: np.ndarray,
        time_points: np.ndarray,
        intervention: Optional[Callable[[np.ndarray, float], np.ndarray]]= None,
        rng: np.random.mtrand.RandomState = DEFAULT_RANGE,
    ) -> np.ndarray:
        """Runs a simulation of a coupled map lattice.

        Args:
            initial_condition (ndarray): A (m,) array of the initial
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
        if self.tsteps_btw_obs == 1:
            return super().simulate(
                initial_condition, time_points, intervention, rng)
        else:
            # Simulate longer and then down sample.
            new_times = np.arange(len(time_points) * self.tsteps_btw_obs)
            Xbig = super().simulate(
                initial_condition, new_times, intervention, rng)
            X = Xbig[::self.tsteps_btw_obs, :]
            return X


def coupled_logistic_map(
    adjacency_matrix: np.array,
    logistic_param=3.72,
    eps=0.5,
    measurement_noise_std: Optional[np.ndarray] = None
):
    """Initializes an N-dimensional coupled logistic map model.
    
    A coupled map lattice where coupling is determined by the passed
    adjacency matrix and the map applied is the logistic map.

    x[n+1]_i = (1 - eps) * f(x[n]_i) + (eps / degree(i)) sum_j Aij f(x[n]_j)

    where f(x) = logistic_param * x * (1 - x).

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
    return CoupledMapLattice(
        adjacency_matrix, eps, logistic_map, (logistic_param,), measurement_noise_std=measurement_noise_std)

    

def coupled_map_1dlattice(
    ndims: int,
    alpha: float,
    eps: float,
    tsteps_btw_obs: int = 1,
    measurement_noise_std: Optional[np.ndarray] = None
):
    """Intitializes a specific CoupledMapLattice where the lattice is 1D.

    The dynamics are governed by the following equation:

    x[n+1]_i = (1 - eps) * f(x[n]_i) + (eps / 2) * (f(x[n]_i+1) + f(x[n]_i-1))

    where f(x) = 1 - alpha * x^2.

    Args:
        ndmis: The number of nodes in the 1D lattice.
        alpha: The quadratic map parameter (1 - alpha * x^2)
        eps: The lattice coupling parameter.
        tsteps_per_obs (int): The number of timesteps between observations. 
                The downsample rate. (Interventions will still be applied between observations.)
        measurement_noise_std (ndarray): None, or a vector with shape (n,)
            where each entry corresponds to the standard deviation of the
            measurement noise for that particular dimension of the dynamic
            model. For example, if the dynamic model had two variables x1
            and x2 and `measurement_noise_std = [1, 10]`, then
            independent gaussian noise with standard deviation 1 and 10
            will be added to x1 and x2 respectively at each point in time.
    """
    # Construct the 1D lattice matrix
    ones = np.ones(ndims - 1)
    adjacency_matrix = np.diag(ones, k=1) + np.diag(ones, k=-1)
    return CoupledMapLattice(
        adjacency_matrix, eps, quadradic_map, (alpha,), tsteps_btw_obs, measurement_noise_std)


def coupled_map_1dlattice_frozen_chaos(
    ndims: int=10,
    measurement_noise_std: Optional[np.ndarray] = None
):
    """Intitializes a specific parameterization of a CoupledMapLattice.

    Parameters taken from S2.3.1 of the supplimental material of
    Cliff et. al. 2023 "Unifying Pairwise..."

    The dynamics are governed by the following equation:

    x[n+1]_i = (1 - eps) * f(x[n]_i) + (eps / 2) * (f(x[n]_i+1) + f(x[n]_i-1))

    with:
        f(x) = 1 - alpha * x^2
        eps = 0.2
        alpha = 1.45

    Args:
        ndmis: The number of nodes in the model.
        measurement_noise_std (ndarray): None, or a vector with shape (n,)
            where each entry corresponds to the standard deviation of the
            measurement noise for that particular dimension of the dynamic
            model. For example, if the dynamic model had two variables x1
            and x2 and `measurement_noise_std = [1, 10]`, then
            independent gaussian noise with standard deviation 1 and 10
            will be added to x1 and x2 respectively at each point in time.
    """
    return coupled_map_1dlattice(
        ndims, 1.45, 0.2, measurement_noise_std=measurement_noise_std)

    
def coupled_map_1dlattice_pattern_selection(
    ndims: int=10,
    measurement_noise_std: Optional[np.ndarray] = None
):
    """Intitializes a specific parameterization of a CoupledMapLattice.

    Parameters taken from S2.3.1 of the supplimental material of
    Cliff et. al. 2023 "Unifying Pairwise..."

    The dynamics are governed by the following equation:

    x[n+1]_i = (1 - eps) * f(x[n]_i) + (eps / 2) * (f(x[n]_i+1) + f(x[n]_i-1))

    with:
        f(x) = 1 - alpha * x^2
        eps = 0.4
        alpha = 1.71

    Args:
        ndmis: The number of nodes in the model.
        measurement_noise_std (ndarray): None, or a vector with shape (n,)
            where each entry corresponds to the standard deviation of the
            measurement noise for that particular dimension of the dynamic
            model. For example, if the dynamic model had two variables x1
            and x2 and `measurement_noise_std = [1, 10]`, then
            independent gaussian noise with standard deviation 1 and 10
            will be added to x1 and x2 respectively at each point in time.
    """
    return coupled_map_1dlattice(
        ndims, 1.71, 0.4, measurement_noise_std=measurement_noise_std)

def coupled_map_1dlattice_chaotic_brownian(
    ndims: int=10,
    measurement_noise_std: Optional[np.ndarray] = None
):
    """Intitializes a specific parameterization of a CoupledMapLattice.

    Parameters taken from S2.3.1 of the supplimental material of
    Cliff et. al. 2023 "Unifying Pairwise..."

    The dynamics are governed by the following equation:

    x[n+1]_i = (1 - eps) * f(x[n]_i) + (eps / 2) * (f(x[n]_i+1) + f(x[n]_i-1))

    with:
        f(x) = 1 - alpha * x^2
        eps = 0.1
        alpha = 1.85

    Args:
        ndmis: The number of nodes in the model.
        measurement_noise_std (ndarray): None, or a vector with shape (n,)
            where each entry corresponds to the standard deviation of the
            measurement noise for that particular dimension of the dynamic
            model. For example, if the dynamic model had two variables x1
            and x2 and `measurement_noise_std = [1, 10]`, then
            independent gaussian noise with standard deviation 1 and 10
            will be added to x1 and x2 respectively at each point in time.
    """
    return coupled_map_1dlattice(
        ndims, 1.85, 0.1, measurement_noise_std=measurement_noise_std)


def coupled_map_1dlattice_defect_turbulence(
    ndims: int=10,
    measurement_noise_std: Optional[np.ndarray] = None
):
    """Intitializes a specific parameterization of a CoupledMapLattice.

    Parameters taken from S2.3.1 of the supplimental material of
    Cliff et. al. 2023 "Unifying Pairwise..."

    The dynamics are governed by the following equation:

    x[n+1]_i = (1 - eps) * f(x[n]_i) + (eps / 2) * (f(x[n]_i+1) + f(x[n]_i-1))

    with:
        f(x) = 1 - alpha * x^2
        eps = 0.1
        alpha = 1.895

    Args:
        ndmis: The number of nodes in the model.
        measurement_noise_std (ndarray): None, or a vector with shape (n,)
            where each entry corresponds to the standard deviation of the
            measurement noise for that particular dimension of the dynamic
            model. For example, if the dynamic model had two variables x1
            and x2 and `measurement_noise_std = [1, 10]`, then
            independent gaussian noise with standard deviation 1 and 10
            will be added to x1 and x2 respectively at each point in time.
    """
    return coupled_map_1dlattice(
        ndims, 1.895, 0.1, measurement_noise_std=measurement_noise_std)


def coupled_map_1dlattice_spatiotemp_intermit1(
    ndims: int=10,
    measurement_noise_std: Optional[np.ndarray] = None
):
    """Intitializes a specific parameterization of a CoupledMapLattice.

    Parameters taken from S2.3.1 of the supplimental material of
    Cliff et. al. 2023 "Unifying Pairwise..."

    The dynamics are governed by the following equation:

    x[n+1]_i = (1 - eps) * f(x[n]_i) + (eps / 2) * (f(x[n]_i+1) + f(x[n]_i-1))

    with:
        f(x) = 1 - alpha * x^2
        eps = 0.00115
        alpha = 1.7522
        and only observed every 12 time steps.

    Args:
        ndmis: The number of nodes in the model.
        measurement_noise_std (ndarray): None, or a vector with shape (n,)
            where each entry corresponds to the standard deviation of the
            measurement noise for that particular dimension of the dynamic
            model. For example, if the dynamic model had two variables x1
            and x2 and `measurement_noise_std = [1, 10]`, then
            independent gaussian noise with standard deviation 1 and 10
            will be added to x1 and x2 respectively at each point in time.
    """
    return coupled_map_1dlattice(
        ndims, 1.71, 0.4, tsteps_btw_obs=12,measurement_noise_std=measurement_noise_std)


def coupled_map_1dlattice_spatiotemp_intermit2(
    ndims: int=10,
    measurement_noise_std: Optional[np.ndarray] = None
):
    """Intitializes a specific parameterization of a CoupledMapLattice.

    Parameters taken from S2.3.1 of the supplimental material of
    Cliff et. al. 2023 "Unifying Pairwise..."

    The dynamics are governed by the following equation:

    x[n+1]_i = (1 - eps) * f(x[n]_i) + (eps / 2) * (f(x[n]_i+1) + f(x[n]_i-1))

    with:
        f(x) = 1 - alpha * x^2
        eps = 0.3
        alpha = 1.75

    Args:
        ndmis: The number of nodes in the model.
        measurement_noise_std (ndarray): None, or a vector with shape (n,)
            where each entry corresponds to the standard deviation of the
            measurement noise for that particular dimension of the dynamic
            model. For example, if the dynamic model had two variables x1
            and x2 and `measurement_noise_std = [1, 10]`, then
            independent gaussian noise with standard deviation 1 and 10
            will be added to x1 and x2 respectively at each point in time.
    """
    return coupled_map_1dlattice(
        ndims, 1.75, 0.3, measurement_noise_std=measurement_noise_std)


def coupled_map_1dlattice_spatiotemp_chaos(
    ndims: int=10,
    measurement_noise_std: Optional[np.ndarray] = None
):
    """Intitializes a specific parameterization of a CoupledMapLattice.

    Parameters taken from S2.3.1 of the supplimental material of
    Cliff et. al. 2023 "Unifying Pairwise..."

    The dynamics are governed by the following equation:

    x[n+1]_i = (1 - eps) * f(x[n]_i) + (eps / 2) * (f(x[n]_i+1) + f(x[n]_i-1))

    with:
        f(x) = 1 - alpha * x^2
        eps = 0.3
        alpha = 2.0

    Args:
        ndmis: The number of nodes in the model.
        measurement_noise_std (ndarray): None, or a vector with shape (n,)
            where each entry corresponds to the standard deviation of the
            measurement noise for that particular dimension of the dynamic
            model. For example, if the dynamic model had two variables x1
            and x2 and `measurement_noise_std = [1, 10]`, then
            independent gaussian noise with standard deviation 1 and 10
            will be added to x1 and x2 respectively at each point in time.
    """
    return coupled_map_1dlattice(
        ndims, 2.0, 0.3, measurement_noise_std=measurement_noise_std)


def coupled_map_1dlattice_traveling_wave(
    ndims: int=10,
    measurement_noise_std: Optional[np.ndarray] = None
):
    """Intitializes a specific parameterization of a CoupledMapLattice.

    Parameters taken from S2.3.1 of the supplimental material of
    Cliff et. al. 2023 "Unifying Pairwise..."

    The dynamics are governed by the following equation:

    x[n+1]_i = (1 - eps) * f(x[n]_i) + (eps / 2) * (f(x[n]_i+1) + f(x[n]_i-1))

    with:
        f(x) = 1 - alpha * x^2
        eps = 0.5
        alpha = 1.67
        and only observed every 2000 time steps.

    Args:
        ndmis: The number of nodes in the model.
        measurement_noise_std (ndarray): None, or a vector with shape (n,)
            where each entry corresponds to the standard deviation of the
            measurement noise for that particular dimension of the dynamic
            model. For example, if the dynamic model had two variables x1
            and x2 and `measurement_noise_std = [1, 10]`, then
            independent gaussian noise with standard deviation 1 and 10
            will be added to x1 and x2 respectively at each point in time.
    """
    return coupled_map_1dlattice(
        ndims, 1.67, 0.5, tsteps_btw_obs=2000,measurement_noise_std=measurement_noise_std)


def coupled_map_1dlattice_chaotic_traveling_wave(
    ndims: int=10,
    measurement_noise_std: Optional[np.ndarray] = None
):
    """Intitializes a specific parameterization of a CoupledMapLattice.

    Parameters taken from S2.3.1 of the supplimental material of
    Cliff et. al. 2023 "Unifying Pairwise..."

    The dynamics are governed by the following equation:

    x[n+1]_i = (1 - eps) * f(x[n]_i) + (eps / 2) * (f(x[n]_i+1) + f(x[n]_i-1))

    with:
        f(x) = 1 - alpha * x^2
        eps = 0.5
        alpha = 1.69
        and only observed every 5000 time steps.

    Args:
        ndmis: The number of nodes in the model.
        measurement_noise_std (ndarray): None, or a vector with shape (n,)
            where each entry corresponds to the standard deviation of the
            measurement noise for that particular dimension of the dynamic
            model. For example, if the dynamic model had two variables x1
            and x2 and `measurement_noise_std = [1, 10]`, then
            independent gaussian noise with standard deviation 1 and 10
            will be added to x1 and x2 respectively at each point in time.
    """
    return coupled_map_1dlattice(
        ndims, 1.69, 0.5, tsteps_btw_obs=5000,measurement_noise_std=measurement_noise_std)
