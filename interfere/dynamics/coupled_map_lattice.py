"""Contains classes for simulating and intervening on stochastic coupled map lattices.

Notes:
    1. For different maps, the domain of the CML is different so initial
       conditions must be sampled correctly.
        a. Logistic map: x_i in (0.0, 1.0)
        b. Quadradic map: x_i in (-1.0, 1.0)
    2. If stochastic noise is introduced, it must respect the domain of the CML
       so that the orbits don't diverge.
    3. More discrete time stochastic maps can be found in Gade et. al. 1997
       "Stochastic Resonance in..."
"""

from typing import Callable, Optional, Union
from warnings import warn

import numpy as np

from .base import DiscreteTimeDynamics
from ..base import DEFAULT_RANGE
from ..utils import copy_doc


def logistic_map(x: np.ndarray, alpha: float = 3.72) -> float:
    return alpha * x * (1 - x)


def quadradic_map(x: np.ndarray, alpha:float = 1.45) -> float:
    return 1 - alpha * x ** 2


def mod_interval(x: np.ndarray, a: float, b: float) -> float:
    """Applies the mod operation to ensure each element of x is in (a, b).

    Uses
    
        a + (b - a)(x mod 1)
    
    on entries of x that are not in the interval (a, b).

    Args:
        x (ndarray): A numpy array.
        a (float): The minimum value for entries in x.
        b (float): The maximum value for entries in x

    Returns:
        x_mod (ndarray): The array x with all entries that were
            outside (a, b) modded so they are within (a, b).
    """
    if b <= a:
        raise ValueError("Argument `a` must be less than `b` argument.")
    
    outside = (x < a) | (x > b)
    x_mod = x.copy()
    x_mod[outside] = a + (b - a)(x_mod[outside] % 1)


class CoupledMapLattice(DiscreteTimeDynamics):

    def __init__(
        self,
        adjacency_matrix: np.array,
        eps: float = 0.5,
        f: Callable = quadradic_map,
        f_params: tuple =(1.45,),
        tsteps_btw_obs: int = 1,
        measurement_noise_std: Optional[np.ndarray] = None,
        sigma: Optional[Union[float, np.ndarray]] = None
    ):
        """N-dimensional coupled logistic map.
        
        A coupled map lattice where coupling is determined by the passed
        adjacency matrix.

        x[n+1]_i = (1 - eps) * f(x[n]_i) + (eps / degree(i)) sum_j Aji f(x[n]_j)


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
            sigma (float or ndarray): The stochastic noise parameter. Can be a
                float, a 1D matrix or a 2D matrix. Dimension must match
                dimension of model.
        """
        self.adjacency_matrix = adjacency_matrix
        self.eps = eps
        self.f = f
        self.f_params = f_params
        self.tsteps_btw_obs = tsteps_btw_obs
        self.row_sums = np.sum(self.adjacency_matrix, axis=1)
        super().__init__(
            self.adjacency_matrix.shape[0], measurement_noise_std, sigma)

    
    def step(
        self, x: np.ndarray, time: float, rng: np.random.mtrand.RandomState
    ) -> np.ndarray:
        """One step forward in time for a coupled map lattice

        A coupled map lattice where coupling is determined by
        `self.adjacency_matrix` and the map is determined by `self.f`.

        x[n+1]_i = (1 - eps) * f(x[n]_i) + (eps / degree(i)) sum_j Aij f(x[n]_j)

        """
        x_next = (1 - self.eps) * self.f(x, *self.f_params)
        x_next += self.eps * self.adjacency_matrix @ self.f(x, *self.f_params)
        x_next /= self.row_sums
        return x_next
    
    @copy_doc(DiscreteTimeDynamics._simulate)
    def _simulate(
        self,
        t: np.ndarray,
        prior_states: np.ndarray,
        prior_t: Optional[np.ndarray] = None,
        intervention: Optional[Callable[[np.ndarray, float], np.ndarray]]= None,
        rng: np.random.mtrand.RandomState = DEFAULT_RANGE,
    ) -> np.ndarray:
        initial_condition = prior_states[-1:, :]
        
        if self.tsteps_btw_obs == 1:
            return super()._simulate(
                t, prior_states, prior_t, intervention, rng)
        else:
            # Simulate longer and then down sample.
            warn("Intervention is ambiguous for downsampled systems. Current "
                 "implementation freezes time between observations.")
            obs = [initial_condition]
            t_prev = t[0]
            for ti in t[1:]:
                # Use the last observation as initial condition.
                xi = obs[-1]
                frozen_times = np.hstack([
                    [t_prev],
                    ti * np.ones(self.tsteps_btw_obs + 1)
                ])
                Xstep = super()._simulate(
                    frozen_times,
                    xi,
                    prior_t=np.array([t_prev]),
                    intervention=intervention,
                    rng=rng
                )
                # The last state is the next observed state.
                obs += [Xstep[-1:, :]]
                # Current time will be previous time in next iteration.
                t_prev = ti

            X = np.vstack(obs)
            return X


class StochasticCoupledMapLattice(CoupledMapLattice):

    def __init__(
        self,
        adjacency_matrix: np.array,
        eps: float = 0.5,
        f: Callable = quadradic_map,
        f_params: tuple =(1.45,),
        sigma: Union[float, np.ndarray] = 0.0,
        x_min: float = 0.0,
        x_max: float = 1.0,
        boundary_condition: Optional[str] = "none",
        tsteps_btw_obs: int = 1,
        measurement_noise_std: Optional[np.ndarray] = None        
    ):
        """N-dimensional coupled logistic map.
        
        A coupled map lattice where coupling is determined by the passed
        adjacency matrix.

        x[n+1]_i = (1 - eps) * f(x[n]_i) + (eps / degree(i)) sum_j Aij
        f(x[n]_j) + omega[n]_i
        
        where omega[n]_i ~ Normal(0, sigma) and x[n+1]_i is constrained to
        stay within the domain.

        When x[n+1]_i is not in (x_min, x_max) we replace x[n+1]_i with
            
            x_min + (x_max - x_min) * (x[n+1]_i mod 1)

        to ensure that the system remains in it's domain.

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
            sigma (float or ndarray): The stochastic noise parameter. Can be a
                float, a 1D matrix or a 2D matrix. Dimension must match
                dimension of model.
            x_min (float): Optional minimum bound (applied to state
                elementwise) to ensure that the noise does not peturb the system out of it's domain.
            x_max (float): Optional maximum bound (applied to state
                elementwise) to ensure that the noise does not peturb the system
                out of it's domain.
            boundary_condition (string): One of ["none", "truncate", "mod"]. If
                "truncate", is selected, entries in `x` are reset to be the
                closer of `x_min` or `x_max` when they leave the domain. If
                "mod" is selected, and and entry in `x`, `x_i` is not in
                `(x_min, x_max)` then `x_i` is set to:
                `x_i = x_min + (x_max - x_min) * (x_i mod 1)`
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
        self.x_max = x_max
        self.x_min = x_min
        self.boundary_condition = boundary_condition
        super().__init__(
            adjacency_matrix, eps, f, f_params, tsteps_btw_obs, measurement_noise_std, sigma)


    
    def step(
        self, x: np.ndarray, time: float, rng: np.random.mtrand.RandomState
    ) -> np.ndarray:
        """One step forward in time for a stochastic coupled map lattice.

        A stochastic coupled map lattice where coupling is determined by
        `self.adjacency_matrix` and the map is determined by `self.f`.

        x[n+1]_i = (1 - eps) * f(x[n]_i) + (eps / degree(i)) sum_j Aij f(x[n]_j)+ w[n]_i

        where w[n] ~ N(0, sigma) and x_i is constrained to be in the interval
        (self.x_min, self.x_max).
        """
        x_next = super().step(x, time, rng) 

        # This check enables sigma == 0.0 to generate deterministic dynamics.
        if not np.all(self.sigma == 0.0):
            x_next += self.sigma @ rng.normal(size=self.dim)

        # See if the state is within boundary and apply appropriate transform. 
        if self.boundary_condition == "mod":
            x_next = mod_interval(x_next, self.x_min, self.x_max)

        elif self.boundary_condition == "truncate":
            x_next[x_next < self.x_min] = self.x_min
            x_next[x_next > self.x_max] = self.x_max

        return x_next


def coupled_logistic_map(
    adjacency_matrix: np.array,
    logistic_param=3.72,
    eps=0.5,
    sigma=0.0,
    measurement_noise_std: Optional[np.ndarray] = None
) -> StochasticCoupledMapLattice:
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
        sigma (float): The standard deviation of the additive gaussian noise
            in the model.
        measurement_noise_std (ndarray): None, or a vector with shape (n,)
            where each entry corresponds to the standard deviation of the
            measurement noise for that particular dimension of the dynamic
            model. For example, if the dynamic model had two variables x1
            and x2 and `measurement_noise_std = [1, 10]`, then
            independent gaussian noise with standard deviation 1 and 10
            will be added to x1 and x2 respectively at each point in time.
    """
    return StochasticCoupledMapLattice(
        adjacency_matrix,
        eps,
        logistic_map,
        (logistic_param,),
        sigma=sigma,
        measurement_noise_std=measurement_noise_std, 
        boundary_condition="truncate"
    )

    

def stochastic_coupled_map_1dlattice(
    dim: int,
    alpha: float,
    eps: float,
    sigma: float = 0.0,
    tsteps_btw_obs: int = 1,
    measurement_noise_std: Optional[np.ndarray] = None
) -> StochasticCoupledMapLattice:
    """Intitializes a specific CoupledMapLattice where the lattice is 1D.

    The dynamics are governed by the following equation:

    x[n+1]_i = (1 - eps) * f(x[n]_i) + (eps / 2) * (f(x[n]_i+1) + f(x[n]_i-1)) + w[n]_i

    where f(x) = 1 - alpha * x^2, w[n]_i ~ N(0, sigma) and x[n+1]_i is
    constrained to stay in the interval [-1, 1] via truncation. Setting 
    sigma = 0 will result in a deterministic system.

    Args:
        dim: The number of nodes in the 1D lattice.
        alpha: The quadratic map parameter (1 - alpha * x^2)
        eps: The lattice coupling parameter.
        sigma (float): The standard deviation of the additive gaussian noise
            in the model.
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
    ones = np.ones(dim - 1)
    adjacency_matrix = np.diag(ones, k=1) + np.diag(ones, k=-1)
    return StochasticCoupledMapLattice(
        adjacency_matrix,
        eps,
        quadradic_map,
        (alpha,),
        sigma=sigma,
        x_min=-1,
        x_max=1,
        boundary_condition="truncate",
        tsteps_btw_obs=tsteps_btw_obs, measurement_noise_std=measurement_noise_std
    )


def coupled_map_1dlattice_frozen_chaos(
    dim: int=10,
    sigma: float=0,
    measurement_noise_std: Optional[np.ndarray] = None
) -> StochasticCoupledMapLattice:
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
        dim: The number of nodes in the model.
        sigma (float): The standard deviation of the additive gaussian noise
            in the model.
        measurement_noise_std (ndarray): None, or a vector with shape (n,)
            where each entry corresponds to the standard deviation of the
            measurement noise for that particular dimension of the dynamic
            model. For example, if the dynamic model had two variables x1
            and x2 and `measurement_noise_std = [1, 10]`, then
            independent gaussian noise with standard deviation 1 and 10
            will be added to x1 and x2 respectively at each point in time.
    """
    return stochastic_coupled_map_1dlattice(
        dim, 1.45, 0.2, sigma, measurement_noise_std=measurement_noise_std)

    
def coupled_map_1dlattice_pattern_selection(
    dim: int=10, sigma: float=0,
    measurement_noise_std: Optional[np.ndarray] = None
) -> StochasticCoupledMapLattice:
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
        dim (int): The number of nodes in the model.
        sigma (float): The standard deviation of the additive gaussian noise
            in the model.
        measurement_noise_std (ndarray): None, or a vector with shape (n,)
            where each entry corresponds to the standard deviation of the
            measurement noise for that particular dimension of the dynamic
            model. For example, if the dynamic model had two variables x1
            and x2 and `measurement_noise_std = [1, 10]`, then
            independent gaussian noise with standard deviation 1 and 10
            will be added to x1 and x2 respectively at each point in time.
    """
    return stochastic_coupled_map_1dlattice(
        dim, 1.71, 0.4, sigma, measurement_noise_std=measurement_noise_std)

def coupled_map_1dlattice_chaotic_brownian(
    dim: int=10, sigma: float=0,
    measurement_noise_std: Optional[np.ndarray] = None
) -> StochasticCoupledMapLattice:
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
        dim (int): The number of nodes in the model.
        sigma (float): The standard deviation of the additive gaussian noise
            in the model.
        measurement_noise_std (ndarray): None, or a vector with shape (n,)
            where each entry corresponds to the standard deviation of the
            measurement noise for that particular dimension of the dynamic
            model. For example, if the dynamic model had two variables x1
            and x2 and `measurement_noise_std = [1, 10]`, then
            independent gaussian noise with standard deviation 1 and 10
            will be added to x1 and x2 respectively at each point in time.
    """
    return stochastic_coupled_map_1dlattice(
        dim, 1.85, 0.1, sigma, measurement_noise_std=measurement_noise_std)


def coupled_map_1dlattice_defect_turbulence(
    dim: int=10, sigma: float=0,
    measurement_noise_std: Optional[np.ndarray] = None
) -> StochasticCoupledMapLattice:
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
        dim (int): The number of nodes in the model.
        sigma (float): The standard deviation of the additive gaussian noise
            in the model.
        measurement_noise_std (ndarray): None, or a vector with shape (n,)
            where each entry corresponds to the standard deviation of the
            measurement noise for that particular dimension of the dynamic
            model. For example, if the dynamic model had two variables x1
            and x2 and `measurement_noise_std = [1, 10]`, then
            independent gaussian noise with standard deviation 1 and 10
            will be added to x1 and x2 respectively at each point in time.
    """
    return stochastic_coupled_map_1dlattice(
        dim, 1.895, 0.1, sigma=sigma,
        measurement_noise_std=measurement_noise_std)


def coupled_map_1dlattice_spatiotemp_intermit1(
    dim: int=10, sigma: float=0,
    measurement_noise_std: Optional[np.ndarray] = None
) -> StochasticCoupledMapLattice:
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
        dim (int): The number of nodes in the model.
        sigma (float): The standard deviation of the additive gaussian noise
            in the model.
        measurement_noise_std (ndarray): None, or a vector with shape (n,)
            where each entry corresponds to the standard deviation of the
            measurement noise for that particular dimension of the dynamic
            model. For example, if the dynamic model had two variables x1
            and x2 and `measurement_noise_std = [1, 10]`, then
            independent gaussian noise with standard deviation 1 and 10
            will be added to x1 and x2 respectively at each point in time.
    """
    return stochastic_coupled_map_1dlattice(
        dim, 1.71, 0.4, tsteps_btw_obs=12, sigma=sigma,
        measurement_noise_std=measurement_noise_std)


def coupled_map_1dlattice_spatiotemp_intermit2(
    dim: int=10, sigma: float=0,
    measurement_noise_std: Optional[np.ndarray] = None
) -> StochasticCoupledMapLattice:
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
        dim (int): The number of nodes in the model.
        sigma (float): The standard deviation of the additive gaussian noise
            in the model.
        measurement_noise_std (ndarray): None, or a vector with shape (n,)
            where each entry corresponds to the standard deviation of the
            measurement noise for that particular dimension of the dynamic
            model. For example, if the dynamic model had two variables x1
            and x2 and `measurement_noise_std = [1, 10]`, then
            independent gaussian noise with standard deviation 1 and 10
            will be added to x1 and x2 respectively at each point in time.
    """
    return stochastic_coupled_map_1dlattice(
        dim, 1.75, 0.3, sigma=sigma,
        measurement_noise_std=measurement_noise_std)


def coupled_map_1dlattice_spatiotemp_chaos(
    dim: int=10, sigma: float=0,
    measurement_noise_std: Optional[np.ndarray] = None
) -> StochasticCoupledMapLattice:
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
        dim (int): The number of nodes in the model.
        sigma (float): The standard deviation of the additive gaussian noise
            in the model.
        measurement_noise_std (ndarray): None, or a vector with shape (n,)
            where each entry corresponds to the standard deviation of the
            measurement noise for that particular dimension of the dynamic
            model. For example, if the dynamic model had two variables x1
            and x2 and `measurement_noise_std = [1, 10]`, then
            independent gaussian noise with standard deviation 1 and 10
            will be added to x1 and x2 respectively at each point in time.
    """
    return stochastic_coupled_map_1dlattice(
        dim, 2.0, 0.3, sigma=sigma,
        measurement_noise_std=measurement_noise_std)


def coupled_map_1dlattice_traveling_wave(
    dim: int=10, sigma: float=0,
    measurement_noise_std: Optional[np.ndarray] = None
) -> StochasticCoupledMapLattice:
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
        dim (int): The number of nodes in the model.
        sigma (float): The standard deviation of the additive gaussian noise
            in the model.
        measurement_noise_std (ndarray): None, or a vector with shape (n,)
            where each entry corresponds to the standard deviation of the
            measurement noise for that particular dimension of the dynamic
            model. For example, if the dynamic model had two variables x1
            and x2 and `measurement_noise_std = [1, 10]`, then
            independent gaussian noise with standard deviation 1 and 10
            will be added to x1 and x2 respectively at each point in time.
    """
    return stochastic_coupled_map_1dlattice(
        dim, 1.67, 0.5, tsteps_btw_obs=2000, sigma=sigma,
        measurement_noise_std=measurement_noise_std)


def coupled_map_1dlattice_chaotic_traveling_wave(
    dim: int=10, sigma: float=0,
    measurement_noise_std: Optional[np.ndarray] = None
) -> StochasticCoupledMapLattice:
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
        dim (int): The number of nodes in the model.
        sigma (float): The standard deviation of the additive gaussian noise
            in the model.
        measurement_noise_std (ndarray): None, or a vector with shape (n,)
            where each entry corresponds to the standard deviation of the
            measurement noise for that particular dimension of the dynamic
            model. For example, if the dynamic model had two variables x1
            and x2 and `measurement_noise_std = [1, 10]`, then
            independent gaussian noise with standard deviation 1 and 10
            will be added to x1 and x2 respectively at each point in time.
    """
    return stochastic_coupled_map_1dlattice(
        dim, 1.69, 0.5, tsteps_btw_obs=5000, sigma=sigma,
        measurement_noise_std=measurement_noise_std)