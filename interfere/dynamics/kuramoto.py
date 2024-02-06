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

import numpy as np
from pyclustering.nnet.fsync import fsync_network

from .base import StochasticDifferentialEquation, DEFAULT_RANGE
from .pyclustering_utils import CONN_TYPE_MAP
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
    
    def kuramoto_intervention(x: np.array, t: float):
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
        self.omega = omega
        self.K = K
        self.adjacency_matrix = adjacency_matrix
        self.Sigma = sigma * np.diag(np.ones(dim))
        super().__init__(dim, measurement_noise_std)


    @copy_doc(StochasticDifferentialEquation.simulate)
    def simulate(
        self,
        initial_condition: np.ndarray,
        time_points: np.ndarray,
        intervention: Optional[Callable[[np.ndarray, float], np.ndarray]]= None,
        rng: np.random.mtrand.RandomState = DEFAULT_RANGE,
        dW: Optional[np.ndarray] = None,
    ) -> np.ndarray:        
        # Check initial condition.
        if np.any(np.abs(initial_condition) > 1):
            raise ValueError("Kuramoto Models require initial conditions in "
                             "the interval (-1, 1).")
        # Extract phase of the initial condition.
        theta0 = np.arcsin(initial_condition)

        # Wrap the intervention in arcsin. Its range must be [-1, 1].
        if intervention is not None:
            intervention=kuramoto_intervention_wrapper(intervention)

        # Turn off measurment noise in order to add it after the transformation.
        measurement_noise_std = self.measurement_noise_std
        self.measurement_noise_std = None

        X_do = super().simulate(
            theta0,
            time_points,
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


    def drift(self, theta: np.ndarray, t: float):
        one = np.ones(self.dim)
        prefactor = self.K / self.dim
        theta_j = np.outer(one, theta)
        theta_i = np.outer(theta, one)

        return self.omega + prefactor * (
            self.adjacency_matrix * np.sin(theta_j - theta_i)).dot(one)
        
    def noise(self, theta: np.ndarray, t):
        return self.Sigma


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
            (omega_i + (K/M) sum_j a_{ij} sin(theta_j - theta_i - A_{ij)}) dt
            + sigma dW_i

        where M is the number of nodes in the network and Z is the phase
        frustration matrix.

        Args:
            omega (np.ndarray): The  natural frequency of each oscilator.
            K (float): The coupling constant.
            adjacency_matrix (np.ndarray): A matrix containing the connectivity.
            phase_frustration (np.ndarray): 
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

    def drift(self, theta: np.ndarray, t: float):
        one = np.ones(self.dim)
        prefactor = self.K / self.dim
        theta_j = np.outer(one, theta)
        theta_i = np.outer(theta, one)

        return self.omega + prefactor * (
            self.adjacency_matrix * np.sin(
                theta_j - theta_i - self.phase_frustration)).dot(one)


class StuartLandauKuramoto(StochasticDifferentialEquation):

    def __init__(
        self,
        omega: np.ndarray,
        rho: np.ndarray,
        K: float,
        sigma: float = 0,
        type_conn = "all_to_all",
        convert_to_real = True,
        measurement_noise_std: Optional[np.ndarray] = None
    ):
        """
        Model of an oscillatory network that uses Landau-Stuart oscillator and Kuramoto model as a synchronization mechanism.
    
        The dynamics of each oscillator in the network is described by following differential Landau-Stuart equation with feedback:
    
        dz_j/dt = (i omega_j + rho_j^2 - |z_j|^2) z_j  # Stuart-landau part
                + (K/N) sum_{k=0}^N A_jk (z_k - z_j)  # Kuramoto part
    
        where i is the complex number, omega_j is the natural frequency, rho_j
        is the radius.

        Args:
            omega (np.ndarray): 1D array of natural frequencies
            rho (np.ndarray): Radius of oscillators that affects amplitude. 1D
                array with the same length as omega
            K (float): Coupling strength between oscillators.
            sigma (float): Scale of the brownian increments. Model is
                deterministic when sigma == 0.
            type_conn (str): Type of connection between oscillators. One
                of ["all_to_all", "grid_four", "grid_eight", "list_bdir",
                "dynamic"]. See pyclustering.nnet.__init__::conn_type for
                details.
            convert_to_real (bool): If true, self.simulate returns only the 
                real part or the time series.
            measurement_noise_std (ndarray): None, or a vector with shape (n,)
                where each entry corresponds to the standard deviation of the
                measurement noise for that particular dimension of the dynamic
                model. For example, if the dynamic model had two variables x1
                and x2 and measurement_noise_std = [1, 10], then independent
                gaussian noise with standard deviation 1 and 10 will be added to
                x1 and x2 respectively at each point in time. 
    """
        dim = len(omega)
        if len(rho) != dim:
            raise ValueError("omega and rho arguments must have the same size.")
        
        self.omega = omega
        self.rho = rho
        self.K = K
        self.sigma = sigma
        self.type_conn = type_conn
        self.convert_to_real = convert_to_real

        # Make independent noise matrix.
        self.Sigma = sigma * np.diag(np.ones(dim))

        # Initialize the pyclustering model.
        self.pyclustering_model = fsync_network(
            dim, omega, rho, K, CONN_TYPE_MAP[type_conn])
        
        super().__init__(dim, measurement_noise_std)

    @copy_doc(StochasticDifferentialEquation.simulate)
    def simulate(
        self,
        initial_condition: np.ndarray,
        time_points: np.ndarray,
        intervention: Optional[Callable[[np.ndarray, float], np.ndarray]]= None,
        rng: np.random.mtrand.RandomState = DEFAULT_RANGE,
        dW: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # Must have a complex initial condition.
        z0 = np.array(initial_condition, dtype=np.complex128)
        Z_do = super().simulate(
            z0,
            time_points,
            intervention=intervention,
            rng=rng,
            dW=dW
        )
        if self.convert_to_real:
            Z_do = np.real(Z_do)
        return Z_do

    def drift(self, z: np.ndarray, t: float):
        """Deterministic part of Stuart-Landau-Kuramoto dynamics.
        
        The pyclustering.nnet.fsync model uses an internal amplitude attribute
        (which is the observed node states) to compute the kuramoto
        synchronization. This internal amplitude is only
        updated on observed timesteps, however, the ode solver is used at a
        small scale to perform updates BETWEEN timesteps with neighbor amplitude
        held constant and equal to the stored amplitude.
         
        We overwrite fsync_network.__amplitude here to compute instinataneous
        dynamics without the update delay that is built into the pyclustering
        model.

        Args:
            z (complex np.ndarray): 1d array of current state. Complex numbers.
            t (float): Current time.
        """
        z_column = z.reshape(-1, 1)
        # In pyclustering.nnet.fysnc.fsync_dynamic.simulate 
        self.pyclustering_model._fsync_dynamic__amplitude = z_column

        # The function _fsync_network__calculate_amplitude accepts and returns
        # 2 float64s  to represent the complex numbers. We convert
        # the imaginary numbers to 2 floats before passing to the function.
        z_2d_float = z_column.view(np.float64)
        
        # We call the function on each node. Then we stack the 2D outputs into
        # A (self.dim x 2) array.
        dz_2d = np.vstack([
            self.pyclustering_model._fsync_network__calculate_amplitude(
                z_2d_float[node_index, :], t, node_index)
            for node_index in range(self.dim)
        ])

        # Last we convert the real representation to a 1D imaginary array
        dz = dz_2d.view(np.complex128)[:, 0]
        return dz
    
    def noise(self, x: np.ndarray, t: float):
        """Independent noise matrix scaled by scalar sigma in self.__init__."""
        return self.Sigma