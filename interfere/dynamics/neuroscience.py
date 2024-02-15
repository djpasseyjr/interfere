from typing import Callable, Optional

import numpy as np
from pyclustering.nnet.hhn import hhn_network, hhn_parameters
from pyclustering.nnet.legion import legion_network, legion_parameters

from .base import (
    StochasticDifferentialEquation, DEFAULT_RANGE, DiscreteTimeDynamics
)
from .pyclustering_utils import CONN_TYPE_MAP
from ..utils import copy_doc

# Default Hodgkin Huxley neural net parameters.
DEFAULT_HHN_PARAMS = hhn_parameters()
DEFAULT_HHN_PARAMS.deltah = 400

# Default LEGION parameters
DEFAULT_LEGION_PARAMETERS = legion_parameters()


class HodgkinHuxleyPyclustering(StochasticDifferentialEquation):

    def __init__(
        self,
        stimulus: np.array,
        sigma: float = 1,
        parameters: hhn_parameters = DEFAULT_HHN_PARAMS,
        type_conn: str = "all_to_all",
        measurement_noise_std: Optional[np.ndarray] = None
    ):
        """

        Args:
            stimulus (np.ndarray): Array of stimulus for oscillators, number of
                stimulus. Length equal to number of oscillators.
            sigma (float): Scale of the independent stochastic noise added to
                the system.
            parameters (hhn_parameters): A pyclustering.nnet.hhn.hhn_paramerers 
                object.
            type_conn (str): Type of connection between oscillators. One
                of ["all_to_all", "grid_four", "grid_eight", "list_bdir",
                "dynamic"]. See pyclustering.nnet.__init__::conn_type for
                details.
            measurement_noise_std (ndarray): None, or a vector with shape (n,)
                where each entry corresponds to the standard deviation of the
                measurement noise for that particular dimension of the dynamic
                model. For example, if the dynamic model had two variables x1
                and x2 and measurement_noise_std = [1, 10], then independent
                gaussian noise with standard deviation 1 and 10 will be added to
                x1 and x2 respectively at each point in time. 
        """
        dim = len(stimulus)
        self.stimulus = stimulus
        self.sigma = sigma
        self.parameters = parameters
        self.type_conn = type_conn

        # Make independent noise matrix.
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
        
        # Initialize pyclustering model.
        self.hhn_model = hhn_network(
            self.dim,
            self.stimulus,
            self.parameters,
            CONN_TYPE_MAP[self.type_conn],
            ccore=False
        )
        # Overwrite pyclustering initial noise generation with noise
        # controllable via the passed random state.
        self.hhn_model._noise = [
            rng.random() * 2.0 - 1.0
            for i in range(self.hhn_model._num_osc)
        ]

        # Allocate array to hold observed states.
        m = len(time_points)
        X_do = np.zeros((m, self.dim), dtype=initial_condition.dtype)

        # Optionally apply intervention to initial condition
        if intervention is not None:
            initial_condition = intervention(
                initial_condition.copy(),
                time_points[0]
            )
        X_do[0, :] = initial_condition

        # Asign initial condition to internal model.
        self.hhn_model._membrane_potential = list(initial_condition)

        # Compute timestep size.
        dt = (time_points[-1] - time_points[0]) / m

        if dW is None:
            # Generate sequence of weiner increments
            dW = rng.normal(0.0, np.sqrt(dt), (m - 1, self.dim))

        # Since each neuron has one observed state and three unobserved, we
        # create a matrix to house the current state of the model. Additionally
        # The HH model contains neurons that are not observed. We allocate space
        # for these too.
        num_neurons = self.hhn_model._num_osc + len(
            self.hhn_model._central_element)
        N = np.zeros((num_neurons, 4))

        for i, t in zip(range(m - 1), time_points):
            # Current state of the model.
            x = X_do[i, :]

            # Noise differential.
            dw = self.noise(x, t) @ dW[i, :]

            # Deterministic change in neuron states.
            dN = self.drift(N, t) * dt

            # Add noise (to membrane potential only).
            dN[:self.dim, 0] += dw

            # Next state of the model via Euler-Marayama update.
            next_N = N + dN

            # Optionally apply the intervention (to membrane potential only).
            if intervention is not None:
                next_N[:self.dim, 0] = intervention(
                    next_N[:self.dim, 0], time_points[i + 1])
                
                # Intervene on pyclustering model internal potential
                self.hhn_model._membrane_potential = list(next_N[:self.dim, 0])

            # Store membrane potential only.
            X_do[i + 1, :] = next_N[:self.dim, 0]

            # Update internal model neuron states
            self.step(next_N, t, dt, rng)

            # Update neuron state array.
            N = next_N

        if self.measurement_noise_std is not None:
            # Don't add measurement noise to initial condition
            X_do[1:, :] = self.add_measurement_noise(X_do[1:, :], rng)
        
        return X_do
        
        
    def step(
        self, N: np.ndarray, t: float, dt: float, rng: np.random.RandomState):
        """Discrete time dynamics, to be computed after continuous time updates.

        Args:
            N (np.ndarray): 2D array. Dimensions = (num_neurons x 4). Contains
                the current state of the model. Each row represents a neuron
                and the columns contain, membrane potential, active sodium
                channels, inactive sodium channels and active potassium
                channels respectively.
            t (float): Current time.
            dt (float): Time step size.
            rng (np.random.RandomState)
        """
        # Adapted from pyclustering.nnet.hhn_network._calculate_states().
        num_periph = self.hhn_model._num_osc

        # Noise generation. I copied it don't judge me.
        self.hhn_model._noise = [ 
            1.0 + 0.01 * (rng.random() * 2.0 - 1.0) 
            for i in range(num_periph)
        ]

        # Updating states of peripheral neurons
        self.hhn_model._hhn_network__update_peripheral_neurons(
            t, dt, *N[:num_periph, :].T)
        
        # Updation states of central neurons
        self.hhn_model._hhn_network__update_central_neurons(
            t, *N[num_periph:, :].T)


    def drift(self, N, t):
        """Computes the deterministic derivative of the model."""

        num_neurons = self.hhn_model._num_osc + len(
            self.hhn_model._central_element)

        # We initialize an array of derivatives. The dimensions are 
        # (num_neurons x 4) because each neuron has four states: membrane
        # potential, active sodium channels, inactive sodium channels and
        # active potassium channels.
        dN = np.zeros((num_neurons, 4))

        # Peripheral neuron derivatives.
        for i in range(self.hhn_model._num_osc):

            # Collect peripheral neuron state into a list.
            neuron_state = [
                self.hhn_model._membrane_potential[i],
                self.hhn_model._active_cond_sodium[i],
                self.hhn_model._inactive_cond_sodium[i],
                self.hhn_model._active_cond_potassium[i]
            ]

            # Compute the derivative of each state.
            dN[i] = self.hhn_model.hnn_state(neuron_state, t, i)

        # Central neuron derivatives.
        for i in range(len(self.hhn_model._central_element)):

            # Collect central neuron state into a list.
            central_neuron_state = [
                self.hhn_model._central_element[i].membrane_potential,
                self.hhn_model._central_element[i].active_cond_sodium,
                self.hhn_model._central_element[i].inactive_cond_sodium,
                self.hhn_model._central_element[i].active_cond_potassium
            ]

            # Compute the derivative of each state.
            dN[self.hhn_model._num_osc + i] = self.hhn_model.hnn_state(
                central_neuron_state, t, self.hhn_model._num_osc + i
            )

        return dN


    def noise(self, x: np.ndarray, t: float):
        return self.Sigma


class LEGIONPyclustering(DiscreteTimeDynamics):


    def __init__(
        self,
        num_neurons: int,
        sigma: float=0.0,
        parameters: legion_parameters = DEFAULT_LEGION_PARAMETERS,
        type_conn: str = "all_to_all",
        measurement_noise_std: Optional[np.ndarray] = None
    ):
        """LEGION (local excitatory global inhibitory oscillatory network).
        
        Args:
            num_neurons (int): Number of neurons in the model. Must be an even  
                number.
            sigma (float): Scale of the independent stochastic noise added to
                the system.
            parameters (hhn_parameters): A pyclustering.nnet.hhn.hhn_paramerers 
                object.
            type_conn (str): Type of connection between oscillators. One
                of ["all_to_all", "grid_four", "grid_eight", "list_bdir",
                "dynamic"]. See pyclustering.nnet.__init__::conn_type for
                details.
            measurement_noise_std (ndarray): None, or a vector with shape (n,)
                where each entry corresponds to the standard deviation of the
                measurement noise for that particular dimension of the dynamic
                model. For example, if the dynamic model had two variables x1
                and x2 and measurement_noise_std = [1, 10], then independent
                gaussian noise with standard deviation 1 and 10 will be added to
                x1 and x2 respectively at each point in time. 
        """
        if num_neurons % 2 == 1:
            raise ValueError("LEGION model requires an even number of neurons.")

        self.num_excite = num_neurons // 2
        self.parameters = parameters
        self.sigma = sigma
        self.type_conn = type_conn
        self.Sigma = sigma * np.diag(np.ones(num_neurons))  # Noise covariance.
        super().__init__(num_neurons, measurement_noise_std)


    @copy_doc(DiscreteTimeDynamics.simulate)
    def simulate(
        self,
        initial_condition: np.ndarray,
        time_points: np.ndarray,
        intervention: Optional[Callable[[np.ndarray, float], np.ndarray]]= None,
        rng: np.random.mtrand.RandomState = DEFAULT_RANGE,
    ):
        self.legion_model = legion_network(
            self.dim // 2,
            self.parameters,
            CONN_TYPE_MAP[self.type_conn],
            ccore=False
        )
        # Assumes equally spaced time points.
        self.dt = (time_points[-1] - time_points[0]) / len(time_points)


        X_do = super().simulate(
            initial_condition, time_points, intervention, rng)
        return X_do
    

    def step(
        self,
        x: np.ndarray,
        t: float = None,
        rng: np.random.RandomState = None,
    ):

        # Unpack the state of the excitatory and inhibitory neurons
        x_excite = x[:self.num_excite]
        x_inhib = x[self.num_excite:]

        # Overwrite the states in the legion model 
        self.legion_model._excitatory = list(x_excite)
        self.legion_model._global_inhibitor = list(x_inhib)

        # Calulate next states and extract them.
        self.legion_model._calculate_states(0, t, self.dt, self.dt/10)
        x_next = np.hstack([
            self.legion_model._excitatory, self.legion_model._global_inhibitor
        ])

        # Add stochastic system noise:
        if self.sigma != 0:
            x_next +=  self.Sigma @ rng.normal(0.0, np.sqrt(self.dt))

        return x_next