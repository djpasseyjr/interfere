import traceback
from typing import Callable, Optional

import numpy as np

try:
    from pyclustering.nnet.hhn import hhn_network, hhn_parameters
    from pyclustering.nnet.legion import legion_network, legion_parameters
    from pyclustering.nnet import conn_type
    from pyclustering.nnet.fsync import fsync_network
    
except ImportError as e:
    raise ImportError(
        "ImportError occured in pyclustering import."
        "\n\n This likely occurred because `interfere` requires a special fork"
        " of the `pyclustering` library. To use the neural models, first "
        "install the fork via: "
        "\n\n pip install pyclustering@"
        "git+https://github.com/djpasseyjr/pyclustering"
        f"\n\nOriginating error text: {e}"
        f"\n\nTraceback:\n\n {traceback.format_exc()}"   
)


from .base import (
    StochasticDifferentialEquation, DEFAULT_RANGE, DiscreteTimeDynamics
)
from ..utils import copy_doc

# Maps string arguments to pyclustering arguments
CONN_TYPE_MAP = {
    "all_to_all": conn_type.ALL_TO_ALL,
    "grid_four": conn_type.GRID_FOUR,
    "grid_eight": conn_type.GRID_EIGHT,
    "list_bdir": conn_type.LIST_BIDIR,
    "dynamic": conn_type.DYNAMIC
}

# Default LEGION parameters
DEFAULT_LEGION_PARAMETERS = legion_parameters()


class HodgkinHuxleyPyclustering(StochasticDifferentialEquation):

    def __init__(
        self,
        stimulus: np.array,
        sigma: float = 1,
        nu: float = 0,
        vNa: float = 50.0,
        vK: float = -77.0,
        vL: float = -54.4,
        vRest: float = -65.0,        
        Icn1: float = 5.0,
        Icn2: float = 30.0,
        Vsyninh: float = -80.0,
        Vsynexc: float = 0.0,
        alfa_inhibitory: float = 6.0,
        betta_inhibitory: float = 0.3,
        alfa_excitatory: float = 40.0,
        betta_excitatory: float = 2.0,
        w1: float = 0.1,
        w2: float = 9.0,
        w3: float = 5.0,
        deltah: float = 400.0,
        threshold: float = -10,
        eps: float = 0.16,       
        type_conn: str = "all_to_all",
        measurement_noise_std: Optional[np.ndarray] = None
    ):
        """

        Args:
            stimulus (np.ndarray): Array of stimulus for oscillators, number of
                stimulus. Length equal to number of oscillators.
            sigma (float or ndarray): The stochastic noise parameter. Can be a
                float, a 1D matrix or a 2D matrix. Dimension must match
                dimension of model.
            nu (float): Intrinsic noise.
            vNa (float): Reverse potential of sodium current [mV].
            vK (float): Reverse potential of potassium current [mV].
            vL (float): Reverse potential of leakage current [mV].
            vRest (float): Rest potential [mV].    
            Icn1 (float): External current [mV] for central element 1.
            Icn2 (float): External current [mV] for central element 2.
            Vsyninh (float): Synaptic reversal potential [mV] for inhibitory 
                effects.
            Vsynexc (float): Synaptic reversal potential [mV] for exciting 
                effects.
            alfa_inhibitory (float): Alfa-parameter for alfa-function for 
                inhibitory effect.
            betta_inhibitory (float): Betta-parameter for alfa-function for 
                inhibitory effect.
            alfa_excitatory (float): Alfa-parameter for alfa-function for 
                excitatory effect.
            betta_excitatory (float): Betta-parameter for alfa-function for 
                excitatory effect.
            w1 (float): Strength of the synaptic connection from PN to CN1.
            w2 (float): Strength of the synaptic connection from CN1 to PN.
            w3 (float): Strength of the synaptic connection from CN2 to PN.
            deltah (float): Period of time [ms] when high strength value of 
                synaptic connection exists from CN2 to PN.
            threshold (float): Threshold of the membrane potential that should 
                exceeded by oscillator to be considered as an active.
            eps (float): Affects pulse counter.
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
        self.type_conn = type_conn

        ## Maximal conductivity for sodium current.
        gNa = 120.0 * (1 + 0.02 * nu)
        ## Maximal conductivity for potassium current.
        gK = 36.0 * (1 + 0.02 * nu)
        ## Maximal conductivity for leakage current.
        gL = 0.3 * (1 + 0.02 * nu) 
    
        # Make hhn parameter class and set parameters.
        parameters = hhn_parameters()    
        parameters.nu = nu
        parameters.gNa = gNa
        parameters.gK = gK
        parameters.gL = gL     
        parameters.vNa = vNa
        parameters.vK = vK
        parameters.vL = vL
        parameters.vRest = vRest       
        parameters.Icn1 = Icn1
        parameters.Icn2 = Icn2
        parameters.Vsyninh = Vsyninh
        parameters.Vsynexc = Vsynexc
        parameters.alfa_inhibitory = alfa_inhibitory
        parameters.betta_inhibitory = betta_inhibitory
        parameters.alfa_excitatory = alfa_excitatory
        parameters.betta_excitatory = betta_excitatory
        parameters.w1 = w1
        parameters.w2 = w2
        parameters.w3 = w3
        parameters.deltah = deltah
        parameters.threshold = threshold
        parameters.eps = eps

        # Store hhm_parameters class for use in pyclustering simulator.
        self.parameters = parameters
        
        super().__init__(dim, measurement_noise_std, sigma)

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

        # Model expects 1D initial condition.
        initial_condition = prior_states[-1, :]
        
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
        m = len(t)
        X_do = np.zeros((m, self.dim), dtype=initial_condition.dtype)

        # Optionally apply intervention to initial condition
        if intervention is not None:
            initial_condition = intervention(
                initial_condition.copy(),
                t[0]
            )
        X_do[0, :] = initial_condition

        # Asign initial condition to internal model.
        self.hhn_model._membrane_potential = list(initial_condition)

        # Compute timestep size.
        dt = (t[-1] - t[0]) / m

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

        for i, ti in zip(range(m - 1), t):
            # Current state of the model.
            x = X_do[i, :]

            # Noise differential.
            dw = self.noise(x, i) @ dW[i, :]

            # Deterministic change in neuron states.
            dN = self.drift(N, ti) * dt

            # Add noise (to membrane potential only).
            dN[:self.dim, 0] += dw

            # Next state of the model via Euler-Marayama update.
            next_N = N + dN

            # Optionally apply the intervention (to membrane potential only).
            if intervention is not None:
                next_N[:self.dim, 0] = intervention(
                    next_N[:self.dim, 0], t[i + 1])
                
                # Intervene on pyclustering model internal potential
                self.hhn_model._membrane_potential = list(next_N[:self.dim, 0])

            # Store membrane potential only.
            X_do[i + 1, :] = next_N[:self.dim, 0]

            # Update internal model neuron states
            self.step(next_N, ti, dt, rng)

            # Update neuron state array.
            N = next_N

        if self.measurement_noise_std is not None:
            # Don't add measurement noise to initial condition
            X_do[1:, :] = self.add_measurement_noise(X_do[1:, :], rng)
        
        return X_do
        
        
    def step(
        self, N: np.ndarray, ti: float, dt: float, rng: np.random.RandomState):
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
            ti, dt, *N[:num_periph, :].T)
        
        # Updation states of central neurons
        self.hhn_model._hhn_network__update_central_neurons(
            ti, *N[num_periph:, :].T)


    def drift(self, N: np.ndarray, ti: float) -> np.ndarray:
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
            dN[i] = self.hhn_model.hnn_state(neuron_state, ti, i)

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
                central_neuron_state, ti, self.hhn_model._num_osc + i
            )

        return dN


    def noise(self, x: np.ndarray, ti: float) -> np.ndarray:
        return self.sigma


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
            sigma (float or ndarray): The stochastic noise parameter. Can be a
                float, a 1D matrix or a 2D matrix. Dimension must match
                dimension of model.
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
        super().__init__(num_neurons, measurement_noise_std, sigma)


    @copy_doc(DiscreteTimeDynamics._simulate)
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

        self.legion_model = legion_network(
            self.dim // 2,
            self.parameters,
            CONN_TYPE_MAP[self.type_conn],
            ccore=False
        )
        # Assumes equally spaced time points.
        self.dt = (t[-1] - t[0]) / len(t)


        X_do = super()._simulate(
            t,
            initial_condition,
            intervention=intervention,
            rng=rng
        )
        return X_do
    

    @copy_doc(DiscreteTimeDynamics.step)
    def step(
        self,
        x: np.ndarray,
        time: float = None,
        rng: np.random.RandomState = None,
    ) -> np.ndarray:

        # Unpack the state of the excitatory and inhibitory neurons
        x_excite = x[:self.num_excite]
        x_inhib = x[self.num_excite:]

        # Overwrite the states in the legion model 
        self.legion_model._excitatory = list(x_excite)
        self.legion_model._global_inhibitor = list(x_inhib)

        # Calulate next states and extract them.
        self.legion_model._calculate_states(0, time, self.dt, self.dt/10)
        x_next = np.hstack([
            self.legion_model._excitatory, self.legion_model._global_inhibitor
        ])

        # Add stochastic system noise:
        if self.sigma != 0:
            x_next +=  self.sigma @ rng.normal(0.0, np.sqrt(self.dt))

        return x_next
    

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
            omega (np.ndarray): 1D array of natural frequencies.
            rho (np.ndarray): Radius of oscillators that affects amplitude. 1D
                array with the same length as omega.
            K (float): Coupling strength between oscillators.
            sigma (float or ndarray): The stochastic noise parameter. Can be a
                float, a 1D matrix or a 2D matrix. Dimension must match
                dimension of model.
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

        # Initialize the pyclustering model.
        self.pyclustering_model = fsync_network(
            dim, omega, rho, K, CONN_TYPE_MAP[type_conn])
        
        super().__init__(dim, measurement_noise_std, sigma)


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
        # Must have a complex initial condition.
        z0 = np.array(initial_condition, dtype=np.complex128)
        Z_do = super()._simulate(
            t,
            z0,
            prior_t,
            intervention=intervention,
            rng=rng,
            dW=dW
        )
        if self.convert_to_real:
            Z_do = np.real(Z_do)
        return Z_do

    def drift(self, z: np.ndarray, t: float) -> np.ndarray:
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
    
    def noise(self, x: np.ndarray, t: float) -> np.ndarray:
        """Independent noise matrix scaled by scalar sigma in self.__init__."""
        return self.sigma