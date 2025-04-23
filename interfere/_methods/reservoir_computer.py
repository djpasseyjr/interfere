"""A reservoir computer forecasting algorithm.

Adapted from the `rescomp` package. See Harding et. al. 2024 "Global forecasts
in reservoir computers."
"""
from math import floor
from typing import Optional
from warnings import warn

import numpy as np
import networkx as nx
from scipy import sparse
from scipy.interpolate import CubicSpline
from scipy import integrate
from scipy import optimize

from ..base import ForecastMethod, DEFAULT_RANGE
from ..utils import copy_doc


class ResComp(ForecastMethod):
    
    def __init__(
        self,
        res_sz=100,
        activ_f=np.tanh,
        mean_degree=2.0,
        ridge_alpha=1e-4,
        spect_rad=.9,
        sparse_res=True,
        sigma=0.1,
        delta=0.01,
        uniform_weights=True,
        gamma=1.,
        max_weight=2,
        min_weight=0,
        batchsize=2000,
        map_initial="relax",
        window=None,
        overlap=0.0,
        rng=DEFAULT_RANGE 
    ):
        """Initializes a reservoir computer object.


        Args:
            res_sz (int): Number of nodes in reservoir.
            mean_degree (float): Average number of edges per node in the
                reservoir network. Defaults to 2.0
            spect_rad (Float): Desired reservoir spectral radius. Defaults to 
                0.9.
            sigma (Float): Reservoir ode hyperparameter. Defaults to 0.1.
            gamma (Float): Reservoir ode hyperparameter. Defaults to 1.0.
            ridge_alpha (Float): Regularization parameter for the ridge
                regression solver. Defaults to 1e-4. 
            activ_f (Function): Activation function for reservoir nodes. Used in
                ODE. Defaults to `numpy.tanh`.
            sparse_res (Bool): Chose to use sparse matrixes or dense matrixes.
                Defaults to True. 
            uniform_weights (Bool): Choose between uniform or random edge weights.
                Defaults to True. 
            max_weight (Float): Maximim edge weight if uniform_weights=False.
                Defaults to 2.0.
            min_weight (Float): Minimum edge weight if uniform_weights=False.
                Defaults to 0.0. 
            batchsize (Int): Maximum length of training batch. Defaults to 2000.
            map_initial (str): How to pick an initial reservoir node condition.
                One of ['fixed point', 'relax', activ_f', 'psuedoinverse',
                'random', 'W_in'].See documentation of self.initial_condition()
                for details. Defaults to "relax".
            window (float): If window is not `None` the training algorithm will
                subdivide the input signal into blocks where each block
                contains `window` seconds of time. Defaults to None.
            overlap (float): Must be less than one and greater or equal to zero.
                If greater than zero, this will cause subdivided input signal
                blocks to overlap. The `overlap` variable specifies the percent
                that each signal window overlaps the previous signal window.
                Defaults to 0.0.
            rng (np.random.RandomState): A random number generator.

        Notes: `res` properties take precedence over keyword
        arguments. i.e. If `A` is dense, `ResComp(A, sparse_res=True)` will have
        a dense reservoir matrix. However, adjacency matrix weights are scaled
        after initialization to achive desired spectral radius.
    """
        self.gamma = gamma
        self.sigma = sigma
        self.delta = delta
        self.activ_f = activ_f
        self.ridge_alpha = ridge_alpha
        self.sparse_res = sparse_res
        self.spect_rad = spect_rad
        self.mean_degree = mean_degree
        self.res_sz = res_sz
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.uniform_weights = uniform_weights
        self.batchsize = batchsize
        self.is_trained = False
        self.map_initial = map_initial
        self.window = window
        self.overlap = overlap
        self.rng = rng

        super().__init__()


    def set_res_data_members(self, signal_dim, exog_dim):
        """Initialize parameter arrays that will be fit to data."""
        # W_in initialized from a uniform distribution on [-1, 1]
        self.W_in_ = 2*(self.rng.random((self.res_sz, signal_dim)) - 0.5)

        # W_exog_ initialized from a uniform distribution on [-1, 1]
        self.W_exog_ = 2*self.rng.random((self.res_sz, exog_dim)) - 1.0

        # W_out has not yet been computed
        self.W_out_ = np.zeros((signal_dim, self.res_sz))
        # Arrays to store pieces of the Tikhonov regression solution
        self.Rhat_ = np.zeros((self.res_sz, self.res_sz))
        self.Yhat_ = np.zeros((signal_dim, self.res_sz))

        # Make reservoir.
        # Create random graph adjacency matrix
        n = self.res_sz
        p = self.mean_degree / n
        A = self.random_graph(n, p)
        if not self.sparse_res:
            # Convert to dense
            A = A.toarray()
        if self.uniform_weights:
            # Set non zero entries to 1.0 (Make edge weights uniform)
            A = (A != 0).astype(float)
        # Multiply matrix by a constant to achive the desired spectral radius
        self.res_ = A
        self.scale_spect_rad()
        


    def _spectral_rad(self, A):
        """ Compute spectral radius via max radius of the strongly connected components """
        g = nx.DiGraph(A.T)
        if self.sparse_res:
            A = A.copy().todok()
        scc = nx.strongly_connected_components(g)
        rad = 0
        for cmp in scc:
            # If the component is one node, spectral radius is the edge weight of it's self loop
            if len(cmp) == 1:
                i = cmp.pop()
                max_eig = A[i,i]
            else:
                # Compute spectral radius of strongly connected components
                adj = nx.adjacency_matrix(nx.subgraph(g,cmp))
                max_eig = np.max(np.abs(np.linalg.eigvals(adj.T.toarray())))
            if max_eig > rad:
                rad = max_eig
        return rad


    def scale_spect_rad(self):
        """ Scales the spectral radius of the reservoir so that
            _spectral_rad(self.res_) = self.spect_rad
        """
        curr_rad = self._spectral_rad(self.res_)
        if not np.isclose(curr_rad,0, 1e-8):
            self.res_ *= self.spect_rad/curr_rad
        else:
            warn("Spectral radius of reservoir is close to zero. Edge weights will not be scaled")
        # end
        # Convert to csr if sparse
        if sparse.issparse(self.res_):
            self.res_ = self.res_.tocsr()


    #-------------------------------------
    # ODEs governing reervoir node states
    #-------------------------------------

    def res_ode(self, t, r, u, d):
        """ ODE to drive the reservoir node states with u(t) and a input signal d(t)"""
        transform_train = self.sigma * self.W_in_ @ u(t)
        transform_exog = self.delta * self.W_exog_ @ d(t)
        return self.gamma * (
            -1 * r + self.activ_f(
                self.res_ @ r + transform_train + transform_exog)
        )


    def trained_res_ode(self, t, r, d):
        """ Reservoir prediction ode. Assumes precomputed W_out. Accepts an input signal d(t) """
        recurrence = self.sigma * self.W_in_ @ (self.W_out_ @ r)
        transform_exog = self.delta * self.W_exog_ @ d(t)
        return self.gamma * (
            -1*r + self.activ_f(self.res_ @ r + recurrence + transform_exog))


    def initial_condition(self, u0, d0):
        """ Function to map external system initial conditions to reservoir initial conditions
            Options are set by changing the value of self.map_initial. The options work as follows:
            "fixed point"
                This sets the initial reservoir node condition to the fixed point induced by the initial
                state of the training signal. Theoretically, this should eliminate transience in the node state.
                The nonlinear root finder is sensitive to initial conditions and may not converge.
            "relax"
                This method allows the reservoir nodes to relax into a steady state corresponding to `u0`.
                This typically conincided with the fixed point above but unlike the nonlinear solver, this method
                always converged.
            "activ_f"
                This sets the reservoir initial condition to r0 = activ_f(sigma * W_in @ u0). Incidentally, should send
                the reservoir initial condition close to the attracting fixed points of the system
            "activ_f_unscaled"
                This sets the reservoir initial condition to r0 = activ_f(W_in @ u0). Included for legacy reasons.
            "pseudoinverse"
                Only for use after training. This uses the pseudoinverse of W_out to compute the initial node
                state from an inital condition from the learned system
            "random"
                Sets node states at random. Draws from [-1,1] for tanh and sin activation functions and [0, 1]
                otherwise.
        """
        u = lambda x: u0
        d = lambda x: d0

        # Solve for fixed point of the reservoir ode for a specific input and
        # control signal. May not converge.
        if self.map_initial == "fixed point":
            fixed_res_ode = lambda r: self.res_ode(0, r, u, d)
            r0 = optimize.fsolve(fixed_res_ode, np.ones(self.res_sz))

        # Allow reservoir ode to relax to its fixed point. Rarely fails.
        elif self.map_initial == "relax":

            fixed_res_ode = lambda t, r: self.res_ode(0, r, u, d)
            initial = 2*self.rng.random(self.res_sz) - 1
            tvals = np.linspace(0, 10000, 100)
            R = integrate.odeint(fixed_res_ode, initial, tvals, tfirst=True)
            r0 = R[-1,:]
            err = np.max(np.abs(r0 - R[-2, :]))
            if  err > 1e-3:
                warn(f"Reservoir fixed point failed to converge. ||r_n - r_(n+1)|| = {err}")

        # An arbitrary initial condition mapping.
        elif self.map_initial == "activ_f":
            r0 = self.activ_f(
                self.W_in_ @ (self.sigma * u0) + self.W_exog_ @ (
                    self.delta * d0)
            )

        # Another older version of the arbitrary initial condition mapping.
        elif self.map_initial == "activ_f_unscaled":
            r0 = self.activ_f(self.W_in_ @ u0 + self.W_exog_ @ d0)

        # Map between reservoir space and state space using a pseudoinverse. Can
        # only be done after the reservoir is trained.
        elif self.map_initial == "pseudoinverse":
            if not self.is_trained:
                raise ValueError("Cannot use `map_initial='pseudoinverse'` because the reservoir is untrained")
            W = self.W_out_
            r0 = np.linalg.inv(W.T @ W) @ (W.T @ u0)
        
        # Random initial reservoir condition.
        elif self.map_initial == "random":
            if (self.activ_f == np.tanh) or (self.activ_f == np.sin):
                r0 = 2*self.rng.random(self.res_sz) - 1
            else:
                r0 = self.rng.random(self.res_sz)

        # A simple mapping used here for compatability with old code.
        elif self.map_initial == "W_in":
            r0 = self.W_in_ @ u0 + self.W_exog_ @ d0

        else:
            raise ValueError(f"The value of `map_initial`='{self.map_initial}'. It must be in ['fixed point', 'relax', activ_f', 'psuedoinverse', 'random', 'W_in'], or it must be callable.")
        return r0


    #-------------------------------------
    # Default reservoir topology
    #-------------------------------------
    def weights(self,n):
        """ Weights for internal reservoir"""
        if self.uniform_weights:
            return np.ones(n)
        else:
            return (self.max_weight-self.min_weight)*self.rng.random(n) + self.min_weight


    def random_graph(self, n, p):
        """ Create the sparse adj matrix of a random directed graph
            on n nodes with probability of any link equal to p
        """
        A = sparse.random(
            n,
            n,
            density=p,
            dtype=float,
            format="lil",
            data_rvs=self.weights,
            random_state=self.rng
        )
        # Remove self edges
        for i in range(n):
             A[i,i] = 0.0
        # Add one loop to ensure positive spectral radius
        if n > 1:
            A[0, 1] = self.weights(1)
            A[1, 0] = self.weights(1)
        return A


    #---------------------------
    # Train and Predict
    #---------------------------

    def train(self, t, U, D=None):
        """ Train the reservoir computer so that it can replicate the data in U.

            Paramters
            ---------
            t (1-d array): Array of m equally spaced time 
                values corresponding to signal U.
            U (2-d array): Input signal array where the ith row corresponds to
                the training time series value at time t[i]
            D (2-d array): For each i, D[i, :] produces the exogenous signals at
                time t[i]. 
        """
        if D is None:
            D = np.zeros((len(t), 1))
            
        # Initialize internal parameter arrays.
        signal_dim = U.shape[1]
        exog_dim = D.shape[1]
        self.set_res_data_members(signal_dim, exog_dim)

        # Check if window is too small.
        if self.window is not None:
            if np.any(np.diff(t) > self.window):
                warn(f"ResComp.window = {self.window} is too small. Increasing window size.")
                window = 2 * np.min(np.diff(t))
            else:
                window = self.window
        else:
            window = None

        idxs = self._partition(t, window, self.overlap)
        for start, end in idxs:
            ti = t[start:end]
            Ui = U[start:end, :]
            Di = D[start:end, :]
            self.update_tikhanov_factors(ti, Ui, Di)
        self.W_out_ = self.solve_wout()
        self.is_trained = True


    def update_tikhanov_factors(self, t, U, D):
        """ Drive the reservoir with the u and collect state information into
            self.Rhat_ and self.Yhat_
            Parameters
            t (1 dim array): array of time values
            U (array): for each i, U[i, :] produces the state of the training signal
                at time t[i]
            D (array): For each i, D[i, :] produces the state of the input signal
                at time t[i]
        """
        # The i + batchsize + 1 ending adds one timestep of overlap to provide
        # the initial condition for the next batch. Overlap is removed after
        # the internal states are generated
        idxs = [(i, i + self.batchsize + 1) for i in range(0, len(t), self.batchsize)]
        # Set initial condition for reservoir nodes
        r0 = self.initial_condition(U[0, :], D[0, :])
        for start, end in idxs:
            ti = t[start:end]
            Ui = U[start:end, :]
            Di = D[start:end, :]
            states = self.internal_state_response(ti, Ui, Di, r0)
            # Get next initial condition and trim overlap
            states, r0 = states[:-1, :], states[-1, :]
            # Update Rhat and Yhat
            self.Rhat_ += states.T @ states
            self.Yhat_ += Ui[:-1, :].T @ states
        self.r0 = r0

    
    def internal_state_response(self, t, U, D, r0):
        """ Drive the reservoir node states with the training signal U and input signal D
            Parameters
            ----------
            t (1 dim array): array of time values
            U (array): for each i, U[i, :] produces the state of the training signal
                at time t[i]
            r0 (array): Initial condition of reservoir nodes
            D (array): For each i, D[i, :] produces the state of the input signal
                at time t[i]
            Returns
            -------
            states (array): A (len(t) x self.res_sz) array where states[i, :] corresponds
                to the reservoir node states at time t[i]

        """
        u = CubicSpline(t, U)
        d = CubicSpline(t, D)
        states = integrate.odeint(self.res_ode, r0, t, tfirst=True, args=(u,d))
        return states


    def solve_wout(self):
        """ Solve the Tikhonov regularized least squares problem (Ridge regression)
            for W_out (The readout mapping)
        """
        #Check that Rhat and Yhat aren't overflowed
        if not (np.all(np.isfinite(self.Rhat_)) and np.all(np.isfinite(self.Yhat_))):
            warn('Overflow occurred while computing ResComp regression.')

        try:
            W_out = self.Yhat_ @ np.linalg.inv(self.Rhat_ + self.ridge_alpha * np.eye(self.res_sz))
        except np.linalg.LinAlgError:
            #Try the pseudoinverse instead
            W_out = self.Yhat_ @ np.linalg.pinv(self.Rhat_ + self.ridge_alpha * np.eye(self.res_sz))
        return W_out


    def forecast(self, t, D=None, u0=None, r0=None, return_states=False):
        """ Predict the evolution of the learned system.

        Args:        
            t (ndarray): One dimensional array of time values
            D (array): for each i, D[i, :] produces the state of the exogenous
                input signal at time t[i]
            u0 (ndarray): One dimensional array of initial conditions
                corresponding to the learned system
            r0 (ndarray): One dimensional array of initial conditions 
                corresponding to reservoir nodes
            return_states (bool): Option to return states of the reservoir 
                nodes in addition to prediction
            
        Returns:
            pred (ndarray): Array with dimensions len(t) x n.   
                Pred[i,:] is a prediction of u(t[i]) where u is the learned 
                signal.
            states (ndarray): Only returned if return_states is True. Reservoir 
                node states. states[i] = r(t[i]) where r is the reservoir node
                states.
                
        Notes:
            Typically, predict is passed the state of the reservoir nodes r0 at
            the end of training. EX:
            ```
                rcomp = rc.ResComp
                rcomp.train(train_t, U) # Automatically stores the last node state
                r0 = rcomp.r0
                prediction = rcomp.forecast(test_t, r0=r0)
            ```
            If you want to see how the reservoir computer predicts the trained system will respond to an 
            arbitrary initial condition try:
            ```
                u0 = np.array([1.0, 1.0, 1.0])  
                prediction = rcomp.forecast(test_t, u0=u0)
            ```
            Unless specialized training methods are used, (i.e. windows) the above is likely to fail.
        """
        # Check for exogenous signal.
        if D is None:
            D = np.zeros((len(t), 1))
            
        # Determine reservoir condition.
        if (u0 is not None):
            r0 = self.initial_condition(u0, D[0, :])
        # When no initial condition information is provided, r0 is set to the
        # last simulated state of the reservoir computer.
        elif r0 is None :
            r0 = self.r0

        # Check if reservoir is trained.
        if not self.is_trained:
            raise Exception("Reservoir is untrained")
        
        # Integrate trained reservoir computer.
        d = CubicSpline(t, D)
        states = integrate.odeint(
            self.trained_res_ode, r0, t, tfirst=True, args=(d,))
        pred = self.W_out_ @ states.T

        # Return internal states and prediction or just return prediction.
        if return_states:
            return pred.T, states
        return pred.T


    def _partition(self, t, time_window, overlap=0.0):
        """ Partition `t` into subarrays that each include `time_window` seconds. The variable
            `overlap` determines what percent of each sub-array overlaps the previous sub-array.
            The last subarray may not contain a full time window.
        """
        if (overlap >= 1) or (overlap < 0.0):
            raise ValueError("Overlap argument must be greater than or equal to zero and less than one")
        if time_window is None:
            return ((0, -1),)
        idxs = ()
        start = 0
        tmax = t[start] + time_window
        for i,time in enumerate(t):
            while time > tmax:
                end = i
                if end - start == 1:
                    warn("rescomp.ResComp._partition partitioning time array into single entry arrays. Consider increasing time window")
                idxs += ((start,end),)
                diff = floor((end - start) * (1.0 - overlap))
                start += max(diff, 1)
                tmax = t[start] + time_window
        if len(t)-start > 1:
            idxs += ((start, len(t)),)
        return idxs
    

    @copy_doc(ForecastMethod._fit)
    def _fit(
        self,
        t: np.ndarray,
        endog_states: np.ndarray,
        exog_states: Optional[np.ndarray] = None,
    ):
        # Check for exogenous.
        if exog_states is None:
            exog_states = np.zeros((len(t), 1))

        # Train on the data.
        self.train(t, endog_states, exog_states)


    @copy_doc(ForecastMethod._predict)
    def _predict(
        self,
        t: np.ndarray,
        prior_endog_states: np.ndarray,
        prior_exog_states: Optional[np.ndarray] = None,
        prior_t: Optional[np.ndarray] = None,
        prediction_exog: Optional[np.ndarray] = None,
        rng: np.random.RandomState = DEFAULT_RANGE,
    ) -> np.ndarray:
        if prediction_exog is None:
            prediction_exog = np.zeros((len(t), 1))

        # Initial condition.
        u0 = prior_endog_states[-1, :]
        # Forecast.
        endog_pred = self.forecast(t, prediction_exog, u0=u0)
        return endog_pred
    

    def get_test_params(parameter_set="default"):
        return dict(
            res_sz=50,
            activ_f=np.tanh,
            mean_degree=2.0,
            ridge_alpha=1e-4,
            spect_rad=.9,
            sparse_res=True,
            sigma=0.1,
            uniform_weights=True,
            gamma=10.,
            max_weight=2,
            min_weight=0,
            batchsize=2000,
            map_initial="activ_f",
            window=0.02,
            overlap=0.0,
            rng = np.random.default_rng(11)
        )
    

    @staticmethod
    @copy_doc(ForecastMethod._get_optuna_params)
    def _get_optuna_params(trial, **kwargs):
        return {
            "gamma": trial.suggest_float("gamma", 0.1, 20),
            "sigma": trial.suggest_float("sigma", 0, 1),
            "delta": trial.suggest_float("delta", 0, 1),
            "mean_degree": trial.suggest_float("mean_degree", 0.1, 5),
            "res_sz": trial.suggest_int("res_sz", 10, 100),
            "window": trial.suggest_float("window", 0.001, 100, log=True),
            "spect_rad": trial.suggest_float("spect_rad", 0.1, 0.9),
            "ridge_alpha": trial.suggest_float(
                "ridge_alpha", 1e-10, 0.1, log=True),
            "map_initial": "activ_f"
        }