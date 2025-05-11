from abc import abstractmethod
import logging
from typing import Callable, Optional

import numpy as np
from scipy import integrate
from sdeint.wiener import Ikpw
import sdeint


from ..base import DynamicModel, DEFAULT_RANGE
from ..interventions import IdentityIntervention, ExogIntervention
from ..utils import copy_doc

SDE_INTEGRATORS = ["EulerMaruyama", "SRI2"]


class OrdinaryDifferentialEquation(DynamicModel):


    @copy_doc(DynamicModel.simulate)
    def _simulate(
        self,
        t: np.ndarray,
        prior_states: np.ndarray,
        prior_t: Optional[np.ndarray] = None,
        intervention: ExogIntervention= None,
        rng: np.random.mtrand.RandomState = DEFAULT_RANGE,
    ) -> np.ndarray:
        
        initial_condition = prior_states[-1, :]

        if intervention is None:
            return integrate.odeint(self.dXdt, initial_condition, t)
        
        # Define the derivative of the intervened system.
        intervention_dXdt = lambda x, ti: self.dXdt(intervention(x, ti), ti)

        # Integrate.
        X = integrate.odeint(intervention_dXdt, initial_condition, t)

        # Appy the intervention to the states produced by the integrator.
        X_do = np.vstack([intervention(x, ti) for x, ti in zip(X, t)])

        # Optionally add measurement noise
        if self.measurement_noise_std is not None:
            # Don't add measurement noise to initial condition
            X_do[1:, :] = self.add_measurement_noise(X_do[1:, :], rng)

        return X_do


    @abstractmethod
    def dXdt(self, x: np.ndarray, t: float):
        """Produces the derivative of the system at the supplied state and time.

        Note: Use __init__() to set the parameters of the ODE. 
        
        Args:
            x (ndarray): The current state of the system.
            t (float): The current time.

        Returns:
            The derivative of the system at x and t with respect to time.
        """
        raise NotImplementedError
    

class StochasticDifferentialEquation(DynamicModel):


    # Convienience overwrite of DynamicModel.simulate to show the SDE specific
    # arguments to the user:

    def simulate(
        self,
        t: np.ndarray,
        prior_states: np.ndarray,
        prior_t: Optional[np.ndarray] = None,
        intervention: ExogIntervention = None,
        rng: np.random.mtrand.RandomState = DEFAULT_RANGE,
        dW: Optional[np.ndarray] = None,
        numerical_method: str = "EulerMaruyama",
        I: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Simulates intervened SDE.

        Args:
            t (ndarray): A (n,) array of the time points where the   
                dynamic model will be simulated. The first entry of `t` must
                equal the last entry of `prior_t`. If `prior_t` is None, then
                the values of `prior_t` will be assumed to be evenly spaced time
                values ending with the first entry of `t`. If `t` does not
                contain evenly spaced time values, then the inference of
                `prior_t` will throw an error.
            prior_states (ndarray): A (m,) or (p, m) array of the initial
                condition or the prior states of the system.
            prior_t (ndarray): A time array with shape (p,) corresponding to the
                rows of `prior_states`. The last entry of `prior_t` must equal
                the first entry of `t`. If `prior_t` is None, then
                the values of `prior_t` will be assumed to be evenly spaced time
                values ending with the first entry of `t`. If `t` does not
                contain evenly spaced time values, then the inference of
                `prior_t` will throw an error.
            intervention (callable): A function that accepts (1) a vector of the
                current state of the dynamic model and (2) the current time. It should return a modified state. The function will be used in the
                following way. If the dynamic model without the intervention can be described 
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
            rng (RandomState): A numpy random state for reproducibility. (Uses 
                numpy's mtrand random number generator by default.)
            dW: optional array of shape (len(time_points)-1, self.dim). This is
                for detailed control, if you want to use a specific realization 
                of the  Wiener processes. If not provided, Wiener
                increments will be generated randomly.
            numerical_method (str): One of ["EulerMaruyama", "SRI2"]. See
                _simulate_ito_euler() and simulate_sri2() for details.
            I: optional array of shape (len(tspan)-1, m, m). Only used for SRI2 
                method. Specific realization of the d independent Wiener 
                processes and their multiple integrals at each time step. If not provided, suitable values will be generated randomly.

        Returns:
            X (ndarray): An (n, m) array containing a realization of the   
                trajectory of the m dimensional system corresponding to the n
                times in `t`. The first row of X contains the last row of 
                `prior_states`.
        """
        if numerical_method not in SDE_INTEGRATORS:
            raise ValueError(
                f"Numerical method must be one of {SDE_INTEGRATORS}. Recieved:"
                f"\n\t{numerical_method=}"
            )

        # Calls parents simulate() method for input validation, which then calls
        # this class's _simulate() function.
        return super().simulate(
            t=t,
            prior_states=prior_states,
            prior_t=prior_t,
            intervention=intervention,
            rng=rng,
            dW=dW,
            numerical_method=numerical_method,
            I=I,
        )


    @copy_doc(simulate)
    def _simulate(
        self,
        t: np.ndarray,
        prior_states: np.ndarray,
        prior_t: Optional[np.ndarray] = None,
        intervention: ExogIntervention = None,
        rng: np.random.mtrand.RandomState = DEFAULT_RANGE,
        dW: Optional[np.ndarray] = None,
        numerical_method: str = "EulerMaruyama",
        I: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        
        # Dispach method.
        if numerical_method == "SRI2":
            return self._simulate_sri2(
                t=t, 
                prior_states=prior_states,
                prior_t=prior_t, 
                intervention=intervention,
                rng=rng,
                dW=dW,
                I=I
            )
        
        if numerical_method == "EulerMaruyama":
            return self._simulate_ito_euler(
                t=t, 
                prior_states=prior_states,
                prior_t=prior_t, 
                intervention=intervention,
                rng=rng,
                dW=dW,
            )    

        else: 
            raise ValueError(
                f"Internal numerical method mismatch: {numerical_method=}"
                f"\nShould be one of: {SDE_INTEGRATORS}"
            )    
    

    def _simulate_ito_euler(
        self,
        t: np.ndarray,
        prior_states: np.ndarray,
        prior_t: Optional[np.ndarray] = None,
        intervention: ExogIntervention = None,
        rng: np.random.mtrand.RandomState = DEFAULT_RANGE,
        dW: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Simulates intervened SDE with Ito's method.

        Args:
            t (ndarray): A (n,) array of the time points where the   
                dynamic model will be simulated. The first entry of `t` must
                equal the last entry of `prior_t`. If `prior_t` is None, then
                the values of `prior_t` will be assumed to be evenly spaced time
                values ending with the first entry of `t`. If `t` does not
                contain evenly spaced time values, then the inference of
                `prior_t` will throw an error.
            prior_states (ndarray): A (m,) or (p, m) array of the initial
                condition or the prior states of the system.
            prior_t (ndarray): A time array with shape (p,) corresponding to the
                rows of `prior_states`. The last entry of `prior_t` must equal
                the first entry of `t`. If `prior_t` is None, then
                the values of `prior_t` will be assumed to be evenly spaced time
                values ending with the first entry of `t`. If `t` does not
                contain evenly spaced time values, then the inference of
                `prior_t` will throw an error.
            intervention (callable): A function that accepts (1) a vector of the
                current state of the dynamic model and (2) the current time. It should return a modified state. The function will be used in the
                following way. If the dynamic model without the intervention can be described 
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
            rng (RandomState): A numpy random state for reproducibility. (Uses 
                numpy's mtrand random number generator by default.)
            dW: optional array of shape (len(time_points)-1, self.dim). This is
                for advanced use, if you want to use a specific realization of
                the independent Wiener processes. If not provided Wiener
                increments will be generated randomly.

        Returns:
            X (ndarray): An (n, m) array containing a realization of the   
                trajectory of the m dimensional system corresponding to the n
                times in `t`. The first row of X contains the last row of 
                `prior_states`.
        """
        initial_condition = prior_states[-1, :]
        m = len(t)
        X_do = np.zeros((m, self.dim), dtype=initial_condition.dtype)

        # Optionally apply intervention to initial condition
        if intervention is not None:
            initial_condition = intervention(
                initial_condition.copy(),
                t[0]
            )
        X_do[0, :] = initial_condition

        dt = (t[-1] - t[0]) / m

        if dW is None:
            # Generate sequence of weiner increments
            dW = rng.normal(0.0, np.sqrt(dt), (m - 1, self.dim))

        for i, ti in zip(range(m - 1), t):
            # Current state of the model.
            x = X_do[i, :]

            # Noise differential.
            dw = self.noise(x, ti) @ dW[i, :]

            # Change in x.
            dx = self.drift(x, ti) * dt + dw

            # Next state of the model.
            next_x = x + dx

            # Optionally apply the intervention.
            if intervention is not None:
                next_x = intervention(next_x, t[i + 1])

            X_do[i + 1, :] = next_x

        if self.measurement_noise_std is not None:
            # Don't add measurement noise to initial condition
            X_do[1:, :] = self.add_measurement_noise(X_do[1:, :], rng)
        
        return X_do


    def _simulate_sri2(
        self,
        t: np.ndarray,
        prior_states: np.ndarray,
        prior_t: Optional[np.ndarray] = None,
        intervention: ExogIntervention = None,
        rng: np.random.Generator = np.random.default_rng(),
        dW: Optional[np.ndarray] = None,
        I: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Simulates intervened SDE with Strong RK2 (SRI2/SRS2).

        Args:
            t (ndarray): A (n,) array of the time points where the   
                dynamic model will be simulated. The first entry of `t` must
                equal the last entry of `prior_t`. If `prior_t` is None, then
                the values of `prior_t` will be assumed to be evenly spaced time
                values ending with the first entry of `t`. If `t` does not
                contain evenly spaced time values, then the inference of
                `prior_t` will throw an error.
            prior_states (ndarray): A (m,) or (p, m) array of the initial
                condition or the prior states of the system.
            prior_t (ndarray): A time array with shape (p,) corresponding to the
                rows of `prior_states`. The last entry of `prior_t` must equal
                the first entry of `t`. If `prior_t` is None, then
                the values of `prior_t` will be assumed to be evenly spaced time
                values ending with the first entry of `t`. If `t` does not
                contain evenly spaced time values, then the inference of
                `prior_t` will throw an error.
            intervention (callable): A function that accepts (1) a vector of the
                current state of the dynamic model and (2) the current time. It should return a modified state. The function will be used in the
                following way. If the dynamic model without the intervention can be described 
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
            rng (RandomState): A numpy random state for reproducibility. (Uses 
                numpy's mtrand random number generator by default.)
            dW: optional array of shape (len(time_points)-1, self.dim). This is
                for advanced use, if you want to use a specific realization of
                the independent Wiener processes. If not provided Wiener
                increments will be generated randomly.
            I: optional array of shape (len(tspan)-1, m, m).
                These optional arguments dW and I are for advanced use, if you 
                want to use a specific realization of the d independent Wiener 
                processes and their multiple integrals at each time step. If not provided, suitable values will be generated randomly.

        Returns:
            X (ndarray): An (n, m) array containing a realization of the   
                trajectory of the m dimensional system corresponding to the n
                times in `t`. The first row of X contains the last row of 
                `prior_states`.
        
         with optional exogenous intervention.

        See Andreas Rößler, (2010) Runge–Kutta Methods for the Strong..." 
        for the tableau.
        """
        # Set up initial state and dimensions.
        y0 = prior_states[-1, :].copy()
        if intervention is not None:
            y0 = intervention(y0, t[0])
        N = len(t)
        d = self.dim
        Y = np.zeros((N, d), dtype=y0.dtype)
        Y[0] = y0

        # Time step size. Assumed equally spaced.
        h = (t[N-1] - t[0]) / (N - 1)

        # Pre‐generate Wiener increments if needed.
        if dW is None:
            # shape (N-1, d)
            dW = rng.normal(0.0, np.sqrt(h), size=(N - 1, d))

        # Pre‐generate repeated integrals.
        if I is None:
            _, I = Ikpw(dW, h, generator=rng)

        # Define f and G wrappers.
        def f(y, tau):
            return self.drift(y, tau)

        def G(y, tau):
            return self.noise(y, tau)

        # Perform SRI2/SRS2 steps.
        for n in range(N - 1):
            Yn = Y[n, :]
            tn = t[n]
            tn1 = t[n + 1]
            sqrth = np.sqrt(h)

            # elementary increments
            Ik = dW[n, :]      # shape (d,)
            Iij = I[n, :, :]     # shape (d, d)
            assert Iij.shape == (d, d)

            # compute f*h
            fnh = f(Yn, tn) * h     # shape 

            # evaluate G at Yn
            Gn = G(Yn, tn)

            # sum1 = G·Iij / sqrt(h)
            sum1 = np.dot(Gn, Iij) / sqrth

            # H2 and H3 stages
            H20 = Yn + fnh
            H20b = np.reshape(H20, (d, 1))
            H2  = H20b + sum1
            H30 = Yn
            H3  = H20b - sum1

            # f at H20 and H20 at next time
            # TODO: Potentially intervene here.
            fn1h = f(H20, tn1) * h

            # base strong‐RK2 update
            Yn1 = Yn + 0.5 * (fnh + fn1h) + np.dot(Gn, Ik)

            # corrector term
            for k in range(d):
                # TODO: Potentially apply intervention here.
                G_H2k = G(H2[:, k], tn1)    # shape (d,d)
                G_H3k = G(H3[:, k], tn1)

                Yn1 += 0.5 * sqrth * (G_H2k[:, k] - G_H3k[:, k])

            # apply intervention
            if intervention is not None:
                Yn1 = intervention(Yn1, tn1)

            Y[n + 1] = Yn1

        if self.measurement_noise_std is not None:
            Y[1:] = self.add_measurement_noise(Y[1:], rng)

        return Y


    @abstractmethod
    def drift(self, x: np.ndarray, t: float) -> np.ndarray:
        """Returns the deterministic part of the incremental change in the SDE.

        The assumed form of the SDE is

            dX = a(x_t, t)dt + b(x_t, t)dW_t

        Where x_t is a vector, t is a scalar and dW_t is a vector of normally
        distributed realizations of independed Weiner increments.

        This function, `drift` should implement a(x, t) and map R^n x R -> R^n.

        Args:
            x (ndarray): A vector with shape (self.dim, ) containing the current
                state of the SDE.
            t (float): The current time.

        Returns:
            A vector with shape (self.dim,).
        """
        raise NotImplementedError
    

    @abstractmethod
    def noise(self, x: np.ndarray, t: float) -> np.ndarray:
        """A matrix valued function used to rescale the Weiner increments..

        The assumed form of the SDE is

            dX = a(x_t, t)dt + b(x_t, y)dW_t

        Where x_t is a vector, t is a scalar and dW_t is a vector of normally
        distributed realizations of independed Weiner increments.

        This function, `noise`, should implement b(x, t) and should 
        map R^n x R -> R^(n x n).

        Args:
            x (ndarray): A vector with shape (self.dim, ) containing the current
                state of the SDE.
            t (float): The current time.

        Returns:
            An array with shape (self.dim, self.dim).
        """
        raise NotImplementedError
    

class DiscreteTimeDynamics(DynamicModel):

    def _simulate(
        self,
        t: np.ndarray,
        prior_states: np.ndarray,
        prior_t: Optional[np.ndarray] = None,
        intervention: ExogIntervention = None,
        rng: np.random.mtrand.RandomState = DEFAULT_RANGE,
    ) -> np.ndarray:
        """Runs a simulation of the discrete time dynamic model.

       Args:
            t (ndarray): A (n,) array of the time points where the   
                dynamic model will be simulated. The first entry of `t` must
                equal the last entry of `prior_t`. If `prior_t` is None, then
                the values of `prior_t` will be assumed to be evenly spaced time
                values ending with the first entry of `t`. If `t` does not
                contain evenly spaced time values, then the inference of
                `prior_t` will throw an error.
            prior_states (ndarray): A (m,) or (p, m) array of the initial
                condition or the prior states of the system.
            prior_t (ndarray): A time array with shape (p,) corresponding to the
                rows of `prior_states`. The last entry of `prior_t` must equal
                the first entry of `t`. If `prior_t` is None, then
                the values of `prior_t` will be assumed to be evenly spaced time
                values ending with the first entry of `t`. If `t` does not
                contain evenly spaced time values, then the inference of
                `prior_t` will throw an error.
            intervention (callable): A function that accepts (1) a vector of the
                current state of the dynamic model and (2) the current time. It should return a modified state. The function will be used in the
                following way. If the dynamic model without the intervention can be described 
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
            rng (RandomState): A numpy random state for reproducibility. (Uses 
                numpy's mtrand random number generator by default.)

        Returns:
            X (ndarray): An (n, m) array containing a realization of the   
                trajectory of the m dimensional system corresponding to the n
                times in `t`. The first row of X contains the last row of 
                `prior_states`.
        """
        nsteps = len(t)
        initial_condition = prior_states[-1:, :]

        # Make sure that the simulation is not passed continuous time values
        if np.any(np.round(t) != t):
            raise ValueError("DiscreteTimeDynamics require integer time points")
        
        # Initialize array of realizations of the trajectory.
        X_do = np.zeros((nsteps, self.dim))
        X_do[0] = initial_condition

        for i in range(nsteps - 1):

            # Apply intervention to current value
            if intervention is not None:
                X_do[i] = intervention(X_do[i], t[i])

            # Compute next state
            X_do[i+1] = self.step(X_do[i], time=t[i], rng=rng)

        # After the loop, apply interention to the last step
        if intervention is not None:
            X_do[-1] = intervention(X_do[-1], t[-1])

        if self.measurement_noise_std is not None:
            # Don't add measurement noise to initial condition
            X_do[1:, :] = self.add_measurement_noise(X_do[1:, :], rng)

        return X_do
    

    @abstractmethod
    def step(
        self,
        x: np.ndarray,
        time: float = None,
        rng: np.random.mtrand.RandomState = None
    ) -> np.ndarray:
        """Uses the current state to compute the next state of the system.
        
        Args:
            x (np.ndarray): The current state of the system.
            time (float): The current time.
            rng (RandomState): A numpy random state for generating random
                numbers.

        Returns:
            x_next (np.ndarray): The next state of the system.
        """
        raise NotImplementedError()