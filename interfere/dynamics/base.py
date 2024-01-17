from abc import abstractmethod
from typing import Callable, Optional

import numpy as np
from scipy import integrate


from ..base import DynamicModel, DEFAULT_RANGE


class OrdinaryDifferentialEquation(DynamicModel):

    def simulate(
        self,
        initial_condition: np.ndarray,
        time_points: np.ndarray,
        intervention: Optional[Callable[[np.ndarray, float], np.ndarray]]= None,
        rng: np.random.mtrand.RandomState = DEFAULT_RANGE
    ) -> np.ndarray:
        """
        Runs a simulation of the differential equaltion model.

        Args:
            initial_condition: A (m,) array of the initial condition
                or the historical conditions of the dynamic model.
            time_points: A (n,) array of the time points where the dynamic 
                model will be simulated.
            intervention: A function that accepts (1) a vector of the
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
            X_do: An (n, m) array containing a realization of the trajectory of 
                the m dimensional system in response to the intervention where
                the n rows correspond to the n times in `time_points`.
        """
        if intervention is None:
            return integrate.odeint(self.dXdt, initial_condition, time_points)
        
        # Define the derivative of the intervened system.
        intervention_dXdt = lambda x, t: self.dXdt(intervention(x, t), t)

        # Integrate.
        X = integrate.odeint(intervention_dXdt, initial_condition, time_points)

        # Appy the intervention to the states produced by the integrator.
        X_do = np.vstack([intervention(x, t) for x, t in zip(X, time_points)])

        # Optionally add measurement noise
        if self.measurement_noise_std is not None:
            X_do = self.add_measurement_noise(X_do, rng)

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

    def simulate(
        self,
        initial_condition: np.ndarray,
        time_points: np.ndarray,
        intervention: Optional[Callable[[np.ndarray, float], np.ndarray]]= None,
        # TODO: Change measurement noise so it is a data member.
        rng: np.random.mtrand.RandomState = DEFAULT_RANGE,
        dW: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Simulates intervened SDE with Ito's method.

        Args:
            initial_condition: A (m,) array of the initial condition
                of the dynamic model.
            time_points: A (n,) array of the time points where the dynamic 
                model will be simulated.
            intervention: A function that accepts (1) a vector of the
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
            dW: optional array of shape (len(time_points)-1, self.dim). This is
                for advanced use, if you want to use a specific realization of
                the independent Wiener processes. If not provided Wiener
                increments will be generated randomly.

        Returns:
            X_do: An (n, m) array containing a realization of the trajectory of 
                the m dimensional system in response to the intervention where
                the n rows correspond to the n times in `time_points`.
        """
        m = len(time_points)
        X_do = np.zeros((m, self.dim))

        # Optionally apply intervention to initial condition
        if intervention is not None:
            initial_condition = intervention(
                initial_condition.copy(),
                time_points[0]
            )
        X_do[0, :] = initial_condition

        dt = (time_points[-1] - time_points[0]) / m

        if dW is None:
            # Generate sequence of weiner increments
            dW = rng.normal(0.0, np.sqrt(dt), (m - 1, self.dim))

        for i, t in zip(range(m - 1), time_points):
            # Current state of the model.
            x = X_do[i, :]

            # Noise differential.
            dw = self.noise(x, t) @ dW[i, :]

            # Change in x.
            dx = self.drift(x, t) * dt + dw

            # Next state of the model.
            next_x = x + dx

            # Optionally apply the intervention.
            if intervention is not None:
                next_x = intervention(next_x, time_points[i + 1])

            X_do[i + 1, :] = next_x

        if self.measurement_noise_std is not None:
            X_do = self.add_measurement_noise(X_do, rng)
        
        return X_do

    @abstractmethod
    def drift(self, x: np.ndarray, t: float):
        """Returns the deterministic part of the incremental change in the SDE.

        The assumed form of the SDE is

            dX = a(x_t, t)dt + b(x_t, y)dW_t

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
    def noise(self, x: np.ndarray, t: float):
        """A matrix valued function used to rescale the Weiner increments..

        The assumed form of the SDE is

            dX = a(x_t, t)dt + b(x_t, y)dW_t

        Where x_t is a vector, t is a scalar and dW_t is a vector of normally
        distributed realizations of independed Weiner increments.

        This function, `noise`, should implement b(x, t) and should 
        map R^n x R -> R^n x R^n.

        Args:
            x (ndarray): A vector with shape (self.dim, ) containing the current
            state of the SDE.
            t (float): The current time.

        Returns:
            An array with shape (self.dim, self.dim).
        """
        raise NotImplementedError
    

class DiscreteTimeDynamics(DynamicModel):

    def simulate(
        self,
        initial_condition: np.ndarray,
        time_points: np.ndarray,
        intervention: Optional[Callable[[np.ndarray, float], np.ndarray]]= None,
        rng: np.random.mtrand.RandomState = DEFAULT_RANGE,
    ) -> np.ndarray:
        """Runs a simulation of the discrete time dynamic model.

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
        nsteps = len(time_points)

        # Make sure that the simulation is not passed continuous time values
        if np.any(np.round(time_points) != time_points):
            raise ValueError("DiscreteTimeDynamics require integer time points")
        
        # Initialize array of realizations of the trajectory.
        X_do = np.zeros((nsteps, self.dim))
        X_do[0] = initial_condition

        for i in range(nsteps - 1):

            # Apply intervention to current value
            if intervention is not None:
                X_do[i] = intervention(X_do[i], time_points[i])

            # Compute next state
            X_do[i+1] = self.step(X_do[i])

        # After the loop, apply interention to the last step
        if intervention is not None:
            X_do[-1] = intervention(X_do[-1], time_points[-1])

        return X_do
    
    @abstractmethod
    def step(self, x_n: np.ndarray):
        """Uses the current state to compute the next state of the system.
        
        Args:
            x_n (np.ndarray): The current state of the system.
        """
        raise NotImplementedError()