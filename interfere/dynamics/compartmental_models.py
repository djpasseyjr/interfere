from typing import Callable, Dict, List, Optional

import numpy as np

from ..base import StochasticDifferentialEquation

class CompartmentalModel(StochasticDifferentialEquation):
    
    def __init__(
        self,
        compartments: List[str],
        states: Dict[str, float],
        parameters: Dict[str, float],
        transitions: List[Dict],
        scalar_noise: bool = False 
        
    ):
        """Initializes a generalizable compartmental model with given compartments, initial state, transitions, and model parameters.

        E.g. the SIR model, governed by
            
            dSdt = -self.beta * S * I
            dIdt = self.beta * S * I - self.gamma * I
            dRdt = self.gamma * I

        Args: 
            compartments: an n-length list of compartments
            states: an n-length dictionary/map of compartments and initial states
            parameters: a list of parameters used within the model (e.g. beta, gamma)
            transitions: a list of transitions between compartments of the form [{'prev': [prev], 'next': [next], 'noise': [noise], 'rate': [lambda-defined relationship]}]
            scalar_noise: an indicator for whether noise should be addative or scalar (False and True respectively) 
        """
        # Input validation
        if len(compartments) != len(states.keys()):
            raise ValueError(
                "Parameters for compartments and states must have matching lengths."
                "Argument lengths: "
                f"\n\compartments = {len(compartments)}"
                f"\n\tsigma = {len(states.keys())}"
            )

        # Set dimension
        dim = len(compartments)
        super().__init__(dim)
        
        # Assign class attributes
        self.compartments = compartments
        self.states = {compartment: float(state) for compartment, state in states.items()}
        self.parameters = parameters
        self.transitions = transitions
        self.scalar_noise = scalar_noise

    def drift(self, x: np.ndarray, ti: float):
        """Deterministic component of the compartmental model"""
        # Defines each state and derivative
        current_states = dict(zip(self.compartments, x))
        derivatives = {compartment: 0 for compartment in self.compartments}

        for transition in self.transitions:
            prev_state = transition.get('prev')
            next_state = transition.get('next')

            # transition['rate'] of form: lambda states, parameters: (parameters['gamma'] * states['S'])
            rate = transition['rate'](current_states, self.parameters)

            # Retrieves non-zero, normalized noise (either 'scalar' or 'static')
            noise = max(transition.get('noise', 0.0) * np.random.normal(), 0.0)

            # Scales noise by transition quantity if instantiated as such
            if self.scalar_noise: 
                noise *= rate
            
            rate += noise
            
            # Modifying states using rate
            if prev_state is not None: 
                derivatives[prev_state] -= rate
            if next_state is not None:
                derivatives[next_state] += rate

        # Maintain ordered array of derivatives that corresponds to original set of compartments
        derivatives_arr = np.array([derivatives[compartment] for compartment in self.compartments])

        return derivatives_arr

    def noise(self, x: np.ndarray, ti: float):
        sigma = np.zeros((self.dim, self.dim))        
        return sigma        
        