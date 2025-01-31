from typing import Optional, Union

import numpy as np

from ..utils import copy_doc
from .base import StochasticDifferentialEquation


class WilsonCowan(StochasticDifferentialEquation):
    """Wilson–Cowan dynamic model.

    Description:
        Wilson–Cowan model. Here, the nodal state xi is the activity of 
        neuron i, and the parameters τ and μ are the slope and the threshold
        of the neural activation function. The link weight Aij specifies the
        number and strength of synapses from neuron j to neuron i.

        dxi/dt = -xi + ∑ Aij / (1 + exp(-τ (xj - µ)))
    """

    def __init__(
        self,
        tau: float,
        mu: float,
        adjacency_matrix: np.ndarray,
        measurement_noise_std: Optional[np.ndarray] = None,
        sigma: Optional[Union[np.ndarray, float]] = None,
    ):
        """Initializes a model.

        Description:
            Wilson–Cowan model. Here, the nodal state xi is the activity of 
            neuron i, and the parameters τ and μ are the slope and the threshold
            of the neural activation function. The link weight Aij specifies the
            number and strength of synapses from neuron j to neuron i.

            dxi/dt = -xi + ∑ Aij / (1 + exp(-τ (xj - µ)))

        Args:
            tau (float): The slope parameter.
            mu (float): The threshold parameter.
            adjacency_matrix (ndarray (n, n)): Adjacency matrix. Aij quantifies
                number and strength of synapses from neuron j to neuron i.
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

        References:
            Prasse, B. and Van Mieghem, P. (2022) ‘Predicting network dynamics 
            without requiring the knowledge of the interaction graph’, PNAS
        """
        # Make sure A is square.
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("Adjacency matrix must be square.")

        dim = adjacency_matrix.shape[0]
        self.tau = tau
        self.mu = mu
        self.adjacency_matrix = adjacency_matrix

        super().__init__(dim, measurement_noise_std, sigma)


    @copy_doc(StochasticDifferentialEquation.drift)
    def drift(self, x: np.ndarray, t: float) -> np.ndarray:
        return -x + self.adjacency_matrix @ (1 / (1 + np.exp(-self.tau * (x - self.mu))))


    @copy_doc(StochasticDifferentialEquation.noise)
    def noise(self, x: np.ndarray, t: float) -> np.ndarray:
        return self.sigma