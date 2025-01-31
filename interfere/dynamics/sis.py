from typing import Optional, Union

import numpy as np

from ..utils import copy_doc
from .base import StochasticDifferentialEquation


class SIS(
    StochasticDifferentialEquation):
    """Susceptible–infected–susceptible model.

    Description:
            Susceptible–infected–susceptible model (SIS). The nodal 
            state xi equals the infection probability of node i . The parameter 
            δi > 0 denotes the curing rate, and the link weight Aij is the 
            infection rate from node j to node i.

            dxi/dt = -δi xi + ∑ Aij (1 - xi) xj
    """

    def __init__(
        self,
        delta: np.ndarray,
        adjacency_matrix: np.ndarray,
        measurement_noise_std: Optional[np.ndarray] = None,
        sigma: Optional[Union[np.ndarray, float]] = None,
    ):
        """Initializes an SIS model.

        Description:
            Susceptible–infected–susceptible model (SIS). The nodal 
            state xi equals the infection probability of node i . The parameter 
            δi > 0 denotes the curing rate, and the link weight Aij is the 
            infection rate from node j to node i.

            dxi/dt = -δi xi + ∑ Aij (1 - xi) xj

        Args:
            delta (ndarray (n,)): Curing rate parameter.
            adjacency_matrix (ndarray (n, n)): Adjacency matrix. Aij quantifies
               is the infection rate from node j to node i.
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
        # Check that shapes match.
        if delta.shape[0] != adjacency_matrix.shape[0]:
            raise ValueError(
                "delta and adjacency matrix must have the"
                " same dimension."
                "\nDimensions:"
                f"\n\tdelta: {delta.shape}"
                f"\n\tadjacency_matrix: {adjacency_matrix.shape}"
            )

        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("Adjacency matrix must be square.")

        dim = delta.shape[0]
        self.delta = delta
        self.adjacency_matrix = adjacency_matrix
        super().__init__(dim, measurement_noise_std, sigma)


    @copy_doc(StochasticDifferentialEquation.drift)
    def drift(self, x: np.ndarray, t: float) -> np.ndarray:
        return -self.delta * x + (1 - x) * self.adjacency_matrix @ x


    @copy_doc(StochasticDifferentialEquation.noise)
    def noise(self, x: np.ndarray, t: float) -> np.ndarray:
        return self.sigma