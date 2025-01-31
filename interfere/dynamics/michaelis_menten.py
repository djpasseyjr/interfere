from typing import Optional, Union

import numpy as np

from ..utils import copy_doc
from .base import StochasticDifferentialEquation


class MichaelisMenten(StochasticDifferentialEquation):
    """Michaelis–Menten regulatory dynamics.

    Description:
        Michaelis–Menten regulatory dynamics. The nodal state xi
        is the expression level of gene i , the Hill coefficient is denoted 
        by h, and the link weights Aij > 0 are the reaction rate constants.

        dxi/dt = -xi + ∑ Aij xj^h / (1 + xj^h)
    """

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        h: float = 2,
        measurement_noise_std: Optional[np.ndarray] = None,
        sigma: Optional[Union[np.ndarray, float]] = None,
    ):
        """Initializes a Micheals Menten dynamic model.

        Description:
            Michaelis–Menten regulatory dynamics. The nodal state xi
            is the expression level of gene i , the Hill coefficient is denoted 
            by h, and the link weights Aij > 0 are the reaction rate constants.

            dxi/dt = -xi + ∑ Aij xj^h / (1 + xj^h)

        Args:
            adjacency_matrix (ndarray (n, n)): Adjacency matrix. Aij quantifies
                the strength of mutualism between species i and species j.
            h (float): The Hill coefficient.
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
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("Adjacency matrix must be square.")
            
        dim = adjacency_matrix.shape[0]
        self.adjacency_matrix = adjacency_matrix
        self.h = h
        super().__init__(dim, measurement_noise_std, sigma)


    @copy_doc(StochasticDifferentialEquation.drift)
    def drift(self, x: np.ndarray, t: float) -> np.ndarray:
        return -x + self.adjacency_matrix @ (x**self.h / (1 + x**self.h))


    @copy_doc(StochasticDifferentialEquation.noise)
    def noise(self, x: np.ndarray, t: float) -> np.ndarray:
        return self.sigma