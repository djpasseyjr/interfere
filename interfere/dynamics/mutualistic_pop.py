from typing import Optional, Union

import numpy as np

from ..utils import copy_doc
from .base import StochasticDifferentialEquation


class MutualisticPopulation(StochasticDifferentialEquation):
    """Mutualistic population dynamics.

    Description:
        The dynamics of the mutualistic population are governed by the
        following equation:

        dxi/dt = xi (alpha_i - theta_i xi) + ∑ Aij xi xj^h / (1 + xj^h)
    """

    def __init__(
        self,
        alpha: np.ndarray,
        theta: np.ndarray,
        adjacency_matrix: np.ndarray,
        h: float = 2,
        measurement_noise_std: Optional[np.ndarray] = None,
        sigma: Optional[Union[np.ndarray, float]] = None,
    ):
        """Initializes a model.

        Description:
            The dynamics of the mutualistic population are governed by the
            following equation:

            dxi/dt = xi (alpha_i - theta_i xi) + ∑ Aij xi xj^h / (1 + xj^h)

            Here alpha and theta are growth parameters and A is an adjacency 
            matrix describing the interactions between species.

        Args:
            alpha (ndarray (n,)): Growth rate parameter.
            theta (ndarray (n,)): Growth rate parameter.
            adjacency_matrix (ndarray (n, n)): Adjacency matrix. Aij quantifies
                the strength of mutualism between species i and species j.
            h (float): The exponent in the mutualism equation.
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
        # Check array shapes.
        if alpha.shape != theta.shape:
            raise ValueError("alpha and theta must have the same shape.")

        if alpha.shape[0] != adjacency_matrix.shape[0]:
            raise ValueError(
                "alpha, theta, and adjacency matrix must have the"
                " same dimension."
                "\nDimensions:"
                f"\n\talpha: {alpha.shape}"
                f"\n\ttheta: {theta.shape}"
                f"\n\tadjacency_matrix: {adjacency_matrix.shape}"
            )

        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("Adjacency matrix must be square.")

        self.alpha = alpha
        self.theta = theta
        self.adjacency_matrix = adjacency_matrix
        self.h = h
        dim = alpha.shape[0]

        super().__init__(dim, measurement_noise_std, sigma)


    @copy_doc(StochasticDifferentialEquation.drift)
    def drift(self, x: np.ndarray, t: float) -> np.ndarray:
        growth = x * (self.alpha - self.theta * x)
        interaction = x * self.adjacency_matrix @ (x**self.h / (1 + x**self.h))
        return growth + interaction


    @copy_doc(StochasticDifferentialEquation.noise)
    def noise(self, x: np.ndarray, t: float) -> np.ndarray:
        return self.sigma