"""Dynamic models where an intervention should have no effect.

Contains the noise models used by Cliff et. al. (2023) "Unifying pairwise
interactions..." which are:

1. Standard Normal distribution (mean μ = 0 and standard deviation σ = 1)
2. Standard Cauchy distribution (location μ = 0 and scale γ = 1).
3. Standard Exponential distribution (rate parameter γ = 1).
4. Standard Gamma distribution (shape k = 1 and scale θ = 1).
5. Standard t-distribution (degrees of freedom μ = 2).

"""
from abc import abstractmethod
from typing import Callable, Optional
import numpy as np

from .base import DEFAULT_RANGE, DynamicModel
from ..utils import copy_doc


class UncorrelatedNoise(DynamicModel):

    def _simulate(
        self,
        t: np.ndarray,
        prior_states: np.ndarray,
        prior_t: Optional[np.ndarray] = None,
        intervention: Optional[Callable[[np.ndarray, float], np.ndarray]]= None,
        rng: np.random.mtrand.RandomState = DEFAULT_RANGE,
    ) -> np.ndarray:
        """A time series where each value is i.i.d. from some distribution.

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
        if self.measurement_noise_std is not None:
            raise ValueError(
                "UncorrelatedNoise classes do not support measurement noise."
                "Initialize with `measurement_noise_std = None`."
            )
        
        initial_condition = prior_states[-1, :]
        X = self.generate_noise(len(t), self.dim, rng=rng)
        X[0] = initial_condition

        # Optionally apply intervention
        if intervention is not None:
            X_do = np.vstack([
                intervention(x, ti)
                for x, ti in zip(X, t)
            ])
            return X_do

        return X
    

    @abstractmethod
    def generate_noise(
        self, nrows: int, ncols: int, rng: np.random.RandomState=DEFAULT_RANGE
    ) -> np.ndarray:
        """Generates an array with iid entries drawn from a distribution.

        Abstract method for noise classes.

        Args:
            nrows (int): number of rows in generated noise array.
            ncols (int): number of columns in generated noise array.
            rng (RandomState): A numpy random state for reproducibility. (Uses 
                numpy's mtrand random number generator by default.)

        Returns:
            N (ndarray): A matrix with shape (nrows, ncols) containing noise.
        """
        raise NotImplementedError()
    
    @copy_doc(_simulate)
    def simulate(
        self,
        t: np.ndarray,
        prior_states: np.ndarray,
        prior_t: Optional[np.ndarray] = None,
        intervention: Optional[Callable[[np.ndarray, float], np.ndarray]]= None,
        rng: np.random.mtrand.RandomState = DEFAULT_RANGE,
    ) -> np.ndarray:
        return super().simulate(t, prior_states, prior_t, intervention, rng)


class StandardNormalNoise(UncorrelatedNoise):
    """Generates time series of i.i.d. from the standard normal."""
    
    def generate_noise(self, nrows, ncols, rng=DEFAULT_RANGE):
        return rng.normal(size=(nrows, ncols))
    
class StandardCauchyNoise(UncorrelatedNoise):
    """Generates time series of i.i.d. from the standard Cauchy distribution.
    
    Location μ = 0 and scale γ = 1.
    """
    
    def generate_noise(self, nrows, ncols, rng=DEFAULT_RANGE):
        return rng.standard_cauchy(size=(nrows, ncols))
    
class StandardExponentialNoise(UncorrelatedNoise):
    """Generates time series of i.i.d. from the standard exponential.
    
    Rate parameter γ = 1.
    """
    
    def generate_noise(self, nrows, ncols, rng=DEFAULT_RANGE):
        return rng.standard_exponential(size=(nrows, ncols))
    

class StandardGammaNoise(UncorrelatedNoise):
    """Generates time series of i.i.d. from the standard gamma.
    
    Shape k = 1 and scale θ = 1.
    """
    
    def generate_noise(self, nrows, ncols, rng=DEFAULT_RANGE):
        return rng.standard_gamma(1.0, size=(nrows, ncols))
    

class StandardTNoise(UncorrelatedNoise):
    """Generates time series of i.i.d. from the standard t distribution.
    
    Degrees of freedom μ = 2.
    """
    
    def generate_noise(self, nrows, ncols, rng=DEFAULT_RANGE):
        return rng.standard_t(2, size=(nrows, ncols))