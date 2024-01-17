from typing import Callable, Iterable, List, Union

import numpy as np

from .base import Intervention


class ExogIntervention(Intervention):
    """A class describing an exogeneous intervention on a system.
    
    In an exogeneous intervention, some of the signals are treated as
    though they were under exogenous control.

    This class must contain the indexes of the signals that are being
    controled exogenously and must also contain a mechanism for computing
    the intervened signal from the non intervened signal. (i.e. replacing
    values with the exogenous values.)
    """
    def __init__(self, intervened_idxs: List[int]):
        self.intervened_idxs = intervened_idxs
    
    def split_exogeneous(self, X: np.ndarray):
        """Splits exogeneous columns from endogenous."""
        exog_X = X[:, self.intervened_idxs]
        endo_X = np.delete(X, self.intervened_idxs, axis=1)
        return endo_X, exog_X
    
    def combine_exogeneous(self, endo_X: np.ndarray, exog_X: np.ndarray):
        """Recombines endogenous and endogenous signals."""
        X = np.insert(endo_X, self.intervened_idxs, exog_X, axis=1)
        return X
    

class PerfectIntervention(ExogIntervention):

    def __init__(
        self,
        intervened_idxs: Union[int, Iterable[int]],
        constants: Union[float, Iterable[float]]
    ):
        """Creates a perfect intervention function.

        A perfect intervention replaces variables with constant values.
        This function generates intervention functions that replace

        Args:
            intervened_idxs (int or collection of ints): The indexes where the intervention
                will be applied.
            constants (float or collection of floats): The values that the variables
                at corresponding to each index will be pinned to.

        Returns:
            intervention: A function

        Examples:
            intervention = perfect_intervention(0, 1.6)
            intervention([10.0, 4.0, 4.0], 0) == [1.6, 4, 4] # (True.)

            intervention = perfect_intervention([0, 2], [1.6, .7])
            intervention([10.0, 4.0, 4.0], 0) == [1.6, 4.0, 0.7] # (True.)
        """
        # The case where indexs and constants are floats or ints
        if isinstance(intervened_idxs, int) and isinstance(constants, (int, float)):
            i = intervened_idxs
            c = float(constants)
            intervened_idxs = [intervened_idxs]
            constants = [c]

        self.intervened_idxs = intervened_idxs
        self.constants = constants

    def __call__(self, x: np.ndarray, t: float):
        """A perfect intervention on multiple variables.

        Args:
            x (ndarray): In the context of this package, x represents
                the current state of a dynamic model.
            t: (ndarray): In the context of this package, t represents
                the current time in a dynamic model.
        
        Returns:
            x_do (ndarray): In the context of this package, x_do represents
                the state of the dynamic model after the intervention is applied.
        """
        x_do = x.copy()
        for i, c in zip(self.intervened_idxs, self.constants):
            x_do[..., i] = c
        return x_do
    

def perfect_intervention(
    idxs: Union[int, Iterable[int]],
    constants: Union[float, Iterable[float]]
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Creates a perfect intervention function.

    A perfect intervention replaces variables with constant values.
    This function generates intervention functions that replace

    Args:
        idxs (int or collection of ints): The indexes where the intervention
            will be applied.
        constants (float or collection of floats): The values that the variables
            at corresponding to each index will be pinned to.

    Returns:
        intervention: A function

    Examples:
        intervention = perfect_intervention(0, 1.6)
        intervention([10.0, 4.0, 4.0], 0) == [1.6, 4, 4] # (True.)

        intervention = perfect_intervention([0, 2], [1.6, .7])
        intervention([10.0, 4.0, 4.0], 0) == [1.6, 4.0, 0.7] # (True.)
    """
    # The case where indexs and constants are floats or ints
    if isinstance(idxs, int) and isinstance(constants, (int, float)):
        i = idxs
        c = float(constants)
        # Make the intervention function.
        return perfect_intervention([i], [c])


    def intervention(x: np.array, t: float) -> np.array:
        """A perfect intervention on multiple variables.

        Args:
            x (ndarray): In the context of this package, x represents
                the current state of a dynamic model.
            t: (ndarray): In the context of this package, t represents
                the current time in a dynamic model.
        
        Returns:
            x_do (ndarray): In the context of this package, x_do represents
                the state of the dynamic model after the intervention is applied.
        """
        x_do = x.copy()
        for i, c in zip(idxs, constants):
            x_do[i] = c
        return x_do
    
    return intervention
        
def signal_intervention(
        idx: int,
        u: Callable[[float], float]
    ) -> Callable[[np.ndarray, float], np.ndarray]:
    """Creates an intervention function that replaces the variable a signal.

    Args:
        idx (int): The index where the intervention will be applied.
        u (callable): A function that accepts the current time and returns
            the value that should be assigned to the variable at idx.

    Returns:
        intervention (callable): Maps a numpy array and a time value to
            a new numpy array.

    Examples:
        
        x = np.array([1.1, 2, -1.2])

        g = interfere.signal_intervention(1, np.sin)
        np.allclose(g(x, 0), np.array([1.1, 0.0, -1.2]))
        np.allclose(g(x, np.pi/2), np.array([1.1, 1.0, -1.2]))

        g = interfere.signal_intervention(2, lambda t: t ** 2)
        np.allclose(g(x, 1.0), np.array([1.1, 2.0, 1.0]))
        np.allclose(g(x, -2.0), np.array([1.1, 2.0, 4.0]))
    """
    intervention = lambda x, t: x + np.array(
        [
            u(t) - x[i] if i == idx else 0.0 
            for i in range(len(x))
        ]
    )
    return intervention