from typing import Callable, Iterable, List, Tuple, Union
from typing_extensions import TypeAlias

import numpy as np

from .base import ExogIntervention

ScalarFunction: TypeAlias = Callable[[float], float]



class IdentityIntervention(ExogIntervention):
    """The trivial intervention. An intervention that does nothing."""

    def __init__(self) -> None:
        super().__init__([])
    
    def eval_at_times(self, t: np.ndarray):
        return None

    def __call__(self, x: np.array, t: float):
        return x
    
    def __eq__(self, other):
        return isinstance(other, IdentityIntervention)
    

class PerfectIntervention(ExogIntervention):

    def __init__(
        self,
        iv_idxs: Union[int, Iterable[int]],
        constants: Union[float, Iterable[float]]
    ):
        """Creates a perfect intervention function.

        A perfect intervention replaces variables with constant values.
        This function generates intervention functions that replace

        Args:
            iv_idxs (int or collection of ints): The indexes where the intervention
                will be applied.
            constants (float or collection of floats): The values that the variables
                at corresponding to each index will be pinned to.

        Examples:
            intervention = PerfectIntervention(0, 1.6)
            intervention([10.0, 4.0, 4.0], 0) == [1.6, 4, 4] # (True.)

            intervention = PerfectIntervention([0, 2], [1.6, .7])
            intervention([10.0, 4.0, 4.0], 0) == [1.6, 4.0, 0.7] # (True.)
        """
        # The case where indexs and constants are floats or ints
        if isinstance(iv_idxs, (int, float)):
            iv_idxs = [int(iv_idxs)]
            
        if isinstance(constants, (int, float)):
            constants = [float(constants)]

        if len(constants) != len(iv_idxs):
            raise ValueError(
                "Intervened indexes must be same length as provided constants")

        self.iv_idxs = iv_idxs
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
        for i, c in zip(self.iv_idxs, self.constants):
            x_do[..., i] = c
        return x_do
    
    def __eq__(self, other):
        """Determine if two perfect interventions are equal."""
        if not isinstance(other, PerfectIntervention):
            return False
        equal = True
        equal = equal and np.all(self.iv_idxs == other.iv_idxs)
        equal = equal and np.all(self.constants == other.constants)
        return equal
        
    
class SignalIntervention(ExogIntervention):

    def __init__(
        self,
        iv_idxs: Union[int, Iterable[int]],
        signals: Union[
            ScalarFunction,
            Iterable[ScalarFunction],
            Tuple[np.ndarray, np.ndarray],
            Iterable[Tuple[np.ndarray, np.ndarray]]
        ]
    ):
        """Creates an intervention that applies passed one arg functions.

            A perfect intervention replaces variables with constant values.
            This function generates intervention functions that replace

            Args:
                iv_idxs (int or collection of ints): The indexes where the intervention
                    will be applied.
                signals (Scalar function or collection of scalar functions): 
                    The functions that will replace the value of variables at
                    `iv_idxs` at each time point.
        """
        if isinstance(iv_idxs, int):

            i = iv_idxs
            s = signals
            iv_idxs = [iv_idxs]
            signals = [s]

        if len(signals) != len(iv_idxs):
            raise ValueError(
                "Number of intervened indexes must equal number of signals.")

        self.iv_idxs = iv_idxs
        self.signals = signals


    def __call__(self, x: np.ndarray, t: float):
        """A signal intervention on multiple variables.

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
        for i, s in zip(self.iv_idxs, self.signals):
            x_do[..., i] = s(t)
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