from typing import Callable, Iterable, List, Union
from typing_extensions import TypeAlias

import numpy as np

from .base import Intervention

ScalarFunction: TypeAlias = Callable[[float], float]

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
        exog_X = X[..., self.intervened_idxs]
        endo_X = np.delete(X, self.intervened_idxs, axis=-1)
        return endo_X, exog_X
    
    def combine_exogeneous(self, endo_X: np.ndarray, exog_X: np.ndarray):
        """Recombines endogenous and endogenous signals."""
        X = np.insert(endo_X, self.intervened_idxs, exog_X, axis=1)
        return X
    
    def eval_at_times(self, t: np.ndarray):
        """Produces exogeneous signals only at the time values in t.

        Args:
            t (np.ndarray): A 1D array of time points

        Returns:
            exog_X_do (np.ndarray): An (m, ) or (m, p) array where `m = len(t)` 
                and `p = len(self.intervened_idx)`. Contains the values of the
                exogenous signal at the time points in `t`. The order of the
                columns corresponds to the order of indexes in 
                `self.intervened_idx`.
        """
        # Exogeneous interventions do not depend on x so we can use a dummy var.
        dummy_x = np.zeros(np.max(self.intervened_idxs) + 1)
        # The size of the dummy variable only needs to be as big as the max
        # intervened index and self.split_exogeneous will still work.
        # This is done to avoid the need for an additional argument containing
        # the dimension of the system.

        X_do = np.vstack([
            self.__call__(dummy_x, ti) for ti in t
        ])
        _, exog_X_do = self.split_exogeneous(X_do)
        return exog_X_do

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
        raise NotImplementedError()

    

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

        Examples:
            intervention = PerfectIntervention(0, 1.6)
            intervention([10.0, 4.0, 4.0], 0) == [1.6, 4, 4] # (True.)

            intervention = PerfectIntervention([0, 2], [1.6, .7])
            intervention([10.0, 4.0, 4.0], 0) == [1.6, 4.0, 0.7] # (True.)
        """
        # The case where indexs and constants are floats or ints
        if isinstance(intervened_idxs, int) and isinstance(constants, (int, float)):
            i = intervened_idxs
            c = float(constants)
            intervened_idxs = [intervened_idxs]
            constants = [c]

        if len(constants) != len(intervened_idxs):
            raise ValueError(
                "Intervened indexes must be same length as provided constants")

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
    
class SignalIntervention(ExogIntervention):

    def __init__(
        self,
        intervened_idxs: Union[int, Iterable[int]],
        signals: Union[ScalarFunction, Iterable[ScalarFunction]]
    ):
        """Creates an intervention that applies passed one arg functions.

            A perfect intervention replaces variables with constant values.
            This function generates intervention functions that replace

            Args:
                intervened_idxs (int or collection of ints): The indexes where the intervention
                    will be applied.
                signals (float or collection of floats): The functions that will
                    replace the value of variables at `intervened_idxs` at each
                    time point.
        """
        if isinstance(intervened_idxs, int):

            i = intervened_idxs
            s = signals
            intervened_idxs = [intervened_idxs]
            signals = [s]

        if len(signals) != len(intervened_idxs):
            raise ValueError(
                "Number of intervened indexes must equal number of signals.")
        

        self.intervened_idxs = intervened_idxs
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
        for i, s in zip(self.intervened_idxs, self.signals):
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