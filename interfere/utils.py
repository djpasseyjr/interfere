"""Useful extra functions."""

from typing import Any, Callable, Iterable, TypeVar
from typing_extensions import ParamSpec, TypeAlias

import numpy as np
import pandas as pd


T = TypeVar('T')
P = ParamSpec('P')
WrappedFuncDeco: TypeAlias = Callable[[Callable[P, T]], Callable[P, T]]


def copy_doc(copy_func: Callable[..., Any]) -> WrappedFuncDeco[P, T]:
    """Copies the doc string of the given function to another. 
    This function is intended to be used as a decorator.

    .. code-block:: python3

        def foo():
            '''This is a foo doc string'''
            ...

        @copy_doc(foo)
        def bar():
            ...
    """

    def wrapped(func: Callable[P, T]) -> Callable[P, T]:
        func.__doc__ = copy_func.__doc__
        return func

    return wrapped


def to_interfere_time_series(y: pd.DataFrame):
    """Converts time series DataFrames from the sktime to interfere format.
    
    Args:
        y (Iterable[pd.DataFrame]): An iterable containing DataFrames with
            datetime indexes

    Returns:
        t (np.ndarray): A length `m` 1D array containing the time points from
            the index. 
        X (np.ndarray): A 2D (m x n) array containing the time series. Rows are
            observations and columns are variables. Rows correspond to values in
            t.
    """
    t = np.squeeze(IxToX().fit_transform(y)).values.astype("float")
    # Convert to seconds.
    t /= 1_000_000_000.0

    X = y.values
    return t, X