"""Useful extra functions."""

from typing import Any, Callable, Iterable, TypeVar
from typing_extensions import ParamSpec, TypeAlias

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.transformations.compose import IxToX


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


def to_sktime_time_series(time_points: np.ndarray, X: np.ndarray):
    """Converts time series from interfere format to sktime format.
    
    Args:
        time_points (np.ndarray): A 1D array of time points that correspond to
            the rows of of all arrays in Xs.
        Xs Iterable[np.ndarray]: An iterable containing 2D (m x n_i) array
            where rows are observations and columns are variables. The number of
            rows, `m` must equal the length of `time_points`.

    Returns:
        y (pd.DataFrame): A DataFrame containing the endogenous variables.
            Columns are variables and rows are observations.
        X (pd.DataFrame): A DataFrame containing the exogenoug variables.
            Columns are variables and rows are observations.

    """
    index = pd.to_datetime(time_points, unit='s', errors='coerce')
    sktime_X = pd.DataFrame(X, index=index)
    return sktime_X


def to_forecasting_horizon(forecast_times: np.ndarray):
    """Converts an interfere array of time points to a sktime ForcastingHorizon.

    Args:
        forecast_times (np.ndarray): A 1D array of time points.

    Returns:
        fh (ForecastingHorizon): A sktime forecasting horizon
    """
    return ForecastingHorizon(
        pd.to_datetime(forecast_times, unit="s", errors="coerce"), is_relative=False
    )


def fh_to_seconds(fh: ForecastingHorizon, end_time=None):
    """Transforms forecasting horizon to seconds."""
    t = np.squeeze(
            IxToX().fit_transform(
                pd.DataFrame(index=fh.to_pandas())
            )
    ).values.astype(float)

    # Convert to seconds.
    t /= 1_000_000_000.0

    # Check if forecasting horizon is relative to the end time and adjust.
    if fh.is_relative:
        if end_time is None:
            raise ValueError("ForecastingHorizon is relative to some end time but `end_time` argument is None.")
        t += end_time

    return t


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