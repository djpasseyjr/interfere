"""Useful extra functions."""

from typing import Callable, TypeVar, Any
from typing_extensions import ParamSpec, TypeAlias

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