"""This submodule exists to speed up import time for the interfere package.

All actual methods are housed in _methods which imports nothing by default. 
This allows other submodules to access specific methods in _methods without, for example, having to import all of pytorch and the nvidia gpu packages. 

If a user wants to import all methods, they simply run 

`import interfere.methods` 

which takes 15-30 seconds as it needs to build the nvidia drivers and run all the pytorch installation scripts.
"""
from .._methods.nixtla_methods.statsforecast_methods import ARIMA
from .._methods.average_method import AverageMethod
from .._methods.nixtla_methods.neuralforecast_methods import LSTM, NHITS
from .._methods.reservoir_computer import ResComp
from .._methods.sindy import SINDy
from .._methods.vector_autoregression import VAR

