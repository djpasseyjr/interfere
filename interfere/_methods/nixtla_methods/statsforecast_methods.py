from typing import Any, List, Dict, Optional, Tuple

import statsforecast.models

from ...base import ForecastMethod
from .nixtla_adapter import NixtlaAdapter
from ...utils import copy_doc



class ARIMA(NixtlaAdapter): 

    @copy_doc(statsforecast.models.ARIMA)
    def __init__(
        self,
        order: Tuple[int, int, int] = (0, 0, 0),
        season_length: int = 1,
        seasonal_order: Tuple[int, int, int] = (0, 0, 0),
        include_mean: bool = True,
        include_drift: bool = False,
        include_constant: Optional[bool] = None,
        blambda: Optional[float] = None,
        biasadj: bool = False,
        method: str = "CSS-ML",
        fixed: Optional[dict] = None,
        alias: str = "ARIMA",
        prediction_intervals: Optional[statsforecast.utils.ConformalIntervals] = None,
        p = None,
        d = None,
        q = None,
    ):
        self.method_params = locals()
        self.method_params.pop("self")

        # Optionally accepts individual order parameters to optimize separately.

        # Time lags.
        p = self.method_params.pop("p", None)

        # Seasonal differencing.
        d = self.method_params.pop("d", None)

        # Autocorrelation lags.
        q = self.method_params.pop("q", None)

        if p:
            ord = self.method_params["order"]
            self.method_params["order"] = (p, ord[1], ord[2])
        if d:
            ord = self.method_params["order"]
            self.method_params["order"] = (ord[0], d, ord[2])
        if q:
            ord = self.method_params["order"]
            self.method_params["order"] = (ord[0], ord[1], q)

        # Denote method type and forecaster class.
        self.method_type = statsforecast.models.ARIMA
        self.nixtla_forecaster_class = statsforecast.StatsForecast

    def get_window_size(self):
        """Returns how many historic time steps are needed to make a
        prediction."""

        return max(self.method_params["order"][0], 2)
        

    def get_horizon(self):
        """Returns the minimum timesteps the method will predict."""
        # Model predicts minimum of one timestep.
        return 1
    

    def get_test_params() -> Dict[str, Any]:
        """Returns default parameters conducive to fast test cases"""
        return {}
    

    @staticmethod
    @copy_doc(ForecastMethod._get_optuna_params)
    def _get_optuna_params(trial, max_lags=15, **kwargs) -> Dict[str, Any]:
        """Returns a parameter grid for testing grid search"""
        return {
            "order": (
                trial.suggest_int("p",1, max_lags),
                trial.suggest_int("d", 1, max_lags), # Seasonal differencing.
                trial.suggest_int("q",1, 15)
            ),
            "include_mean": trial.suggest_categorical(
                "include_mean", [True, False]),
            "include_drift": trial.suggest_categorical(
                "include_drift", [True, False]),
            "include_constant": trial.suggest_categorical(
                "include_constant", [True, False]),
        }