from typing import Any, List, Dict, Optional, Tuple

import statsforecast.models

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
    ):
        self.method_params = locals()
        self.method_params.pop("self")
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
    

    def _get_optuna_params(trial) -> Dict[str, List[Any]]:
        """Returns a parameter grid for testing grid search"""
        return {
            "order": (
                trial.suggest_int("p",1, 15),
                1, # Seasonal differencing.
                trial.suggest_int("q",1, 15)
            ),
            "include_mean": trial.suggest_categorical(
                "include_mean", [True, False]),
            "include_drift": trial.suggest_categorical(
                "include_drift", [True, False]),
            "include_constant": trial.suggest_categorical(
                "include_constant", [True, False]),
        }