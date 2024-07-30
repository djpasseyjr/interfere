from typing import Any, List, Dict, Optional

import statsforecast.models

from .nixtla_adapter import NixtlaAdapter
from ...utils import copy_doc


class AutoARIMA(NixtlaAdapter): 

    @copy_doc(statsforecast.models.AutoARIMA)
    def __init__(
        self,
        d: Optional[int] = None,
        D: Optional[int] = None,
        max_p: int = 5,
        max_q: int = 5,
        max_P: int = 2,
        max_Q: int = 2,
        max_order: int = 5,
        max_d: int = 2,
        max_D: int = 1,
        start_p: int = 2,
        start_q: int = 2,
        start_P: int = 1,
        start_Q: int = 1,
        stationary: bool = False,
        seasonal: bool = True,
        ic: str = "aicc",
        stepwise: bool = True,
        nmodels: int = 94,
        trace: bool = False,
        approximation: Optional[bool] = False,
        method: Optional[str] = None,
        truncate: Optional[bool] = None,
        test: str = "kpss",
        test_kwargs: Optional[str] = None,
        seasonal_test: str = "seas",
        seasonal_test_kwargs: Optional[Dict] = None,
        allowdrift: bool = False,
        allowmean: bool = False,
        blambda: Optional[float] = None,
        biasadj: bool = False,
        season_length: int = 1,
        alias: str = "AutoARIMA",
        prediction_intervals: Optional[statsforecast.utils.ConformalIntervals] = None,
    ):
        self.method_params = locals()
        self.method_params.pop("self")
        self.method_type = statsforecast.models.AutoARIMA
        self.nixtla_forecaster_class = statsforecast.StatsForecast

    def get_window_size(self):
        """Returns how many historic time steps are needed to make a
        prediction."""

        # if hasattr(self, "nixtla_forecaster"):

        #     max_lag = max([
        #         x 
        #         for arma in self.nixtla_forecaster.fitted_[:, 0] 
        #         for x in arma.model_["arma"]
        #     ])
        #     return max_lag

        # else:
        return max(self.method_params["max_p"], self.method_params["max_P"], 2)
        

    def get_horizon(self):
        """Returns the minimum timesteps the method will predict."""
        # Model predicts minimum of one timestep.
        return 1


    def get_test_params() -> Dict[str, Any]:
        """Returns default parameters conducive to fast test cases"""
        return dict(
            nmodels = 10,
        )
    

    def get_test_param_grid() -> Dict[str, List[Any]]:
        """Returns a parameter grid for testing grid search"""
        return {
            "max_p": [10, 5, 1],
            "max_q": [3, 1]
        }