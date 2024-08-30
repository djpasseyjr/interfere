from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.var import VAR as skt_VAR

from .base import BaseInferenceMethod, DEFAULT_RANGE
from ..interventions import ExogIntervention
from ..utils import copy_doc, to_sktime_time_series


class VAR(BaseInferenceMethod):
    """Wrapper of sktime vector autoregression model.
    """

    @copy_doc(skt_VAR)
    def __init__(
        self,
        maxlags: Optional[Any] = 1,
        method: str = "ols",
        verbose: bool = False,
        trend: str = "c",
        missing: str = "none",
        dates: Optional[Any] = None,
        freq: Optional[Any] = None,
        ic: Optional[Any] = None,
        random_state: np.random.RandomState = DEFAULT_RANGE
    ):
        self.method_params = locals()
        self.method_params.pop("self")


    @copy_doc(BaseInferenceMethod._fit)
    def _fit(
        self,
        t: np.ndarray,
        endog_states: np.ndarray,
        exog_states: Optional[np.ndarray] = None,
    ):
        self.model = skt_VAR(**self.method_params)

        # Adjust tags to avoid errors on univariate series.
        self.model._tags["scitype:y"] = "both"

        y = to_sktime_time_series(t, endog_states)
        if endog_states is not None:
            X = to_sktime_time_series(t, exog_states)
        else:
            X = None
        # This uses the skt_VAR fit function.
        self.model.fit(y, X=X)
    

    @copy_doc(BaseInferenceMethod._predict)
    def _predict(
        self,
        t: np.ndarray,
        prior_endog_states: np.ndarray,
        prior_exog_states: Optional[np.ndarray] = None,
        prior_t: Optional[np.ndarray] = None,
        prediction_exog: Optional[np.ndarray] = None,
        rng: np.random.RandomState = DEFAULT_RANGE,
    ) -> np.ndarray:
        
        X = None
        if prediction_exog is not None:
            # Collect exogeneous into sktime format.
            X = to_sktime_time_series(t, prediction_exog)
        
        # Predict endog state evolution.
        endog_pred = self.model.predict(
            X=X,
            fh=ForecastingHorizon(
                [i for i in range(len(t))],
                is_relative=True,
                # Frequency is not used here since we drop the index below
                freq="s"
            )
        )
        return endog_pred.values
    

    @copy_doc(BaseInferenceMethod.get_window_size)
    def get_window_size(self):
        return max(2, self.method_params["maxlags"])
    

    @copy_doc(BaseInferenceMethod.set_params)
    def set_params(self, **params):
        self.method_params = {**self.method_params, **params}
        self.model = skt_VAR(**self.method_params)


    @copy_doc(BaseInferenceMethod.get_params)
    def get_params(self, deep: bool = True) -> Dict:
        return self.method_params
    

    @copy_doc(BaseInferenceMethod.get_test_param_grid)
    def get_test_param_grid() -> Dict[str, List[Any]]:
        return {
            "maxlags": [1, 2, 10],
            "trend" : ["ctt", "n"]
        }
    

    @copy_doc(BaseInferenceMethod.get_test_params)
    def get_test_params() -> Dict[str, Any]:
        return {}