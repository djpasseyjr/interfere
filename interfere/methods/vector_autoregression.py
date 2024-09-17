from typing import Any, Dict, List, Optional

import numpy as np
from statsmodels.tsa.api import VAR as smVAR

from .base import BaseInferenceMethod, DEFAULT_RANGE
from ..utils import copy_doc


class VAR(BaseInferenceMethod):
    """Wrapper of statsmodels vector autoregression model.
    """

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
        self.results = smVAR(
            endog=endog_states,
            exog=exog_states,
            freq=self.method_params["freq"],
            missing=self.method_params["missing"]
        ).fit(
            maxlags=self.method_params["maxlags"],
            method=self.method_params["method"],
            ic=self.method_params["ic"],
            trend=self.method_params["trend"]
        )
    

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
        
        # We need to predict everything except the first time point (because it
        # corresponds to the initial condition).
        steps = len(t) - 1
        exog_future = None
        if prediction_exog is not None:
            exog_future = prediction_exog[1:, :]
        lags = self.results.k_ar
        y = prior_endog_states[-lags:, :]

        endog_pred = self.results.forecast(
            y=y,
            steps=steps,
            exog_future=exog_future
        )
        return np.vstack([prior_endog_states[-1, :], endog_pred])
    

    @copy_doc(BaseInferenceMethod.get_window_size)
    def get_window_size(self):
        return max(2, self.method_params["maxlags"])
    

    @copy_doc(BaseInferenceMethod.set_params)
    def set_params(self, **params):
        self.method_params = {**self.method_params, **params}

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
    

    def _get_optuna_params(trial):
        return {
            "ic": trial.suggest_categorical("ic", ["aic", "fpe"]),
            "maxlags": trial.suggest_int("maxlags", 1, 5),
            "trend" : trial.suggest_categorical(
                "trend", ["c", "ct", "ctt", "n"]),
        }