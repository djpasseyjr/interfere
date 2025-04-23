from typing import Any, Dict, List, Optional

import numpy as np
from statsmodels.tsa.api import VAR as smVAR

from ..base import ForecastMethod, DEFAULT_RANGE
from ..utils import copy_doc


class VAR(ForecastMethod):
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


    @copy_doc(ForecastMethod._fit)
    def _fit(
        self,
        t: np.ndarray,
        endog_states: np.ndarray,
        exog_states: Optional[np.ndarray] = None,
    ):
        self.var = smVAR(
            endog=endog_states,
            exog=exog_states,
            freq=self.method_params["freq"],
            missing=self.method_params["missing"]
        )

        # Make sure that the model doesn't try to estimate too many lags.
        trend = self.method_params["trend"]
        ntrend = len(trend) if trend.startswith("c") else 0
        max_estimable_lags = (self.var.n_totobs - self.var.neqs - ntrend) // (1 + self.var.neqs)
        maxlags = min(self.method_params["maxlags"], max_estimable_lags)
        if maxlags < 0:
            raise ValueError(
                "Not enough data to estimate VAR model parameters for any "
                "number of lags."
                f"\n\t Trend equations: {ntrend}"
                f"\n\t AR equations: {self.var.neqs}"
                f"\n\t Observations: {self.var.n_totobs}"
            )
        
        self.results = self.var.fit(
            maxlags=maxlags,
            method=self.method_params["method"],
            ic=self.method_params["ic"],
            trend=self.method_params["trend"]
        )
    

    @copy_doc(ForecastMethod._predict)
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

        # When VAR determines that the best fit is a constant, predict constant # because there is a bug in the statsmodels forecaster.
        if lags == 0:
            consts = self.results.params[0]
            endog_pred = np.vstack([consts for _ in range(steps)])

        else:
            endog_pred = self.results.forecast(
                y=y,
                steps=steps,
                exog_future=exog_future
            )

        return np.vstack([prior_endog_states[-1, :], endog_pred])
    

    @copy_doc(ForecastMethod.get_window_size)
    def get_window_size(self):
        return max(2, self.method_params["maxlags"])
    

    @copy_doc(ForecastMethod.set_params)
    def set_params(self, **params):
        self.method_params = {**self.method_params, **params}

    @copy_doc(ForecastMethod.get_params)
    def get_params(self, deep: bool = True) -> Dict:
        return self.method_params
    

    def get_test_params() -> Dict[str, Any]:
        return {}
    

    @staticmethod
    @copy_doc(ForecastMethod._get_optuna_params)
    def _get_optuna_params(trial, max_lags=20, **kwargs):
        return {
            # Method-specific parameter suggestions
        }