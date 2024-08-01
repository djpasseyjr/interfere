from typing import Any, Dict, Optional
from matplotlib.pylab import RandomState
import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.ltsf import LTSFLinearForecaster as sktime_LTSFLinearForecaster

from interfere.base import DEFAULT_RANGE

from .base import BaseInferenceMethod
from ..interventions import ExogIntervention
from ..utils import copy_doc, to_sktime_time_series


class LTSFLinearForecaster(BaseInferenceMethod):
    """Uses a transformer for inference."""


    @copy_doc(sktime_LTSFLinearForecaster.__init__)
    def __init__(
        self,
        seq_len: int = 1,
        pred_len: int = 1,
        num_epochs: int = 16,
        batch_size: int = 8,
        in_channels: int = 1,
        individual: bool = False,
        criterion: Optional[Any] = None,
        criterion_kwargs: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        optimizer_kwargs: Optional[Any] = None,
        lr: float = 0.001,
        custom_dataset_train: Optional[Any] = None,
        custom_dataset_pred: Optional[Any] = None
    ):
        self.method_params = locals()
        self.method_params.pop("self") 

        # Make forecasting horizon for fit and predict.  
        self.fh = [i for i in range(1, pred_len + 1)]


    @copy_doc(BaseInferenceMethod._fit)
    def _fit(
        self,
        endog_states: np.ndarray,
        t: np.ndarray,
        exog_states: np.ndarray = None
    ):
        self.model = sktime_LTSFLinearForecaster(
            **self.method_params
        )

        m = len(t)
        y = to_sktime_time_series(np.arange(m), endog_states)
        
        if endog_states is not None:
            X = to_sktime_time_series(np.arange(m), exog_states)
        else:
            X = None
        # This uses the sktime_LTSFLinearForecaster fit function.
        self.model.fit(y, X=X, fh=self.fh)


    @copy_doc(BaseInferenceMethod._predict)
    def _predict(
        self,
        forecast_times: np.ndarray,
        historic_endog: np.ndarray,
        historic_times: np.ndarray,
        exog: np.ndarray = None,
        historic_exog: np.ndarray = None,
        rng: np.random.RandomState = DEFAULT_RANGE
    ) -> np.ndarray:
        
        X = None
        if exog is not None:
            X = to_sktime_time_series(forecast_times, exog)

        # Save original stored endogeneous.
        _y_orig = self.model._y

        futr_endo = []

        for i in range(len(forecast_times)):
            y_next = self.model.predict(X=X, fh=self.fh).iloc[0:1, :]
            self.model._y = pd.concat([self.model._y.iloc[1:, :], y_next])
            futr_endo.append(y_next)

        # Reset stored endogenous to original state.
        self.model._y = _y_orig

        # Extract values and return.
        pred_endo = pd.concat(futr_endo).values        
        return pred_endo


    @copy_doc(BaseInferenceMethod.get_window_size)
    def get_window_size(self):
        return max(2, self.method_params["seq_len"])
    

    @copy_doc(BaseInferenceMethod.set_params)
    def set_params(self, **params):
        self.method_params = {**self.method_params, **params}
        self.model = sktime_LTSFLinearForecaster(**self.method_params)


    @copy_doc(BaseInferenceMethod.get_params)
    def get_params(self, deep: bool = True) -> Dict:
        return self.method_params
    

    @copy_doc(BaseInferenceMethod.get_test_params)
    def get_test_params() -> Dict[str, Any]:
        return sktime_LTSFLinearForecaster.get_test_params()[0]
    

    @copy_doc(BaseInferenceMethod.get_test_param_grid)
    def get_test_param_grid():
        return {
            "seq_len": [2, 5],
            "lr": [0.001, 0.1],
        }