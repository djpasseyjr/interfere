from typing import Any, Dict
from matplotlib.pylab import RandomState
import numpy as np
import pandas as pd
from sktime.forecasting.ltsf import LTSFLinearForecaster as sktime_LTSFLinearForecaster

from interfere.base import DEFAULT_RANGE

from .base import BaseInferenceMethod
from ..interventions import ExogIntervention
from ..utils import (
    copy_doc, to_sktime_time_series
)

class LTSFLinearForecaster(BaseInferenceMethod, sktime_LTSFLinearForecaster):
    """Uses a transformer for inference."""


    @copy_doc(BaseInferenceMethod._fit)
    def _fit(
        self,
        endog_states: np.ndarray,
        t: np.ndarray,
        exog_states: np.ndarray = None
    ):
        y = to_sktime_time_series(t, endog_states)
        if endog_states is not None:
            X = to_sktime_time_series(t, exog_states)
        else:
            X = None
        # This uses the sktime_LTSFLinearForecaster fit function.
        super(BaseInferenceMethod, self).fit(y, X=X, fh=[1])


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
        
        # Save internal state while we overwrite it to run the simulation.
        _y_copy = self._y.copy()
        (m,) = forecast_times
        _, n = historic_endog.shape

        # Make a window of previous observations.
        lag = self.network.seq_len
        prev_window = historic_endog[-lag:, :]

        # Empty array of predictions.
        endo_pred = np.zeros((m + lag, n))

        # Add historical obs to the beginning of preds to start recursive pred.
        endo_pred[:lag, :] = historic_endog[-lag:, :]

        for i in range(n):
            # Set internal history equal to the simulated past
            self._y = pd.DataFrame(
                endo_pred[i:(i + lag), :], index=range(i, i+lag))
            # Predict the next state
            endo_pred[i+1] = super(BaseInferenceMethod, self).predict(
                X=exog[i:i+1, :], fh=[1])

        # Drop historic obs from prediction and return.
        return endo_pred[lag:, :]

    
    def get_window_size(self):
        return self.seq_len
    

    def get_test_params() -> Dict[str, Any]:
        return super().get_test_params()[0]
    

    def get_test_param_grid():
        return {
            "seq_len": [2, 5],
            "lr": [0.001, 0.1],
        }