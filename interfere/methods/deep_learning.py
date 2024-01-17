import numpy as np
import pandas as pd
from sktime.forecasting.ltsf import LTSFLinearForecaster as _LTSFLinearForecaster


class LTSFLinearForecaster(_LTSFLinearForecaster):
    """Uses a transformer for inference."""

    def fit(self, X):
        return super().fit(X, fh=[1])
    
    def simulate_counterfactual(self, X, time_points, intervention):

        # Fit self to the data.
        self.fit(X)
        # Save internal state while we overwrite it to run the simulation.
        _y_copy = self._y.copy()
        lag = self.network.seq_len
        pred_X_do = np.zeros_like(X)
        pred_X_do[:lag, :] = intervention(X[:lag, :], None)
        for i in range(len(time_points) - lag):
            # Set internal history equal to the simulated past
            self._y = pd.DataFrame(pred_X_do[i:i+lag, :], index=range(i, i+lag))
            # Predict the next state and apply intervention
            pred_X_do[i+1] = intervention(self.predict(fh=[1]), None)

        return pred_X_do
        