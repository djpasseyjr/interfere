import numpy as np
import pandas as pd
from sktime.forecasting.ltsf import LTSFLinearForecaster as _LTSFLinearForecaster

from ..interventions import ExogIntervention

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
    

    def counterfactual_forecast(
        self,
        X: np.ndarray,
        time_points: np.ndarray,
        forecast_horizon: np.ndarray,
        intervention: ExogIntervention
    ):
        """Makes a forecast in the prescence of an intervention.

        Args:
            X (np.array): Historic data.
            time_points (np.ndarray): Historic time points
            forecast_horizon (np.ndarray): Time points to forecast.
            intervention (ExogIntervention): The intervention to apply at each
                future time point.
        """
        X_endog, X_exog = intervention.split_exogeneous(X)

        # Fit self to the data.
        super().fit(X_endog, fh=np.arange(len(forecast_horizon)), X=X_exog)

        # Generate the exogeneous signal.
        X_do_exog = intervention.eval_at_times(forecast_horizon)

        # Predict the future states of the system.
        pred_X_do_endo = self.predict(X=X_do_exog)

        # Recombine exogeneous and endogenous.
        pred_X_do = intervention.combine_exogeneous(pred_X_do_endo, X_do_exog)

        return pred_X_do    