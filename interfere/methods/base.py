from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator

from ..interventions import ExogIntervention


class CounterfacualInferenceMethod(BaseEstimator):


    def counterfactual_forecast(
        self,
        X_historic: np.ndarray,
        historic_times: np.ndarray,
        forecast_times: np.ndarray,
        intervention: ExogIntervention
    ):
        """Makes a forecast in the prescence of an intervention.

        Args:
            X_historic (np.array): Historic data.
            historic_times (np.ndarray): Historic time points
            forecast_times (np.ndarray): Time points to forecast.
            intervention (ExogIntervention): The intervention to apply at each
                future time point.
        """
        raise NotImplementedError()