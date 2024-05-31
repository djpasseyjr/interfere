from typing import Any, Dict, List, Optional

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MSE
from neuralforecast.models import LSTM as neuralforecast_LSTM
import numpy as np
import pandas as pd

from .base import BaseInferenceMethod
from ..base import DEFAULT_RANGE
from ..utils import copy_doc


class NeuralForecastAdapter(BaseInferenceMethod):

    def __init__(self, *args, **kwargs):
        self.method_args = args
        self.method_kwargs = kwargs


    @copy_doc(BaseInferenceMethod._fit)
    def _fit(
        self,
        endog_states: np.ndarray,
        t: np.ndarray,
        exog_states: np.ndarray = None
    ):
        
        _, n = endog_states.shape
        _, k = exog_states.shape
    
        # Make names for exogeneous variables.
        exog_names = ["ex" + str(i) for i in range(k)]

        # Initialize model
        self.model = self.model_type(
            *self.method_args,
            **self.method_kwargs,
            futr_exog_list=exog_names
        )



        # Map each endog state to a unique ID and associate with all exogeneous.
        unique_id_exog = [
            np.vstack([
                t, i * np.ones(len(t), int), endog_states[:, i], exog_states.T
            ]).T
            for i in range(n)
        ]

        # Stack each endog block vertically and make dataframe w appropriate
        # column names.
        train_df = pd.DataFrame(
            np.vstack(unique_id_exog),
            columns=["ds", "unique_id", "y"] + exog_names
        )

        # Translate to datetime.
        train_df.ds = pd.to_datetime(
            train_df.ds, unit='s', errors='coerce')
        

        self.neural_forecaster = NeuralForecast(
            models=[self.model],
            freq="s"
        )
        self.neural_forecaster.fit(df=train_df)


    @copy_doc(BaseInferenceMethod._predict)
    def _predict(
        self,
        forecast_times: np.ndarray,
        historic_endog: np.ndarray,
        historic_times: np.ndarray,
        exog: Optional[np.ndarray] = None,
        historic_exog: Optional[np.ndarray] = None,
        rng: np.random.RandomState = DEFAULT_RANGE
    ) -> np.ndarray:
        _, n = historic_endog.shape
        _, k = exog.shape

        # Associate forecast times with each unique ID.
        fcast_times_and_unique_id = [
            np.vstack([
                forecast_times, i * np.ones(len(forecast_times), int)
            ]).T
            for i in range(n)
        ]

        if exog is not None:
            fcast_times_and_unique_id = [
                np.hstack([Y, exog]) for Y in fcast_times_and_unique_id
            ]

        # Stack each endog block vertically and make dataframe w appropriate
        # column names.
        columns = ["ds", "unique_id",] + [
            "ex" + str(i) for i in range(k) if exog is not None
        ]

        exog_df = pd.DataFrame(
            np.vstack(fcast_times_and_unique_id),
            columns=columns
        )
        # Translate to datetime.
        exog_df.ds = pd.to_datetime(exog_df.ds, unit='s', errors='coerce')

        forecast_df = self.neural_forecaster.predict(futr_df=exog_df)
        max_unique_id = forecast_df.index.max()

        endog_pred = np.vstack([
            forecast_df[forecast_df.index == i].iloc[:, 1]
            for i in range(max_unique_id + 1)
        ]).T

        return endog_pred


class LSTM(NeuralForecastAdapter):

    model_type = neuralforecast_LSTM

    def get_window_size(self):
        return self.method_kwargs["context_size"]


    def get_test_params() -> Dict[str, Any]:
        return dict(
            h=12, 
            input_size=-1,
            loss=MSE(),
            scaler_type='robust',
            encoder_n_layers=2,
            encoder_hidden_size=64,
            context_size=10,
            decoder_hidden_size=64,
            decoder_layers=2,
            max_steps=200,
        )
    

    def get_test_param_grid() -> Dict[str, List[Any]]:
        return dict(
            encoder_n_layers = [1, 2],
            encoder_hidden_size = [16, 64]
        )