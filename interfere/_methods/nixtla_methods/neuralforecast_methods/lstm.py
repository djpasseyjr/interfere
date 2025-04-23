"""Collection of predictive methods from neuralforecast."""
from typing import Any, Dict, List, Optional

import neuralforecast
from neuralforecast.losses.pytorch import MAE
import neuralforecast.models
import numpy as np

from ....base import ForecastMethod
from ..nixtla_adapter import NixtlaAdapter, default_exog_names
from ....utils import copy_doc


class LSTM(NixtlaAdapter):


    @copy_doc(neuralforecast.models.LSTM)
    def __init__(
        self,
        h: int = 1,
        input_size: int = -1,
        inference_input_size: int = -1,
        encoder_n_layers: int = 2,
        encoder_hidden_size: int = 200,
        encoder_bias: bool = True,
        encoder_dropout: float = 0.0,
        context_size: int = 10,
        decoder_hidden_size: int = 200,
        decoder_layers: int = 2,
        futr_exog_list=None,
        hist_exog_list=None,
        stat_exog_list=None,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size=32,
        valid_batch_size: Optional[int] = None,
        scaler_type: str = "robust",
        random_seed=1,
        drop_last_loader=False,
        optimizer=None,
        optimizer_kwargs=None,
        **trainer_kwargs
    ):
        # Initialize model
        method_params = locals()
        method_params.pop("self")
        method_params.pop("__class__", None)
        method_params.pop("trainer_kwargs")
        method_params = {
            **method_params,
            **trainer_kwargs,
        }
        method_type = neuralforecast.models.LSTM
        nixtla_forecaster_class = neuralforecast.NeuralForecast

        super().__init__(method_type, method_params, nixtla_forecaster_class)

    @copy_doc(ForecastMethod._fit)
    def _fit(
        self,
        t: np.ndarray,
        endog_states: np.ndarray,
        exog_states: Optional[np.ndarray] = None
    ):
        
        if exog_states is not None:
            _, k = exog_states.shape
        else:
            k = 0
    
        # Assign names to internal exogeneous variables.
        self.set_params(futr_exog_list=default_exog_names(k))

        super()._fit(t, endog_states, exog_states)


    def get_window_size(self):
        return self.method_params["context_size"]
    

    def get_horizon(self):
        return self.model.h


    def get_test_params() -> Dict[str, Any]:
        return dict(
            h=1, 
            loss=MAE(),
            scaler_type='robust',
            encoder_n_layers=2,
            encoder_hidden_size=64,
            context_size=4,
            decoder_hidden_size=64,
            decoder_layers=2,
            max_steps=50,
        )
    

    @staticmethod
    @copy_doc(ForecastMethod._get_optuna_params)
    def _get_optuna_params(trial, max_lags=50, **kwargs):
        return {
            "h": trial.suggest_int("h", 1, 16),

            "encoder_hidden_size": trial.suggest_categorical(
                "encoder_hidden_size", [50, 100, 200, 300]),

            "encoder_n_layers": trial.suggest_int("encoder_n_layers", 1, 4),

            "context_size": trial.suggest_int("context_size", 1, max_lags),

            "decoder_hidden_size": trial.suggest_categorical(
                "decoder_hidden_size", [64, 128, 256, 512]),

            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 1e-1, log=True),

            "max_steps": trial.suggest_categorical("max_steps", [500, 1000]),

            "batch_size": trial.suggest_categorical("batch_size", [16, 32]),

            "random_seed": trial.suggest_int("random_seed", 1, 20),
        }