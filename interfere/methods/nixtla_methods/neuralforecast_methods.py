"""Collection of predictive methods from neuralforecast."""
from typing import Any, Dict, List, Optional

import neuralforecast
from neuralforecast.losses.pytorch import MAE, MSE
import numpy as np

from ..base import BaseInferenceMethod
from .nixtla_adapter import NixtlaAdapter, default_exog_names
from ...utils import copy_doc

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
        num_workers_loader=0,
        drop_last_loader=False,
        optimizer=None,
        optimizer_kwargs=None,
        **trainer_kwargs
    ):
        # Initialize model
        self.method_params = locals()
        self.method_params.pop("self")
        self.method_params.pop("trainer_kwargs")
        self.method_params = {
            **self.method_params,
            **trainer_kwargs,
            # The following two arguments prevent pytorch lightning from writing
            # to files in order to enable parallel grid search.
            "enable_checkpointing": False,
            "logger": False
        }
        self.method_type = neuralforecast.models.LSTM
        self.nixtla_forecaster_class = neuralforecast.NeuralForecast


    @copy_doc(BaseInferenceMethod._fit)
    def _fit(
        self,
        endog_states: np.ndarray,
        t: np.ndarray,
        exog_states: np.ndarray = None
    ):
        
        if exog_states is not None:
            _, k = exog_states.shape
        else:
            k = 0
    
        # Assign names to internal exogeneous variables.
        self.set_params(futr_exog_list=default_exog_names(k))

        super()._fit(endog_states, t, exog_states)


    def get_window_size(self):
        return self.method_params["context_size"]
    

    def get_horizon(self):
        return self.model.h


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
            max_steps=50,
        )
    

    def get_test_param_grid() -> Dict[str, List[Any]]:
        return dict(
            encoder_hidden_size = [8, 16],
            learning_rate = [1e-12, 0.001]
        )