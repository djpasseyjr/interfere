import logging
from typing import Optional


import neuralforecast.models
from neuralforecast.losses.pytorch import BasePointLoss, MAE
import numpy as np

from ....base import ForecastMethod
from ..nixtla_adapter import NixtlaAdapter, default_exog_names
from ....utils import copy_doc


class NHITS(NixtlaAdapter):
    

    @copy_doc(neuralforecast.models.NHITS)
    def __init__(
        self,
        h: int = 1,
        input_size: int = 2,
        futr_exog_list: Optional[list]= None,
        hist_exog_list: Optional[list] = None,
        stat_exog_list: Optional[list] = None,
        exclude_insample_y: bool = False,
        stack_types: list = ["identity", "identity", "identity"],
        n_blocks: list = [1, 1, 1],
        mlp_units: list = 3 * [[512, 512]],
        n_pool_kernel_size: list = [2, 2, 1],
        n_freq_downsample: list = [4, 2, 1],
        pooling_mode: str = "MaxPool1d",
        interpolation_mode: str = "linear",
        dropout_prob_theta: float = 0.0,
        activation: str = "ReLU",
        loss: BasePointLoss = MAE(),
        valid_loss: BasePointLoss = None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = 3,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = -1,
        start_padding_enabled: bool = False,
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        drop_last_loader: bool = False,
        optimizer=None,
        optimizer_kwargs=None,
        **trainer_kwargs,
    ):
        # Initialize model
        method_params = locals()
        method_params.pop("self")
        method_params.pop("trainer_kwargs")
        method_params.pop("__class__", None)
        method_params = {
            **method_params,
            **trainer_kwargs,
        }

        method_type = neuralforecast.models.NHITS
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
        return self.method_params["input_size"]
    

    def get_horizon(self):
        return self.model.h
    

    def get_test_params():
        return {"max_steps": 50}


    @staticmethod
    @copy_doc(ForecastMethod._get_optuna_params)
    def _get_optuna_params(trial, max_lags=50, **kwargs):
        return {

            "h": trial.suggest_int("h", 1, 16),

            "input_size": trial.suggest_int("input_size", 1, max_lags),

            "n_pool_kernel_size": trial.suggest_categorical(
                "n_pool_kernel_size",
                [
                    [2, 2, 1],
                    3 * [1],
                    3 * [2],
                    3 * [4],
                    [8, 4, 1],
                    [16, 8, 1]
                ]
            ),

            "n_freq_downsample": trial.suggest_categorical(
                "n_freq_downsample",
                [
                    [168, 24, 1],
                    [24, 12, 1],
                    [180, 60, 1],
                    [60, 8, 1],
                    [40, 20, 1],
                    [1, 1, 1],
                ]
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 1e-1, log=True),

            "scaler_type": trial.suggest_categorical(
                "scaler_type", [None, "robust", "standard"]),

            "max_steps": trial.suggest_float(
                "max_steps", 500, 1500, step=100),
            
            "batch_size": trial.suggest_categorical(
                "batch_size", [32, 64, 128, 256]),
            
            "windows_batch_size": trial.suggest_categorical(
                "windows_batch_size", [128, 256, 512, 1024]),
            
            "random_seed": trial.suggest_int("random_seed", 1, 20),
        }