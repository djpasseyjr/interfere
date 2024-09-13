from typing import Optional

import neuralforecast.models
from neuralforecast.losses.pytorch import MAE
import numpy as np

from ...base import BaseInferenceMethod
from ..nixtla_adapter import NixtlaAdapter
from ....utils import copy_doc


class NHITS(NixtlaAdapter):
    

    @copy_doc(neuralforecast.models.NHITS)
    def __init__(
        self,
        h: int = 1,
        input_size: int = -1,
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
        dropout_prob_theta=0.0,
        activation="ReLU",
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = 3,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = -1,
        start_padding_enabled=False,
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        num_workers_loader=0,
        drop_last_loader=False,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        **trainer_kwargs,
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

        self.method_type = neuralforecast.models.NHITS
        self.nixtla_forecaster_class = neuralforecast.NeuralForecast


    @copy_doc(BaseInferenceMethod._fit)
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


    def _get_optuna_params(trial):
        return {
            "input_size_multiplier": [1, 2, 3, 4, 5],

            "h": trial.suggest_int("h", 1, 16),

            
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
                "max_steps", lower=500, upper=1500, step=100),
            
            "batch_size": trial.suggest_categorical(
                "batch_size", [32, 64, 128, 256]),
            
            "windows_batch_size": trial.suggest_categorical(
                "windows_batch_size", [128, 256, 512, 1024]),
            
            "random_seed": trial.suggest_int("random_seed", 1, 20),
        }