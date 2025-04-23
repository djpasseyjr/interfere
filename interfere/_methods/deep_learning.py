from typing import Any, Optional
import numpy as np
import pandas as pd

try:
    from sktime.forecasting.ltsf import LTSFLinearForecaster as sktime_LTSFLinearForecaster
    
except ImportError as e:
    raise ImportError(
        "ImportError occured in sktime import."
        "\n\n This likely occurred because `interfere` does not list `sktime` as a direct dependency. To use the "
        "LTSF, try first install `sktime` via pip "
        "install sktime."
        f"\n\nOriginating error text: {e}"   
)

from ..base import ForecastMethod, DEFAULT_RANGE
from ..utils import copy_doc


class LTSF(ForecastMethod):
    """Uses a transformer for inference."""


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
        """LTSF-Linear Forecaster.

        Implementation of the Long-Term Short-Term Feature (LTSF) linear forecaster,
        aka LTSF-Linear, by Zeng et al [1]_.

        Core logic is directly copied from the cure-lab LTSF-Linear implementation [2]_,
        which is unfortunately not available as a package.

        Args:
            seq_len : int
                length of input sequence
            pred_len : int
                length of prediction (forecast horizon)
            num_epochs : int, default=16
                number of epochs to train
            batch_size : int, default=8
                number of training examples per batch
            in_channels : int, default=1
                number of input channels passed to network
            individual : bool, default=False
                boolean flag that controls whether the network treats each channel individually"
                "or applies a single linear layer across all channels. If individual=True, the"
                "a separate linear layer is created for each input channel. If"
                "individual=False, a single shared linear layer is used for all channels."
            criterion : torch.nn Loss Function, default=torch.nn.MSELoss
                loss function to be used for training
            criterion_kwargs : dict, default=None
                keyword arguments to pass to criterion
            optimizer : torch.optim.Optimizer, default=torch.optim.Adam
                optimizer to be used for training
            optimizer_kwargs : dict, default=None
                keyword arguments to pass to optimizer
            lr : float, default=0.003
                learning rate to train model with

        References:
            .. [1] Zeng A, Chen M, Zhang L, Xu Q. 2023.
            Are transformers effective for time series forecasting?
            Proceedings of the AAAI conference on artificial intelligence 2023
            (Vol. 37, No. 9, pp. 11121-11128).
            .. [2] https://github.com/cure-lab/LTSF-Linear
        """

        self.method_params = locals()
        self.method_params.pop("self") 

        # Make forecasting horizon for fit and predict.  
        self.fh = [i for i in range(1, pred_len + 1)]


    @copy_doc(ForecastMethod._fit)
    def _fit(
        self,
        t: np.ndarray,
        endog_states: np.ndarray,
        exog_states: np.ndarray = None
    ):
  
        self.model = sktime_LTSFLinearForecaster(
            **self.method_params
        )

        m = len(t)
        y = to_sktime_time_series(np.arange(m), endog_states)
        
        if exog_states is not None:
            X = to_sktime_time_series(np.arange(m), exog_states)
        else:
            X = None
        # This uses the sktime_LTSFLinearForecaster fit function.
        self.model.fit(y, X=X, fh=self.fh)


    @copy_doc(ForecastMethod._predict)
    def _predict(
        self,
        t: np.ndarray,
        prior_endog_states: np.ndarray,
        prior_exog_states: Optional[np.ndarray] = None,
        prior_t: Optional[np.ndarray] = None,
        prediction_exog: Optional[np.ndarray] = None,
        rng: np.random.RandomState = DEFAULT_RANGE,
    ) -> np.ndarray:
        
        m = len(prior_t)
        prior_y = to_sktime_time_series(np.arange(m), prior_endog_states)
        
        X = None
        if prediction_exog is not None:
            X = to_sktime_time_series(t, prediction_exog)



        # Save original stored endogeneous.
        _y_orig = self.model._y
        self.model._y = prior_y

        futr_endo = []

        for i in range(len(t) - 1):
            y_next = self.model.predict(X=X, fh=self.fh).iloc[0:1, :]
            self.model._y = pd.concat([self.model._y.iloc[1:, :], y_next])
            futr_endo.append(y_next)

        # Reset stored endogenous to original state.
        self.model._y = _y_orig

        # Extract values.
        pred_endo = pd.concat(futr_endo).values
        # Add initial state to correspond with t.     
        pred_endo = np.vstack([prior_endog_states[-1, :], pred_endo])

        return pred_endo


    @copy_doc(ForecastMethod.get_window_size)
    def get_window_size(self):
        return max(2, self.method_params["seq_len"])
    

    @copy_doc(ForecastMethod.set_params)
    def set_params(self, **params):
        self.method_params = {**self.method_params, **params}
        self.model = sktime_LTSFLinearForecaster(**self.method_params)


    @copy_doc(ForecastMethod.get_params)
    def get_params(self, deep: bool = True) -> dict:
        return self.method_params
    

    @copy_doc(ForecastMethod.get_test_params)
    def get_test_params() -> dict[str, Any]:
        return {
            'seq_len': 2,
            'pred_len': 1,
            'lr': 0.005,
            'optimizer': 'Adam',
            'batch_size': 1,
            'num_epochs': 10,
            'individual': False
        }
    

    @staticmethod
    @copy_doc(ForecastMethod._get_optuna_params)
    def _get_optuna_params(
        trial, max_lags=25, max_horizon=50, **kwargs) -> dict[str, object]:

        return {
            "seq_len": trial.suggest_int("seq_len", 1, max_lags),
            "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
            "pred_len": trial.suggest_int("pred_len", 1, max_horizon),
            "num_epochs": trial.suggest_categorical(
                "num_epochs", [50, 100, 300, 500, 1000])
        }
    

def to_sktime_time_series(time_points: np.ndarray, X: np.ndarray):
    """Converts time series from interfere format to sktime format.

    Args:
        time_points (np.ndarray): A 1D array of time points that correspond to
            the rows of of all arrays in Xs.
        Xs Iterable[np.ndarray]: An iterable containing 2D (m x n_i) array
            where rows are observations and columns are variables. The number of
            rows, `m` must equal the length of `time_points`.
    Returns:
        y (pd.DataFrame): A DataFrame containing the endogenous variables.
            Columns are variables and rows are observations.
        X (pd.DataFrame): A DataFrame containing the exogenoug variables.
            Columns are variables and rows are observations.
    """
    index = pd.to_datetime(
        pd.to_numeric(time_points), unit='s', errors='coerce')
    sktime_X = pd.DataFrame(X, index=index)
    return sktime_X