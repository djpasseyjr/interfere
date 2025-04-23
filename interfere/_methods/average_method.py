from typing import Any, Dict, List, Optional

import numpy as np

from ..base import ForecastMethod, DEFAULT_RANGE
from ..utils import copy_doc


class AverageMethod(ForecastMethod):
    """The average method--predicts average of historic data."""

    def __init__(self):
        """Initializes the average method--predicts average of historic data."""
        pass
    
    def _fit(
        self,
        t: np.ndarray,
        endog_states: np.ndarray,
        exog_states: np.ndarray = None
    ):
        self.avgs = np.mean(endog_states, axis=0)


    def _predict(
        self,
        t: np.ndarray,
        prior_endog_states: np.ndarray,
        prior_exog_states: Optional[np.ndarray] = None,
        prior_t: Optional[np.ndarray] = None,
        prediction_exog: Optional[np.ndarray] = None,
        rng: np.random.RandomState = DEFAULT_RANGE,
    ) -> np.ndarray:
        return np.vstack(
            [self.avgs for ti in t]
        )
    
    def get_test_params() -> Dict[str, Any]:
        return {}
    
    @staticmethod
    @copy_doc(ForecastMethod._get_optuna_params)
    def _get_optuna_params(trial, **kwargs) -> Dict[str, Any]:
        return {}
        