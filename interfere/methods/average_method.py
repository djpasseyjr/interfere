from typing import Any, Dict, List

import numpy as np

from ..base import DEFAULT_RANGE
from .base import BaseInferenceMethod


class AverageMethod(BaseInferenceMethod):
    """The average method--predicts average of historic data."""

    def __init__(self):
        """Initializes the average method--predicts average of historic data."""
        pass
    
    def _fit(
        self,
        endog_states: np.ndarray,
        t: np.ndarray,
        exog_states: np.ndarray = None
    ):
        self.avgs = np.mean(endog_states, axis=0)


    def _predict(
        self,
        forecast_times: np.ndarray,
        historic_endog: np.ndarray,
        historic_times: np.ndarray,
        exog: np.ndarray = None,
        historic_exog: np.ndarray = None,
        rng = DEFAULT_RANGE
    ) -> np.ndarray:
        return np.vstack(
            [self.avgs for t in forecast_times]
        )
    
    def get_test_param_grid() -> Dict[str, List[Any]]:
        return {}
    
    def get_test_params() -> Dict[str, Any]:
        return {}
        