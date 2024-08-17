from typing import Dict, List, Optional
from warnings import warn

import numpy as np
import pysindy as ps

from .base import BaseInferenceMethod
from ..base import DEFAULT_RANGE
from ..utils import copy_doc
from ..interventions import ExogIntervention


class SINDY(BaseInferenceMethod):

    @copy_doc(ps.SINDy.__init__)
    def __init__(self, 
        optimizer=None,
        feature_library=None,
        differentiation_method=None,
        feature_names=None,
        t_default=1,
        discrete_time=False,
        max_sim_value = 10000,
        **kwargs
    ):
        self.sindy = ps.SINDy(
            optimizer, feature_library, differentiation_method, feature_names, t_default, discrete_time,
        )
        self.max_sim_value = max_sim_value


    @copy_doc(BaseInferenceMethod._fit)
    def _fit(
        self,
        endog_states: np.ndarray,
        t: np.ndarray,
        exog_states: np.ndarray = None
    ):
        if np.any(endog_states > self.max_sim_value):
            raise ValueError("Supplied endogenous states cannot be simulated "
                f"because they exceed `max_sim_value = {self.max_sim_value}`. "
                "Reinitialize and set `max_sim_value` greater than "
                f"`max(endog_states) = {np.max(endog_states)}` before calling "
                "`fit()`."
            )
        self.__init__(**self.get_params())
        self.sindy.fit(endog_states, t, u=exog_states)


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
        
        # Initial condition (Exogenous signal removed.)
        x0 = historic_endog[-1, :]

        # Sindy uses scipy.integrate.solve_ivp by default and solve_ivp
        # uses event functions with assigned attributes as callbacks.
        # The below code tells scipy to stop integrating when
        # too_big(t, y) == True.
        if self.sindy.discrete_time:
            too_big = lambda t, y: np.any(np.abs(y) > self.max_sim_value)
        else:
            too_big = lambda t, y: np.all(np.abs(y) < self.max_sim_value)

        too_big.terminal = True

        if self.sindy.discrete_time:
            t = len(forecast_times)
        else:
            t = forecast_times

        # Simulate with intervention
        endog_pred = self.sindy.simulate(
            x0, t, u=exog, integrator_kws={"events": too_big}, stop_condition=lambda x: too_big(0, x)
        )

        # Retrive number of successful steps.
        n_steps = endog_pred.shape[0]
        n_missing = len(forecast_times) - n_steps

        # Warn user if SINDY diverges.
        if n_missing > 0:
            warn(
                f"SINDY prediction diverged. Valid prediction for {n_steps} / "
                f"{len(forecast_times)} time steps."
            )

        # When SINDY diverges, repeat the last valid prediction for the
        # remaining prediction points.
        endog_pred =  np.vstack(
            [endog_pred] +  [endog_pred[-1, :] for i in range(n_missing)]
        )
        return endog_pred
        

    def get_params(self, deep=True):
        return self.sindy.get_params(deep=deep)


    def set_params(self, **parameters):
        return self.sindy.set_params(**parameters)


    def get_test_params():
        return {
            "differentiation_method": ps.SINDyDerivative(kind='spectral'),
            "discrete_time": True
        }
    

    def get_test_param_grid():
        return {
            "optimizer": [ps.optimizers.STLSQ()],
            "optimizer__threshold": [0.001, 0.1],
            "differentiation_method": [
                ps.SINDyDerivative(kind='finite_difference', k=1),
                ps.SINDyDerivative(kind='spline', s=0.1)
            ],
            "feature_library": [
                ps.feature_library.PolynomialLibrary(),
                ps.feature_library.FourierLibrary()
            ],
            "discrete_time": [True, False]
        }