from copy import copy
from typing import Optional
from warnings import warn

import numpy as np
import pysindy as ps

from ..base import ForecastMethod
from ..base import DEFAULT_RANGE
from ..utils import copy_doc
from ..interventions import ExogIntervention


# Lists of hyper parameters for optimization.
SINDY_LIB_LIST = [ps.PolynomialLibrary, ps.FourierLibrary]

SINDY_DIFF_LIST = [
    dict(kind='finite_difference', k=1),
    dict(kind='finite_difference', k=2),
    dict(kind='finite_difference', k=3),
    dict(kind='spline', s=0.1),
    dict(kind='spline', s=0.5),
    dict(kind='savitzky_golay', order=2, left=0.1, right=0.1),
    dict(kind='savitzky_golay', order=3, left=0.1, right=0.1),
    dict(kind='trend_filtered', order=0, alpha=0.01),
    dict(kind='trend_filtered', order=1, alpha=0.01),
    dict(kind='trend_filtered', order=1, alpha=0.1),
]


class SINDy(ForecastMethod):

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
        # Optionally accept types for feature library.
        if isinstance(feature_library, type):
            feature_library = feature_library()

        # Optionally accept dictionaries for SINDyDerivative.
        if isinstance(differentiation_method, dict):
            differentiation_method = ps.SINDyDerivative(
                **differentiation_method)

        # Differentiation method and feature library must be copied so that
        # their internal state doesn't carry over across different fits.
        if differentiation_method is not None:
            differentiation_method = copy(differentiation_method)
            differentiation_method = differentiation_method.__init__(
                **differentiation_method.get_params()
            )

        if feature_library is not None:
            feature_library = copy(feature_library)

        self.sindy = ps.SINDy(
            optimizer, feature_library, differentiation_method, feature_names, t_default, discrete_time,
        )
        self.max_sim_value = max_sim_value


    @copy_doc(ForecastMethod._fit)
    def _fit(
        self,
        t: np.ndarray,
        endog_states: np.ndarray,
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
        
        # Initial condition (Exogenous signal removed.)
        x0 = prior_endog_states[-1, :]

        # Sindy uses scipy.integrate.solve_ivp by default and solve_ivp
        # uses event functions with assigned attributes as callbacks.
        # The below code tells scipy to stop integrating when
        # too_big(t, y) == True.
        if self.sindy.discrete_time:
            too_big = lambda ti, y: np.any(np.abs(y) > self.max_sim_value)
        else:
            too_big = lambda ti, y: np.all(np.abs(y) < self.max_sim_value)

        too_big.terminal = True

        if self.sindy.discrete_time:
            sindy_t = len(t)
        else:
            sindy_t = t

        # Simulate with intervention
        endog_pred = self.sindy.simulate(
            x0, sindy_t, u=prediction_exog, integrator_kws={"events": too_big}, stop_condition=lambda x: too_big(0, x)
        )

        # Retrive number of successful steps.
        n_steps = endog_pred.shape[0]
        n_missing = len(t) - n_steps

        # Warn user if SINDy diverges.
        if n_missing > 0:
            warn(
                f"SINDy prediction diverged. Valid prediction for {n_steps} / "
                f"{len(t)} time steps."
            )

        # When SINDy diverges, repeat the last valid prediction for the
        # remaining prediction points.
        endog_pred =  np.vstack(
            [endog_pred] +  [endog_pred[-1, :] for i in range(n_missing)]
        )
        return endog_pred
        

    def get_params(self, deep=True):
        return self.sindy.get_params(deep=deep)


    def set_params(self, **params):

        # Differentiation method and feature library must be copied so that
        # their internal state doesn't carry over across different fits.
        if params["differentiation_method"] is not None:
            params["differentiation_method"] = copy(
                params["differentiation_method"])

        if params["feature_library"] is not None:
            params["feature_library"] = copy(params["feature_library"])

        return self.sindy.set_params(**params)


    def get_test_params():
        return {
            "differentiation_method": ps.SINDyDerivative(kind='spectral'),
            "discrete_time": True
        }
    

    @staticmethod
    @copy_doc(ForecastMethod._get_optuna_params)
    def _get_optuna_params(trial, **kwargs):
        return {
            'optimizer__threshold': trial.suggest_float(
                'optimizer__threshold', 1e-5, 5, log=True),

            'optimizer__alpha': trial.suggest_float(
                'optimizer__alpha', 1e-5, 5, log=True),

            'discrete_time': trial.suggest_categorical(
                'discrete_time', [True, False]),

            'feature_library':
                trial.suggest_categorical('feature_library', SINDY_LIB_LIST),

            'differentiation_method':
                trial.suggest_categorical(
                    "differentiation_method", SINDY_DIFF_LIST)
        }
