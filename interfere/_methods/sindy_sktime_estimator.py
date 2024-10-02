# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# License URL:
# https://github.com/sktime/sktime/blame/df21a0c0275ebf28deb30efac5d469c9f0d178e3/LICENSE
"""An sktime wrapper of the PySINDY algorithm.
"""

import pandas as pd
import pysindy as ps
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon

from ..utils import copy_doc

STLSQ_PARAM_GRID = {
    "threshold": [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
    "alpha": [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0, 5.0]
}

SR3_PARAM_GRID = {
    "threshold": [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
    "thresholder": ["l1", "l2", "CAD"],
    "nu": [0.0, 0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0],
}

SINDY_PARAM_GRID = {
    'differentiation_method': [ps.SINDyDerivative(kind="savitzky_golay", left=0.5, right=0.5, order=3)],
    'feature_library__degree': [2, 3],
    'feature_library': [ps.PolynomialLibrary()],
    'optimizer__alpha': [0.0, 0.04, 0.05, 0.06, 0.1],
    'optimizer__threshold': [0.001],
    'optimizer': [ps.optimizers.STLSQ()],
}


def sktime_to_sindy_arrays(*args):
    """All in one conversion function from sktime to SINDy compatable arrays."""
    sindy_arrays = []
    for x in args:
        if x is None:
            sindy_arrays.append(None)
        if type(x) == pd.DataFrame:
            sindy_arrays.append(x.values)
        if type(x) == ForecastingHorizon:
            sindy_arrays.append( pd.to_numeric(x.to_pandas()).values)
    return sindy_arrays


class SINDYForecaster(BaseForecaster):

    _tags = {
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "both",
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
    }

    @copy_doc(ps.SINDy.__init__)
    def __init__(self, 
        optimizer=None,
        feature_library=None,
        differentiation_method=None,
        feature_names=None,
        t_default=1,
        discrete_time=False,
        **kwargs
    ):
        super().__init__()
        self.sindy_model = ps.SINDy(
            optimizer, feature_library, differentiation_method, feature_names, t_default, discrete_time,
        )
        self.sindy_model.set_params(**kwargs)

    def _fit(self, y, X, fh):
        # Convert time series to numpy arrays
        x, u = sktime_to_sindy_arrays(y, X)
        t = pd.to_numeric(y.index).values

        self._x0 = x[-1]
        self._sindy_model = self.sindy_model.fit(x, t, u=u)
        return self
    
    def _predict(self, fh, X):
        u, t = sktime_to_sindy_arrays(X, fh)
        print(self._x0, t, u)
        y_pred = self._sindy_model.simulate(self._x0, t, u, integrator="odeint")
        y_pred = pd.DataFrame(data=y_pred, index=t)
        return y_pred
    
    def get_params(self, *args, **kwargs):
        return self.sindy_model.get_params()

