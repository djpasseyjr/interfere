# Custom forecast methods

This page explains how to add your own forecasting method to use with Interfere's workflows (fitting, prediction, and hyperparameter tuning with `CrossValObjective` and Optuna). We use a **polynomial forecaster** as the running example: it fits a polynomial in time to each variable and extrapolates, with a single hyperparameter `degree`.

## What you need to implement

Subclass `interfere.ForecastMethod` and implement four pieces:

| Method                                   | Purpose                                                                                 |
| ---------------------------------------- | --------------------------------------------------------------------------------------- |
| `_fit(t, endog_states, exog_states)`   | Train your model on the given time series. Store whatever you need for prediction.      |
| `_predict(t, prior_endog_states, ...)` | Return predictions at times `t`, shape `(len(t), n_variables)`.                     |
| `get_window_size()`                    | Return how many past observations the method needs (for CV and prediction).             |
| `_get_optuna_params(trial, ...)`       | Return a dict of constructor kwargs for Optuna to try (so CV can tune hyperparameters). |

You also need `get_test_params()` (static) if the method will be used in the package test suite; for a standalone example you can return a fixed dict, e.g. `{"degree": 2}`.

The polynomial example uses only endogenous data (no exogenous variables). If your method uses exogenous variables, you must handle `exog_states` in `_fit` and `prediction_exog` / `prior_exog_states` in `_predict`; the [Prediction](prediction.md) and built-in methods (e.g. SINDy, VAR) show the expected shapes.

## Complete example

The code below is a full, runnable example. It defines a **polynomial forecaster** (fit a polynomial in time to each variable and extrapolate), then uses it with `CrossValObjective` and Optuna.

```python
from typing import Any, Dict, Optional
import numpy as np
import optuna
import interfere


class PolynomialForecast(interfere.ForecastMethod):
    """Fits a polynomial (in time) to each variable and extrapolates. Hyperparameter: degree."""

    def __init__(self, degree: int = 2):
        self.degree = degree
        self.coeffs_ = None  # filled in by _fit

    def _fit(
        self,
        t: np.ndarray,
        endog_states: np.ndarray,
        exog_states: Optional[np.ndarray] = None,
    ):
        _, n_vars = endog_states.shape
        self.coeffs_ = [
            np.polyfit(t, endog_states[:, j], self.degree)
            for j in range(n_vars)
        ]

    def _predict(
        self,
        t: np.ndarray,
        prior_endog_states: np.ndarray,
        prior_exog_states: Optional[np.ndarray] = None,
        prior_t: Optional[np.ndarray] = None,
        prediction_exog: Optional[np.ndarray] = None,
        rng=None,
    ) -> np.ndarray:
        n_vars = len(self.coeffs_)
        return np.column_stack([
            np.polyval(self.coeffs_[j], t) for j in range(n_vars)
        ])

    def get_window_size(self) -> int:
        return self.degree + 1

    @staticmethod
    def get_test_params() -> Dict[str, Any]:
        return {"degree": 2}

    @staticmethod
    def _get_optuna_params(
        trial,
        max_lags: Optional[int] = None,
        max_horizon=None,
        **kwargs,
    ) -> Dict[str, Any]:
        max_degree = 10
        if max_lags is not None and max_lags >= 2:
            max_degree = min(max_degree, max_lags - 1)
        max_degree = max(1, max_degree)
        return {"degree": trial.suggest_int("degree", 1, max_degree)}


# --- Use it with the hyperparameter optimizer ---

# Simple 1D time series: quadratic in time + noise
t = np.linspace(0, 10, 100)
y = 0.5 * t**2 - 2 * t + 1 + np.random.RandomState(42).normal(0, 0.5, size=t.size)
data = y.reshape(-1, 1)  # shape (n_timesteps, n_variables)

cv = interfere.CrossValObjective(
    method_type=PolynomialForecast,
    data=data,
    times=t,
    train_window_percent=0.6,
    num_folds=4,
    exog_idxs=[],
    num_val_prior_states=10,
    metric=interfere.metrics.rmse,
    metric_direction="minimize",
)

study = optuna.create_study(direction="minimize")
study.optimize(cv, n_trials=8, show_progress_bar=True)

print("Best degree:", study.best_params["degree"])
print("Best CV RMSE:", round(study.best_value, 6))
```

In the example, **data** has shape `(n_timesteps, n_variables)` and **times** is 1D. **CrossValObjective** gets your class as `method_type`; it uses `_get_optuna_params` by default, so each Optuna trial builds `PolynomialForecast(**params)`, runs sliding-window CV, and returns the metric. Use `study.best_params` to instantiate your method and call `fit` / `predict` (or `simulate` with interventions) as in the [Quick Start](index.md#quick-start) and [Optimization](optimization.md) docs.
