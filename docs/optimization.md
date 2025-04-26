# Optimization

Interfere's optimization module provides a robust cross-validation workflow for hyperparameter tuning of dynamical forecasting methods using Optuna.

## CrossValObjective Overview

The `CrossValObjective` class implements an Optuna-compatible objective function for tuning any `ForecastMethod`. It performs sliding-window cross-validation on time series data, systematically training and scoring forecasting or intervention-response models over multiple folds.

### Key Parameters

- **method_type** *(Type[ForecastMethod])*: Forecasting method class (e.g., `SINDy`, `VAR`, `ResComp`).
- **data** *(np.ndarray)*: Time series array of shape (T, n_vars). Rows are time points, columns are variables.
- **times** *(np.ndarray)*: 1D array of time stamps corresponding to each row in `data`.
- **train_window_percent** *(float)*: Fraction of `data` used for training in each fold (0 < p < 1).
- **num_folds** *(int)*: Total number of folds (including the initial training window).
- **exog_idxs** *(Optional[list[int]])*: Indices of columns in `data` treated as exogenous variables during fitting and validation.
- **val_scheme** *(str)*: Validation strategy: `"forecast"` (score the chunk immediately after training), `"last"` (score a fixed hold-out chunk at end), or `"all"` (score all hold-out chunks).
- **num_val_prior_states** *(int)*: Number of prior observations used as context for each validation prediction.
- **metric** *(Callable[[np.ndarray, np.ndarray], float])*: Callable metric function accepting `actual` and `predicted` arrays and returning a scalar error (e.g., `interfere.metrics.rmse`).
- **metric_direction** *(str)*: `"minimize"` or `"maximize"`, passed to Optuna's study.
- **hyperparam_func** *(Callable)*: Trial-to-parameter mapping function. Defaults to the model's `_get_optuna_params`.
- **store_preds** *(bool)*: If `True`, stores per-fold predictions in `CrossValObjective.trial_results` for inspection.
- **raise_errors** *(bool)*: If `True`, propagates exceptions during CV; otherwise, assigns a large penalty value for that trial.

### Metrics

Interfere provides simple callable metrics in the `interfere.metrics` module. Metrics should accept two arrays (`actual`, `predicted`) of the same shape and return a float. Exogenous variables will be automatically dropped during metric evaluation.

| Function                       | Description                                    |
|--------------------------------|------------------------------------------------|
| `interfere.metrics.rmse`       | Root mean squared error                        |
| `interfere.metrics.mae`        | Mean absolute error                            |
| `interfere.metrics.mse`        | Mean squared error                             |
| `interfere.metrics.mape`       | Mean absolute percentage error                 |
| `interfere.metrics.nrmse`      | Normalized root mean squared error             |
| `interfere.metrics.rmsse`      | Root mean squared scaled error                 |

Example:

```python
import interfere

# compute error between observed and forecasted values
error = interfere.metrics.rmse(actual, predicted)
```

### Cross-Validation Workflow

1. **Partition Data**: Split `data` into a sliding training window (size = `train_window_percent` × T) and subsequent validation chunks.
2. **Slide Window**: For each of the `num_folds`, advance the training window by one validation chunk and retrain the forecasting method from scratch.
3. **Score Predictions**: Depending on `val_scheme`, compute the error metric on hold-out observations immediately after training (`"forecast"`), at the end (`"last"`), or on all chunks (`"all"`).
4. **Aggregate Result**: Return the aggregated metric across folds to Optuna to guide hyperparameter search.

## Supported Forecasting Methods

| Method            | Description                                        |
|-------------------|----------------------------------------------------|
| `AverageMethod`   | Forecasts the historical mean                       |
| `VAR`             | Vector autoregression via StatsModels               |
| `SINDy`           | Sparse regression-based identification of dynamics  |
| `ResComp`         | Reservoir computing with Tikhonov regularization    |

## Example Usage

```python
import numpy as np
import interfere
import optuna

# 1. Simulate training data
t = np.linspace(0, 10, 500)
x0 = np.random.rand(3)
dynamics = interfere.dynamics.Lorenz(sigma=0.3)
data = dynamics.simulate(t, x0)

# 2. Initialize CrossValObjective
cv = interfere.CrossValObjective(
    method_type=interfere.SINDy,
    data=data,
    times=t,
    train_window_percent=0.5,
    num_folds=4,
    exog_idxs=[],
    val_scheme="forecast",
    num_val_prior_states=5,
    metric=interfere.metrics.RootMeanSquaredError(),
    metric_direction="minimize",
)

# 3. Optimize with Optuna
study = optuna.create_study(direction=cv.metric_direction)
study.optimize(cv, n_trials=20)

print("Best hyperparameters:", study.best_params)
``` 