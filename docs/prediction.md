# Prediction

Prediction in Interfere empowers you to translate fitted forecasting methods into precise counterfactual system trajectories. By leveraging the same `simulate` interface, you can compare baseline dynamics to intervention-driven responses with minimal code changes.

## API Reference: `fit()` & `predict()`

### `fit()`

```python
fit(
    self,
    t: np.ndarray,
    endog_states: np.ndarray,
    exog_states: Optional[np.ndarray] = None,
) -> ForecastMethod
```

- **t** *(np.ndarray)*: 1D array of time points with shape `(n,)`, strictly increasing.
- **endog_states** *(np.ndarray)*: 2D array `(n, d)` of endogenous variable observations.
- **exog_states** *(Optional[np.ndarray])*: 2D array `(n, k)` of exogenous signals; defaults to no exogenous inputs.

**Returns**: The fitted forecasting method instance (`self`).

### `predict()`

```python
predict(
    self,
    t: np.ndarray,
    prior_endog_states: np.ndarray,
    prior_exog_states: Optional[np.ndarray] = None,
    prior_t: Optional[np.ndarray] = None,
    prediction_exog: Optional[np.ndarray] = None,
    prediction_max: float = 1e9,
    rng: np.random.RandomState = DEFAULT_RANGE
) -> np.ndarray
```

- **t** *(np.ndarray)*: 1D array `(m,)` of future time points to predict.
- **prior_endog_states** *(np.ndarray)*: 2D array `(p, d)` of historic endogenous observations used as initial conditions or lagged values.
- **prior_exog_states** *(Optional[np.ndarray])*: 2D array `(p, k)` of historic exogenous signals.
- **prior_t** *(Optional[np.ndarray])*: 1D array `(p,)` of times corresponding to `prior_*_states`. If `None`, inferred from `t` spacing.
- **prediction_exog** *(Optional[np.ndarray])*: 2D array `(m, k)` of future exogenous inputs.
- **prediction_max** *(float)*: Threshold to cap predictions and prevent overflow.
- **rng** *(np.random.RandomState)*: Random state for reproducible stochastic forecasts.

**Returns**: A 2D array `(m, d)` of predicted endogenous states, including the first row as the initial condition.

## Prediction Methods

Use any forecasting method that inherits from `ForecastMethod`.

| Method            | Description                                                     |
| ----------------- | --------------------------------------------------------------- |
| `AverageMethod` | Baseline forecast using average historical data                 |
| `VAR`           | Vector autoregression forecast using statsmodels                |
| `SINDY`         | Sparse model identification for nonlinear dynamics              |
| `ResComp`       | Reservoir computing forecast with Tikhonov regularization       |
| `ARIMA`         | Classical time series ARIMA model via Nixtla StatsForecast      |
| `LSTM`          | Long Short-Term Memory RNN forecaster via Nixtla NeuralForecast |
| `NHITS`         | NHITS deep learning forecaster via Nixtla NeuralForecast        |

## Example

```python
import numpy as np
import interfere

# Pre-intervention simulation
t = np.arange(0, 10, 0.1)
x0 = np.random.rand(2)
dynamics = interfere.dynamics.Lorenz(sigma=0.2)
states = dynamics.simulate(t, x0)

# Define intervention
interv = interfere.PerfectIntervention(0, 2.0)

# Generate pre-intervention data
t0 = t
n_endog, n_exog = states.shape[1]-len(interv.iv_idxs), len(interv.iv_idxs)
endog, exog = interv.split_exog(states)

# Fit SINDY to pre-intervention data
method = interfere.SINDY()
method.fit(t0, endog, exog)

# Forecast with intervention
t_pred = np.arange(t[-1], t[-1]+5, 0.1)
pred_states = method.simulate(t_pred, states, intervention=interv)

# Inspect predicted states shape
print(pred_states.shape)  # (50, endog+exog dims)
```
