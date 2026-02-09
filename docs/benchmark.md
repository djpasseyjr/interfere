# Working with the benchmark

The [Interfere Benchmark](https://drive.google.com/file/d/19_Ha-D8Kb1fFJ_iECU62eawbeuCpeV_g/view?usp=sharing) is a set of JSON files. Each file encodes one **observational vs intervention** problem: you get training data, an **observational** trajectory (continuation with the same exogenous driver), and an **intervention** trajectory (the same system under an intervention). The goal is to fit a forecasting method on the training data and compare how well it predicts the observational continuation versus the intervention response.

## What's in the JSON

Each benchmark JSON includes (among other keys) the following at the top level:

| Key                          | Description                                                                     |
| ---------------------------- | ------------------------------------------------------------------------------- |
| `metadata`                 | Variable descriptions and schema info                                           |
| `model_description`        | Text description of the underlying dynamics                                     |
| `initial_condition_times`  | Times for the short history before training                                     |
| `initial_condition_states` | States at those times (prior to `train_*`)                                    |
| `train_times`              | 1D array of training time points                                                |
| `train_states`             | 2D array `(n_train, n_vars)` of training observations                         |
| `train_exog_idxs`          | Column indices of exogenous variables during training (same as forecast)        |
| `forecast_times`           | Time points for the forecast period                                             |
| `forecast_states`          | **Observational** trajectory: system continued with same exogenous driver |
| `forecast_exog_idxs`       | Same as `train_exog_idxs`                                                     |
| `causal_resp_states`       | **Intervention** trajectory: response under the "do" intervention         |
| `causal_resp_times`        | Same length as `forecast_times` (same time grid)                              |
| `causal_resp_exog_idxs`    | Exogenous indices under the intervention (includes any newly driven variables)  |
| `target_idx`               | (Optional) column index of the prediction target                                |

The **intervention is not stored as a separate object**. It is implied by the exogenous columns: during the observational period they follow `train_states` / `forecast_states` at `forecast_exog_idxs`; during the intervention period they follow `train_states` / `causal_resp_states` at `causal_resp_exog_idxs`. You reconstruct callable interventions by interpolating those time series (see below).

## The three stages

1. **Training**Fit a forecasting method on `train_states` at `train_times`, with exogenous variables at `train_exog_idxs`.
2. **Observational forecast**Using the **same** exogenous driver as in training (values from the observational run: training then `forecast_states`), simulate over `forecast_times` and compare to `forecast_states`.
3. **Intervention forecast**
   Using the **intervention** exogenous driver (values from the intervention run: training then `causal_resp_states` at `causal_resp_exog_idxs`), simulate over the same `forecast_times` and compare to `causal_resp_states`.

So: same training data; one continuation is "observational", the other is "under intervention"; the benchmark compares how well a method predicts both.

## Loading the JSON and building interventions

The package expects interventions as **callables** of time (e.g. `SignalIntervention` with scalar functions). The JSON only gives (time, value) arrays. So we load the arrays and turn the exogenous columns into callables via interpolation (e.g. `scipy.interpolate.interp1d`):

```python
import json
import numpy as np
import interfere
from scipy.interpolate import interp1d

def load_benchmark(path):
    """Load benchmark JSON and build interventions from exogenous columns."""
    with open(path) as f:
        data = json.load(f)

    train_t = np.array(data["train_times"])
    forecast_t = np.array(data["forecast_times"])
    train_states = np.array(data["train_states"])
    forecast_states = np.array(data["forecast_states"])
    causal_resp_states = np.array(data["causal_resp_states"])
    obs_exog_idxs = data["forecast_exog_idxs"]
    do_exog_idxs = data["causal_resp_exog_idxs"]

    # Time grid for interpolation: end of train (excl. last) + full forecast
    obs_t = np.hstack([train_t[:-1], forecast_t])

    # Observational intervention: exog follows observational trajectory (train -> forecast_states)
    obs_intervention = interfere.SignalIntervention(
        obs_exog_idxs,
        [
            interp1d(obs_t, np.hstack([train_states[:-1, i], forecast_states[:, i]]))
            for i in obs_exog_idxs
        ],
    )

    # Do-intervention: exog follows intervention trajectory (train -> causal_resp_states)
    do_intervention = interfere.SignalIntervention(
        do_exog_idxs,
        [
            interp1d(obs_t, np.hstack([train_states[:-1, i], causal_resp_states[:, i]]))
            for i in do_exog_idxs
        ],
    )

    return {
        "train_t": train_t,
        "forecast_t": forecast_t,
        "train_states": train_states,
        "forecast_states": forecast_states,
        "causal_resp_states": causal_resp_states,
        "obs_intervention": obs_intervention,
        "do_intervention": do_intervention,
        "model_description": data.get("model_description", ""),
    }
```

This uses only `interfere`, `numpy`, and `scipy`; no separate experiments package.

## Minimal end-to-end example

After loading with `load_benchmark`, fit on training data (with observational exog), run an observational forecast, then fit again with the do-intervention exog and run the intervention forecast. Compare both to the benchmark trajectories:

```python
# Load (path to your benchmark JSON)
data = load_benchmark("examples/AttractingFixedPoint4D.json")
train_t = data["train_t"]
forecast_t = data["forecast_t"]
train_states = data["train_states"]
obs_intervention = data["obs_intervention"]
do_intervention = data["do_intervention"]

# Fit on training data
endog, exog = obs_intervention.split_exog(train_states)
method = interfere.SINDy()
method.fit(train_t, endog, exog)

# Observational forecast: same exog as training continuation
observational_pred = method.simulate(
    forecast_t,
    prior_states=train_states,
    prior_t=train_t,
    intervention=obs_intervention,
)

# Intervention forecast: fit with do-intervention exog, then simulate
method2 = interfere.SINDy()
endog2, exog2 = do_intervention.split_exog(train_states)
method2.fit(train_t, endog2, exog2)
intervention_pred = method2.simulate(
    forecast_t,
    prior_states=train_states,
    prior_t=train_t,
    intervention=do_intervention,
)

# Compare to ground truth
forecast_rmse = np.sqrt(np.mean((observational_pred - data["forecast_states"]) ** 2))
intervention_rmse = np.sqrt(np.mean((intervention_pred - data["causal_resp_states"]) ** 2))
print("RMSE observational forecast:", forecast_rmse)
print("RMSE intervention forecast:", intervention_rmse)
```
