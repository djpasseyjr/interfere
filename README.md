# Interfere

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for modeling and predicting the response of complex dynamic systems to interventions.

## Overview

Interfere is a research-oriented Python package that addresses a fundamental question in complex systems: *When can we predict how complex systems will respond to interventions?* This package provides tools for:

- Modeling dynamic nonlinear multivariate stochastic systems.
- Simulating and analyzing how such systems respond to interventions.
- Generating complex dynamic counterfactuals.
- Studying causal relationships in complex systems.

## Installation

### From GitHub

```bash
pip install git+https://github.com/djpasseyjr/interfere
```

### From Local Clone

```bash
git clone https://github.com/djpasseyjr/interfere.git
cd interfere
pip install .
```

## Quick Start

The Interfere package is designed around three main tasks: counterfactual simulation, predictive method optimization, and prediction. Here's a complete example using the SINDY (Sparse Identification of Nonlinear Dynamics) method:

### 1. Counterfactual Simulation

First, let's create and simulate a dynamic model:

```python
import numpy as np
import interfere

# Set up simulation parameters
initial_cond = np.random.rand(3)
t_train = np.arange(0, 10, 0.05)
dynamic_model = interfere.dynamics.Belozyorov3DQuad()

# Generate observation period
Y = dynamic_model.simulate(t_train, initial_cond)
```

![Original System Trajectory](images/original_trajectory.png)

### 2. Applying an Intervention

Next, we'll apply a sinusoidal intervention to one component of the system:

```python
# Generate forecasting period
test_t = np.arange(t_train[-1], 12, 0.05)

# Create intervention (e.g., controlling x1 with sin(t))
interv = interfere.SignalIntervention(1, np.sin)

# Simulate
Y_treat = dynamic_model.simulate(test_t, prior_states=Y, intervention=interv)

```

![System Trajectory with Intervention](images/intervention_effect.png)

### 3. Model Optimization and Prediction

Finally, we'll optimize a SINDY model and predict the system's response:

```python
import optuna

# Set up cross-validation for optimization
method_type = interfere.SINDY
cv_objv = interfere.CrossValObjective(
    method_type=method_type,
    data=Y,
    times=t_train,
    train_window_percent=0.3,
    num_folds=5,
    exog_idxs=interv.intervened_idxs,
)

# Optimize hyperparameters
study = optuna.create_study()
study.optimize(cv_objv, n_trials=25)
params = study.best_params

# Fit model and make predictions
method = method_type(**params)
Y_endog, Y_exog = interv.split_exog(Y)
method.fit(t_train, Y_endog, Y_exog)

# Predict intervention response
pred_Y_treat = method.simulate(
    t_test,
    prior_states=Y,
    intervention=interv
)
```

![Predicted vs Actual Intervention Response](images/prediction_comparison.png)

The SINDY method identifies the underlying dynamics of the system using sparse regression techniques, making it particularly effective for discovering interpretable mathematical models of complex systems.

## Dependencies

Core dependencies:

- matplotlib
- networkx
- numpy
- optuna
- pyclustering
- pysindy
- scikit-learn
- statsmodels
- typing_extensions

Optional dependencies for additional methods:

- neuralforecast
- statsforecast
- sktime

## Example

The package can be used to simulate and analyze how systems respond to interventions. For example, it can model the effect of stochasticity on intervention response forecasting:

![Stochastic vs Deterministic Systems](https://github.com/djpasseyjr/interfere/blob/c7090043aec4a984a45517794d266df4eb105f79/images/det_v_stoch.png?raw=true)

## Documentation

For detailed documentation and usage examples, please refer to the [paper](paper.md) and [paper.pdf](paper.pdf).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@article{passey2024interfere,
  title={Interfere: Intervention Response Simulation and Prediction for Stochastic Nonlinear Dynamics},
  author={Passey, D. J. and Mucha, Peter J.},
  journal={arXiv preprint},
  year={2024}
}
```

## Contact

- Author: DJ Passey (djpassey@unc.edu)
- Institution: University of North Carolina at Chapel Hill
