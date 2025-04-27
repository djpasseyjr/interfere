# 🌀 Interfere

[![PyPI Version](https://img.shields.io/pypi/v/interfere)](https://pypi.org/project/interfere/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for modeling and predicting the response of complex dynamic systems to interventions.

## Overview

Interfere is a research-oriented Python package that addresses a fundamental question in complex systems: *When can we predict how complex systems will respond to interventions?* This package provides tools for:

- Modeling dynamic nonlinear multivariate stochastic systems.
- Simulating and analyzing how such systems respond to interventions.
- Generating complex dynamic counterfactuals.
- Studying causal relationships in complex systems.

## Interfere Benchmark Dataset ([Download](https://drive.google.com/file/d/19_Ha-D8Kb1fFJ_iECU62eawbeuCpeV_g/view?usp=sharing))

![Sixty dynamic systems and intervention responses.](images/sixty_models.png)

The image above depicts the uninterrupted trajectories of sixty dynamic models
in blue and their response to a particular intervention in red. This data is
available for download as the [Interfere Benchmark
1.1.1](https://drive.google.com/file/d/19_Ha-D8Kb1fFJ_iECU62eawbeuCpeV_g/view?usp=sharing). It can be used to benchmark a forecasting method's ability to predict the
response of a dynamic system to interventions.

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

The Interfere package is designed around three main tasks: counterfactual simulation, predictive method optimization, and prediction. Here's a complete example using the SINDy (Sparse Identification of Nonlinear Dynamics) method:

### 1. Counterfactual Simulation

First, let's create and simulate a dynamic model:

```python
import numpy as np
import interfere
import optuna

# Set up simulation parameters
initial_cond = np.random.rand(3)
t_train = np.arange(0, 10, 0.05)
dynamics = interfere.dynamics.Belozyorov3DQuad(sigma=0.5)

# Generate trajectory
sim_states = dynamics.simulate(t_train, initial_cond)
```

![Original System Trajectory](images/original_trajectory.png)

### 2. Applying an Intervention

Next, we'll apply an intervention to one component of the system:

```python
# Time points for the intervention simulation
test_t = np.arange(t_train[-1], 15, 0.05)

# Intervention initialization
intervention = interfere.SignalIntervention(iv_idxs=1, signals=np.sin)

# Simulate intervention
interv_states = dynamics.simulate(
    test_t,
    prior_states=sim_states,
    intervention=intervention,
)
```

![System Trajectory with Intervention](images/intervention_effect.png)

### 3. Model Optimization and Prediction

Using the generated data, we can run hyperparameter optimization with a
forecasting method. All forecasting methods come with reasonable hyperparameter
ranges built in.

```python
# Select the SINDy method for hyperparameter optimization.
method_type = interfere.SINDy

# Create an objective function that aims to minimize cross validation error
# over different hyper parameter configurations for SINDy
cv_obj = interfere.CrossValObjective(
    method_type=method_type,
    data=sim_states,
    times=t_train,
    train_window_percent=0.3,
    num_folds=5,
    exog_idxs=intervention.iv_idxs,
)

# Run the study using optuna.
study = optuna.create_study()
study.optimize(cv_obj, n_trials=25)

# Collect the best hyperparameters into a dictionary.
best_param_dict = study.best_params
```

### 4. Intervention Response Prediction

Using the best parameters found, we can fit the forecasting method to
pre-intervention data and then make a prediction about how the system will
respond to the intervention.

```python
# Initialize SINDy with the best perfoming parameters.
method = interfere.SINDy(**study.best_params)

# Use an intervention helper function to split the pre-intervention data
# into endogenous and exogenous columns.
Y_endog, Y_exog = intervention.split_exog(sim_states)

# Fit SINDy to the pre-intervention data.
method.fit(t_train, Y_endog, Y_exog)

# Use the inherited interfere.ForecastingMethod.simulate() method
# To simulate intervention response using SINDy
pred_traj = method.simulate(
    test_t, prior_states=sim_states, intervention=intervention
)
```

![Predicted vs Actual Intervention Response](images/prediction_comparison.png)

The SINDy method identifies the underlying dynamics of the system using sparse regression techniques, making it particularly effective for discovering interpretable mathematical models of complex systems.

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

For a more detailed explanation of the purpose of the package refer to [paper.pdf](paper.pdf).

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
  year={2025}
}
```

## Contact

- Author: DJ Passey (djpassey@unc.edu)
- Institution: University of North Carolina at Chapel Hill
