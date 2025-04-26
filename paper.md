---
title: 'Interfere: Intervention Response Simulation and Prediction for Stochastic Nonlinear Dynamics'
tags:
  - Python
  - dynamics
  - stochastic differential equations
  - forecasting
  - time series
  - non linear
  - chaotic
  - simulation
  - modeling
  - complex systems
  - causality
  - intervention
  - manipulation
authors:
  - name: D. J. Passey
    orcid: 0000-0002-9310-3361
    ##equal-contrib: true
    affiliation: 1
  - name: Peter J. Mucha
    orcid: 0000-0002-0648-7230
    ##equal-contrib: true
    affiliation: 2
affiliations:
  - name: University of North Carolina at Chapel Hill, United States
    index: 1
    ror: 0130frc33
  - name: Dartmouth College, United States
    index: 2
    ror: 049s0rh22
date: 20 March 2025
bibliography: paper.bib
---
# Summary

The vision of Interfere is simple: What if we used high quality scientific models to benchmark causal prediction tools? Randomized experimental data and counterfactuals are essential for testing methods that attempt to infer causal relationships from data, but obtaining such datasets can be expensive and difficult. Mechanistic models are commonly developed to simulate scenarios and predict the response of systems to interventions across economics, neuroscience, ecology, systems biology and other areas [@brayton_frbus_2014; @izhikevich_large-scale_2008; @banks_parameter_2017; @baker_mechanistic_2018] . Because these models are painstaking calibrated with the real world, they have the ability to generate diverse and complex synthetic counterfactual data that are characteristic of the real processes they emulate. Interfere offers the first steps towards this vision by combining (1) a general interface for simulating the effect of interventions on dynamic simulation models, (2) a suite of predictive methods and cross validation tools, and (3) an [initial benchmark data set](https://drive.google.com/file/d/19_Ha-D8Kb1fFJ_iECU62eawbeuCpeV_g/view?usp=sharing) of dynamic counterfactual scenarios.

![Three dimensional trajectories of sixty scenarios simulated with the Interfere
package. All models depicted are either differential equations or discrete time
difference equations. The trajectory in blue represents the natural behavior of
the system and the red depicts how the system responds to an intervnetion.
Many of the models pictured have more than three dimensions (in such cases,
only the three dimensions of the trajectory with highest variance
are shown). These sixty scenarios make up the Interfere Benchmark for
intervention response prediction which is available online for download
\label{fig:sixty_models}](images/sixty_models.png)

# Statement of Need

Over the past twenty years the scientific community has experience the emergence of multiple frameworks for identifying causal relationships in observational data [@imbens_causal_2015; @pearl_causality_2009; @wieczorek_information_2019]. The most influential frameworks are probabilistic and, while is not an necessary condition for identifying causality, historically a linear relationship was often assumed. However, when attempting to anticipate the response of complex systems in the medium and long term, a linear approximation of the dynamics is insufficient.  Therefore, scientists have increasingly begun to employ non linear techniques for causal analysis e.g. [@runge_discovering_2022; ]. Still, there are relatively few techniques that are able to fit causal dynamic nonlinear models to data. Because of this, we see an opportunity to bring together the insights from recent breakthroughs in causal inference with historical work in dynamic model development.

In order to facilitate this cross pollination, we focus on a key causal problem --- predicting how a complex system responds to a previously unobserved intervention --- and designed the Interfere package for benchmarking tools aimed at intervention response prediction. The dynamic models contained in Interfere present challenges for causal inference that can likely only be addressed with the incorporation of mechanistic assumptions alongside probabilistic tool. As such, the Interfere package presents an opportunity for cross pollination between the causal inference community and the modeling and simulation community.

![Example experimental setup possible with Interfere: Comparing intervention response prediction for deterministic and stochastic systems.\label{fig:det_vs_stoch}](images/det_v_stoch.png)

# Usage

The Interfere package is designed around four tasks: (1) Simulation, (2) )intervention,
(3) forecasting method optimization and (4) intervention response prediction. The following
section will describe each task in detail with example code.

## 1. Simulation.

The models implemented in the interfere package are mainly stochastic
differential equations simulated with Ito's method (e.g. $d\mathbf{X} = A
\mathbf{X} + d\mathbf{W}$) or difference equations (e.g. $x[n+1] = 0.25 x[n] -
0.5 x[n-1]$),
simulated via initial conditions and stepping forward in time. Each dynamic model class included in the Interfere package has a simulate method.  To run a
simulation, the package requires an array of equally spaced time values and an
initial condition or past observations.

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

![Original System Trajectory Simulation of natural, uninterupted evolution of a chaotic
system studied in [@belozyorov_exponential_2015] with the addition of a small
amount of stochastic noise. For this example, we've let
$x=x_1$, $y=x_2$ and and do not plot $x_0$ for simplicity. \label{fig:orig_traj}](images/original_trajectory.png)

## 2. Intervention

Next, we can take exogenous control of $x$ by pinning it to $sin(t)$ and simulate the response of
 $y$. The resulting simulation describes how the behavior of the system is
 altered by this intervention. See \ref{fig:interv_effect} for an example.

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

![System Trajectory with Intervention: The figure above demonstrates the
effect that taking exogenous control of $x(t)$ by via $\text{do}(x(t)=sin(t))$
has on $y$. The intervention (black) and
response (blue), depict a clear departure from
the natural evolution behavior of the system. \label{fig:interv_effect}](images/intervention_effect.png)

## 3. Optimization

Interfere offers tools to optimize forecasting methods for time series
prediction. By using Interfere's cross validation objective function along with a
hyperparameter optimizer (Optuna), it is possible to compare multiple
hyperparameter setting on multiple folds of time series data. To simplify this
process, every Interfere forecasting method comes with sensible preset
hyperparameter ranges for the optimizer to explore.

```python

# Select the SINDy method for hyperparameter optimization.
method_type = interfere.SINDY

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

## 4. Intervention Response Prediction

Using the best parameters from the hyperparameter optimization run, we can fit
the forecasting
method to all the data that occurred prior to the intervention, treating the states we plan to manipulate as
exogenous. This way, the method expects to be given exogenous data about the
intervention variable(s). After fitting to the unperturbed system, we forecast
the intervention response by treating the desired intervention as an exogenous
input signal to include in the forecast.

```python
# Initialize SINDy with the best perfoming parameters.
method = interfere.SINDY(**study.best_params)

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

![Example of forecasting the response of the system to an intervention. Here,
the intervention consists of 
taking exogenous control of $x(t)$ (black). The ground truth response, $y(t)$ for
$t > 10$ is plotted in blue. Here, an equation discovery algorithm, SINDy
[@brunton_discovering_2016] is fit to the data that occurs prior to the
intervention, and makes an attempt to predict the intervention response (red
curve).](images/prediction_comparison.png)

# Primary Contributions

The Interfere package provides three primary contributions to the scientific community.

## 1. Dynamically Diverse Counterfactuals at Scale

The "dynamics" submodule in the interfere package contains over fifty dynamic models. It contains a mix of linear, nonlinear, chaotic, continuous time, discrete time, stochastic, and deterministic models. The models come from a variety of disciplines including economics, finance, ecology, biology, neuroscience and public health. Each model inherits the from the Interfere BaseDynamics type and gains the ability to take exogenous control of any observed state and to add measurement noise. Most models also gain the ability to make any observed state stochastic where magnitude of stochasticity can be controlled by a simple scalar parameter or fine tuned with a covariance matrix.

Because of the difficulty of building models of complex systems, predictive methods for complex dynamics are typically benchmarked on less than ten dynamical systems [@challu_nhits_2023; @brunton_discovering_2016; @vlachas_backpropagation_2020; @pathak_model-free_2018; @prasse_predicting_2022]. As such, Interfere offers a clear improvement over current benchmarking methods for prediction in complex dynamics.

Most importantly, Interfere is built around interventions: the ability to take exogenous control of the state of a complex system and observe the response. Imbuing scientific models with general exogenous control is no small feat because models can be complex and are implemented in a variety of ways. Thus Interfere offers the ability to produce multiple complex dynamic *counterfactual scenarios* at scale. This unique feature enables large scale evaluation of dynamic causal prediction methods—tested against systems with properties of interest to scientists.

## 2. Cross Disciplinary Forecast Methods

A second contribution of interfere is the integration of dynamic *forecasting* methodologies from deep learning, applied mathematics and social science. The Interfere "ForecastingMethod" class is expressive enough to describe, fit and predict with multivariate dynamic models and intervene on the states of the models during prediction. This cross disciplinary mix of techniques affords new insights into the problem of intervention response prediction.

## 3. Opening Up Intervention Response to the Scientific Community

The third major contribution of Interfere is that it poses the intervention response problem—a highly applicable question, to the broader community. The Interfere Benchmark 1.1.1 has the potential provide simple comprehensive evaluation of computational methods on the intervention response problem and therefore streamline future progress towards correctly anticipating how complex systems will respond to new scenarios.

# Related Software and Mathematical Foundations

## Predictive Methods

The Interfere package draws extensively from the Nixtla open source ecosystem for time series forecasting. Nixtla's NeuralForecast proves three of the methods that are integrated with Interfere's interface and StatsForecast provides one of the methods [@olivares2022library_neuralforecast; @garza2022statsforecast]. Nixtla also provided the inspiration for the cross validation and hyperparameter optimization workflow. Interfere also integrates with predictive methods from the PySINDy and StatsModels packages [@kaptanoglu2022; @seabold2010statsmodels]. An additional reservoir computing method for global forecasts comes from [@harding_global_2024]. Hyperparameter optimization is designed around the Optuna framework [@akiba2019optuna].

Finding forecasting methods to integrate with Interfere was difficult due to the (1) lack of multivariate dynamic forecasting methods (2) lack of dynamic methods that allow exogenous variables (3) the fact that many methods only offer a fixed forecast window do not implement recursive prediction.

## Dynamic Models

See the table below for a full list of dynamic models with attributions that are currently implemented in the interfere package. The dynamic models in were implemented directly from mathematical descriptions except for two which adapt existing simulations from the PyClustering package [@novikov2019].

| Dynamic Model Class            | Description and Source                                                                                  | Properties                     |
| ------------------------------ | ------------------------------------------------------------------------------------------------------- | ------------------------------ |
| Arithmetic Brownian Motion     | Brownian motion with linear drift and constant diffusion [@oksendal_stochastic_2005]                    | Stochastic, Linear             |
| Coupled Logistic Map           | Discrete-time logistic map with spatial coupling [@lloyd_coupled_1995]                                  | Nonlinear, Chaotic             |
| Stochastic Coupled Map Lattice | Coupled map lattice with stochastic noise [@kaneko_coupled_1991]                                        | Nonlinear, Stochastic, Chaotic |
| Michaelis Menten               | Model for enzyme kinetics and biochemical reaction networks [@srinivasan_guide_2022]                    | Nonlinear, Stochastic          |
| Lotka Voltera SDE              | Stochastic Lotka-Volterra predator-prey model [@hening_stochastic_2018]                                 | Nonlinear, Stochastic          |
| Kuramoto                       | Coupled oscillator model to study synchronization [@rodrigues_kuramoto_2016]                            | Nonlinear, Stochastic          |
| Kuramoto Sakaguchi             | Kuramoto model variant with phase frustration [@sakaguchi_soluble_1986]                                 | Nonlinear, Stochastic          |
| Hodgkin Huxley Pyclustering    | Neuron action-potential dynamics based on Hodgkin-Huxley equations [@hodgkin_quantitative_1952]         | Nonlinear                      |
| Stuart Landau Kuramoto         | Coupled oscillators with amplitude-phase dynamics [@cliff_unifying_2023]                                | Nonlinear, Stochastic          |
| Mutualistic Population         | Dynamics of interacting mutualistic species [@prasse_predicting_2022]                                   | Nonlinear                      |
| Ornstein Uhlenbeck             | Mean-reverting stochastic differential equation [@gardiner_stochastic_2009]                             | Stochastic, Linear             |
| Belozyorov 3D Quad             | 3-dimensional quadratic chaotic system [@belozyorov_exponential_2015]                                   | Nonlinear, Chaotic             |
| Liping 3D Quad Finance         | Chaotic dynamics applied in financial modeling [@liping_new_2021]                                       | Nonlinear, Chaotic             |
| Lorenz                         | Classic chaotic system describing atmospheric convection [@lorenz_deterministic_2017]                   | Nonlinear, Chaotic             |
| Rossler                        | Simplified 3D chaotic attractor system [@rossler_equation_1976]                                         | Nonlinear, Chaotic             |
| Thomas                         | Chaotic attractor with simple structure and rich dynamics [@thomas_deterministic_1999]                  | Nonlinear, Chaotic             |
| Damped Oscillator              | Harmonic oscillator with damping and noise (Classical linear model)                                     | Linear, Stochastic             |
| SIS                            | Epidemiological model (Susceptible-Infected-Susceptible) [@prasse_predicting_2022]                      | Nonlinear, Stochastic          |
| VARMA Dynamics                 | Vector AutoRegressive Moving Average for time series modeling [@hamilton_time_2020]                     | Linear, Stochastic             |
| Wilson Cowan                   | Neural mass model for neuronal population dynamics [@wilson_excitatory_1972]                            | Nonlinear                      |
| Geometric Brownian Motion      | Stochastic model widely used in financial mathematics [@black_pricing_1973]                             | Nonlinear, Stochastic          |
| Planted Tank Nitrogen Cycle    | Biochemical cycle modeling nitrogen transformation in aquatic systems [@fazio_mathematical_2006]        | Nonlinear                      |
| Generative Forecaster          | Predictive forecasting models trained on simulation, then used to generate data (Written for Interfere) | Stochastic                     |
| Standard Normal Noise          | IID noise from standard normal distribution [@cliff_unifying_2023]                                      | Stochastic                     |
| Standard Cauchy Noise          | IID noise from standard Cauchy distribution [@cliff_unifying_2023]                                      | Stochastic                     |
| Standard Exponential Noise     | IID noise from standard exponential distribution [@cliff_unifying_2023]                                 | Stochastic                     |
| Standard Gamma Noise           | IID noise from standard gamma distribution [@cliff_unifying_2023]                                       | Stochastic                     |
| Standard T Noise               | IID noise from Student’s t-distribution [@cliff_unifying_2023]                                         | Stochastic                     |

# Acknowledgements

This work was supported by NSF GRFP.

# References
