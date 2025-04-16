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
The vision of Interfere is simple: What if we used high quality scientific models to benchmark causal prediction tools? Randomized experimental data and counterfactuals are essential for testing methods that attempt to infer causal relationships from data, but obtaining such datasets can be expensive and difficult. Mechanistic models have been developed to simulate scenarios and predict the response of systems to interventions across economics, neuroscience, ecology, systems biology and other areas [@brayton_frbus_2014; @izhikevich_large-scale_2008; @banks_parameter_2017; @baker_mechanistic_2018] **(If this citation collection formats right, do the same to others)** . Because these models are painstaking calibrated with the real world, they have the ability to generate diverse and complex synthetic counterfactual data that are characteristic of the real processes they emulate. Interfere offers the first steps towards implementing a vision that leverages such models to test causal prediction tools, combining (1) a general interface for simulating the effect of interventions on dynamic simulation models, (2) a suite of predictive methods and cross validation tools, and (3) an initial benchmark set of dynamic counterfactual scenarios. 

![Three dimensional trajectories of forty scenarios simulated with the Interfere package. Many of the models pictured have more than three dimensions (in such cases, only the three components of the trajectory with highest variance are shown). **(I'm going to add intervention response trajectories to this)** \label{fig:fourty_models}](images/forty_models.png)

# Statement of Need
Over the past twenty years there has been an emergence of multiple frameworks for identifying causal relationships in observational data [@imbens_causal_2015], [@pearl_causality_2009], [@wieczorek_information_2019]. The most influential frameworks are probabilistic and, while is not a requirement of all frameworks, in practice a linear relationship is often assumed [@runge_discovering_2022]. However, when attempting to anticipate the response of complex systems in the medium and long term, linear models are insufficient. For example, static linear models cannot predict scenarios where things get worse before they get better. However, there are relatively few techniques that are able to fit causal dynamic nonlinear models to data. Because of this, we see an opportunity to bring together the insights from recent breakthroughs in causal inference with the descriptive power of mechanistic modeling.  

In order to facilitate this cross pollination, we focus on a key causal problem --- predicting how a complex system responds to a previously unobserved intervention --- **(double check em dashes rendered correctly)** and designed the Interfere package for benchmarking tools aimed at intervention response prediction. The dynamic models contained in Interfere present challenges for causal inference that can likely only be addressed with the incorporation of mechanistic assumptions. As such, the Interfere package creates a much-needed link between the causal inference community and mechanistic modeling community. **(Add cookbook, pedantic description of what the problem is and what exactly we want to do so that a CS person can understand.) (Think about adding Forward Euler)**

![Example experimental setup possible with Interfere: Comparing intervention response prediction for deterministic and stochastic systems.\label{fig:det_vs_stoch}](images/det_v_stoch.png)


# Primary Contributions

The Interfere package provides three primary contributions to the scientific community.

## 1. Dynamically Diverse Counterfactuals at Scale

The "dynamics" submodule in the interfere package contains over fifty dynamic models. It contains a mix of linear, nonlinear, chaotic, continuous time, discrete time, stochastic, and deterministic models. The models come from a variety of diciplines including economics, finance, ecology, biology, neuroscience and public health. Each model inherits the from the Interfere BaseDynamics type and gains the ability to take exogenous control of any observed state and to add measurement noise. Most models also gain the ability to make any observed state stochastic where magnitude of stochasticity can be controlled by a simple scalar parameter or fine tuned with a covariance matrix.

Because of the difficulty of building models of complex systems, predictive methods for complex dynamics are typically benchmarked on less than ten dynamical systems [@challu_nhits_2023], [@brunton_discovering_2016], [@vlachas_backpropagation_2020], [@pathak_model-free_2018], [@prasse_predicting_2022]. As such, Interfere offers a clear improvement over current benchmarking methods for prediction in complex dynamics.

Most importantly, Interfere is built around interventions: the ability to take exogenous control of the state of a complex system and observe the response. Imbuing scientific models with general exogenous control is no small feat because models can be complex and are implemented in a variety of ways. Thus Interfere offers the ability to produce multiple complex dynamic *counterfactual scenarios* at scale. This unqiue feature enables large scale evaluation of dynamic causal prediction methods—tested against systems with properties of interest to scientists.

## 2. Cross Disciplinary Forecast Methods

A second contribution of interfere is the integration of dynamic *forecasting* methodologies from deep learning, applied mathematics and social science. The Interere "ForecastingMethod" class is expressive enough to describe, fit and predict with multivariate dynamic models and intervene on the states of the models during prediction. This cross diciplinary mix of techniques affords new insights into the problem of intervention response prediction.

## 3. Opening Up Intervention Response to the Scientific Community

The third major contribution of Interfere is that it poses the intervention response problem—a highly applicable question, to the broader community. The Interfere Benchmark 1.0.0 has the potential provide simple comprehensive evaluation of computational methods on the intervention response problem and therefore streamline future progress towards correctly anticipating how complex systems will respond to new scenarios.

# Usage

The Interfere package is designed around three tasks: Counterfactual simulation, predictive method optimization and prediction. **(Break up code blocks. Describe all the details at each step.)**

## 1. Counterfactual Simulation of Intervention Response
The following code contains and example of counterfactual intervention response simulation.

```python
import numpy as np
import interfere
import optuna

initial_cond = np.random.rand(3)
t_train = np.arange(0, 10, 0.05)
dynamic_model = interfere.dynamics.Belozyorov3DQuad()
# Observation Period.
Y = dynamic_model.simulate(t_train, initial_cond)
# Forecasting period.
t_test = np.arange(t_train[-1], 12, 0.05)
# Dynamic treatment do(x1(t) = sin(t))
interv = interfere.SignalIntervention(np.sin, 1)
Y_treat = dynamic_model.simulate(t_test, Y, intervention=interv)
# Counterfactual
Y_cntr = dynamic_model.simulate(t_test, initial_cond)
```

## 2. Cross Validation and Hyperparameter Optimization
We can fit a method to the observation period generated in the previous section using Interfere's cross validation objective function along with a hyperparameter optimizer (Optuna). Every Interfere method comes with preset hyperparameter ranges to explore.

```python
method_type = interfere.VAR
cv_objv = interfere.CrossValObjective(
        method_type=method_type,
        data=Y,
        times=t_train,
        train_window_percent=0.3,
        num_folds=5,
        exog_idxs=interv.intervened_idxs,
)
study = optuna.create_study(name="Interfere Demo Study")
study.optimize(cv_objv, ntrials=10)

params = study.best_params
```

## 3. Intervention Response Prediction

Using the best parameters from the hyperparameter optimization run, we can fit a method to the observation data, treating the states we plan to manipulate as exogenous. We then supply an exogenous signal to the method and forecast a response.

```python
method = method_type(**params)
Y_endog, Y_exog = interv.split_exog(Y)
method.fit(t_train, Y_endog, Y_exog)

# Simulate intervention response.
pred_Y_treat = method.simulate(
    t_test,
    prior_states=Y,
    intervention=interv
)
```


# Related Software and Mathematical Foundations

## Predictive Methods
The Interfere package draws extensively on the Nixtla open source ecosystem for time series forecasting. Nixtla's NeuralForecast proves three of the methods that are integrated with Interfere's interface and StatsForecast provides one of the methods [@olivares2022library_neuralforecast], [@garza2022statsforecast]. Nixtla also provided the inspiration for the cross validation and hyperparameter optimization workflow. Interfere also integrates with predictive methods from the PySINDy and StatsModels packages [@kaptanoglu2022], [@seabold2010statsmodels]. An additional reservoir computing method for global forecasts comes from [@harding_global_2024]. Hyperparameter optimization is designed around the Optuna framework [@akiba2019optuna].

Finding forecasting methods to integrate with Interfere was difficult due to the (1) lack of multivariate dynamic forecasting methods (2) lack of dynamic methods that allow exogenous variables (3) the fact that many methods only offer a fixed forecast window do not implement recursive prediction.

## Dynamic Models 

See the table below for a full list of dynamic models with attributions that are currently implemented in the interfere package. The dynamic models in were implemented directly from mathematical descriptions except for two which adapt existing simulations from the PyClustering package [@novikov2019].

| Dynamic Model Class                 | Description and Source                                                                                       | Properties                            |
|------------------------------------|---------------------------------------------------------------------------------------------------------------|----------------------------------------|
| ArithmeticBrownianMotion           | Brownian motion with linear drift and constant diffusion [@oksendal_stochastic_2005]                          | Stochastic, Linear                     |
| Coupled Logistic Map               | Discrete-time logistic map with spatial coupling [@lloyd_coupled_1995]                                       | Nonlinear, Chaotic                    |
| StochasticCoupledMapLattice        | Coupled map lattice with stochastic noise [@kaneko_coupled_1991]                                              | Nonlinear, Stochastic, Chaotic        |
| MichaelisMenten                    | Model for enzyme kinetics and biochemical reaction networks [@srinivasan_guide_2022]                          | Nonlinear, Stochastic                 |
| LotkaVolteraSDE                    | Stochastic Lotka-Volterra predator-prey model [@hening_stochastic_2018]                                      | Nonlinear, Stochastic                 |
| Kuramoto                           | Coupled oscillator model to study synchronization [@rodrigues_kuramoto_2016]                                 | Nonlinear, Stochastic                 |
| KuramotoSakaguchi                  | Kuramoto model variant with phase frustration [@sakaguchi_soluble_1986]                                      | Nonlinear, Stochastic                 |
| HodgkinHuxleyPyclustering          | Neuron action-potential dynamics based on Hodgkin-Huxley equations [@hodgkin_quantitative_1952]              | Nonlinear                             |
| StuartLandauKuramoto               | Coupled oscillators with amplitude-phase dynamics [@cliff_unifying_2023]                                     | Nonlinear, Stochastic                 |
| MutualisticPopulation              | Dynamics of interacting mutualistic species [@prasse_predicting_2022]                                        | Nonlinear                             |
| OrnsteinUhlenbeck                  | Mean-reverting stochastic differential equation [@gardiner_stochastic_2009]                                  | Stochastic, Linear                    |
| Belozyorov3DQuad                   | 3-dimensional quadratic chaotic system [@belozyorov_exponential_2015]                                        | Nonlinear, Chaotic                    |
| Liping3DQuadFinance                | Chaotic dynamics applied in financial modeling [@liping_new_2021]                                            | Nonlinear, Chaotic                    |
| Lorenz                             | Classic chaotic system describing atmospheric convection [@lorenz_deterministic_2017]                        | Nonlinear, Chaotic                    |
| Rossler                            | Simplified 3D chaotic attractor system [@rossler_equation_1976]                                              | Nonlinear, Chaotic                    |
| Thomas                             | Chaotic attractor with simple structure and rich dynamics [@thomas_deterministic_1999]                       | Nonlinear, Chaotic                    |
| DampedOscillator                   | Harmonic oscillator with damping and noise (Classical linear model)                                          | Linear, Stochastic                    |
| SIS                                | Epidemiological model (Susceptible-Infected-Susceptible) [@prasse_predicting_2022]                           | Nonlinear, Stochastic                 |
| VARMADynamics                      | Vector AutoRegressive Moving Average for time series modeling [@hamilton_time_2020]                          | Linear, Stochastic                    |
| WilsonCowan                        | Neural mass model for neuronal population dynamics [@wilson_excitatory_1972]                                 | Nonlinear                             |
| GeometricBrownianMotion            | Stochastic model widely used in financial mathematics [@black_pricing_1973]                                  | Nonlinear, Stochastic                 |
| PlantedTankNitrogenCycle           | Biochemical cycle modeling nitrogen transformation in aquatic systems [@fazio_mathematical_2006]             | Nonlinear                             |
| GenerativeForecaster               | Predictive forecasting models trained on simulation, then used to generate data (Written for Interfere)      | Stochastic                            |
| StandardNormalNoise                | IID noise from standard normal distribution [@cliff_unifying_2023]                                           | Stochastic                            |
| StandardCauchyNoise                | IID noise from standard Cauchy distribution [@cliff_unifying_2023]                                           | Stochastic                            |
| StandardExponentialNoise           | IID noise from standard exponential distribution [@cliff_unifying_2023]                                      | Stochastic                            |
| StandardGammaNoise                 | IID noise from standard gamma distribution [@cliff_unifying_2023]                                            | Stochastic                            |
| StandardTNoise                     | IID noise from Student’s t-distribution [@cliff_unifying_2023]                                               | Stochastic                            |

# Acknowledgements
This work was supported by NSF GRFP.

# References