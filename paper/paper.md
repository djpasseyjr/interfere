---
title: 'Interfere: Studying Intervention Response Prediction in Complex Dynamic Models'
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
    affiliation: 1
  - name: Alice C. Schwarze
    orcid: 0000-0002-9146-8068
    affiliation: 2
  - name: Zachary M. Boyd
    orcid: 0000-0002-6549-7295
    affiliation: 3
  - name: Peter J. Mucha
    orcid: 0000-0002-0648-7230
    affiliation: 2
affiliations:
  - name: University of North Carolina at Chapel Hill, United States
    index: 1
    ror: 0130frc33
  - name: Dartmouth College, United States
    index: 2
    ror: 049s0rh22
  - name: Brigham Young University, United States
    index: 3
    ror: 047rhhm47
date: 20 May 2025
bibliography: paper.bib
---
# Summary

The vision of Interfere is simple: What if we used high-quality scientific models to create causal dynamic benchmark scenarios? Randomized experimental data and intervention response time series are essential for testing methods that attempt to infer dynamic relationships from data, but obtaining such datasets can be expensive and difficult. Mechanistic models are commonly developed to simulate scenarios and predict the response of systems to interventions across economics, neuroscience, ecology, systems biology and other areas [@brayton_frbus_2014; @izhikevich_large-scale_2008; @banks_parameter_2017; @baker_mechanistic_2018]. Because these models are calibrated to the real world, they have the ability to generate diverse, complex, synthetic intervention responses that are characteristic of the real processes they emulate. Interfere offers the first steps towards this vision by combining (1) a general interface for simulating the effect of interventions on dynamic models, (2) a suite of predictive methods and cross validated hyper parameter optimization tools, and (3) the first known [extensible benchmark data set](https://drive.google.com/file/d/19_Ha-D8Kb1fFJ_iECU62eawbeuCpeV_g/view?usp=sharing) of dynamic intervention response scenarios see Figure \ref{fig:sixty_models}.

![Three-dimensional trajectories of sixty scenarios simulated with the Interfere
package. The models simulated here are either differential equations or discrete time
difference equations. For each system, the trajectory in blue represents the natural behavior of
the system and the red depicts how the system responds to a specified intervention.
Many of the models pictured have more than three dimensions (in such cases,
only the three dimensions of the trajectory with the highest variance
are shown). These sixty scenarios make up the [Interfere Benchmark 1.1.1](https://drive.google.com/file/d/19_Ha-D8Kb1fFJ_iECU62eawbeuCpeV_g/view?usp=sharing) for
intervention response prediction which is available online for download.
\label{fig:sixty_models}](../images/sixty_models.png)

# Statement of Need

Over the past twenty years, the scientific community has experienced the emergence
of multiple frameworks for identifying causal relationships in observational
data [@imbens_causal_2015; @pearl_causality_2009; @wieczorek_information_2019].
The most influential frameworks are probabilistic and, while it is not a necessary
condition for identifying causality, historically a static, linear relationship has
often been assumed. However, when attempting to anticipate the response of complex dynamic
systems in the medium and long term, a linear approximation of the dynamics can be
insufficient.  Therefore, researchers have increasingly begun to employ
non-linear, dynamic techniques for causal discovery and forecasting [e.g. @runge_discovering_2022]. Still,
there are relatively few techniques that are able to fit causal dynamic
nonlinear models to data. Because of this, we see an opportunity to bring
together the insights from recent advancements in causal inference with
historical work in dynamic modeling and simulation.

In order to facilitate this cross pollination, we focus on a key problem --- predicting how a complex system responds to a previously unobserved intervention --- and designed the Interfere package for benchmarking tools aimed at intervention response prediction. The dynamic models contained in Interfere present challenges for computational methods that can likely only be addressed with the incorporation of mechanistic assumptions alongside probabilistic frameworks for causality. The Interfere package is a toolbox that allows researcher to validate predictive dynamic methods against simulated intervention scenarios. As such, the Interfere package encourages an opportunity for cross pollination between the probabilistic causal inference community and the modeling and simulation community.

# Primary Contributions

The Interfere package provides three primary contributions. (1) Dynamically diverse counterfactuals at scale, (2) cross disciplinary forecast methods, and (3) comprehensive and extensible benchmarking.

![Example experimental setup possible with Interfere: Can stochasticity help reveal associations between variables? Interfere can be used to compare intervention response prediction for deterministic and stochastic versions of the same system.
 \label{fig:det_vs_stoch}](../images/det_v_stoch.png)

## 1. Dynamically Diverse Counterfactuals at Scale

The "dynamics" submodule in the Interfere package contains over fifty dynamic models. It contains a mix of linear, nonlinear, chaotic, continuous time, discrete time, stochastic, and deterministic models. The models come from a variety of disciplines including finance, ecology, biology, neuroscience and public health. Each model inherits the from the Interfere BaseDynamics type and gains the ability to take exogenous control of any observed state and to add measurement noise. Most models also gain the ability to make any observed state stochastic where magnitude of stochasticity can be controlled by a simple scalar parameter or fine tuned with a covariance matrix.

Because of the difficulty of building models of complex systems, predictive methods for complex dynamics are typically benchmarked on less than ten dynamical systems [@challu_nhits_2023; @brunton_discovering_2016; @vlachas_backpropagation_2020; @pathak_model-free_2018; @prasse_predicting_2022]. As such, Interfere offers a clear improvement over current benchmarking methods for prediction in complex dynamics.

Most importantly, Interfere is built around interventions: the ability to take exogenous control of one or several state variables in a complex system and observe the response. Imbuing a suite of scientific models with general exogenous control is no small feat because models can be complex and are implemented in a variety of ways. Interfere offers the ability to produce complex dynamic intervention response and standard forecasting scenarios at scale. This unique feature enables large scale evaluation of dynamic causal prediction methods—tested against systems with properties of interest to scientists. For example, we can simulate the change in concentration of ammonia based on the nitrogen cycle and an exogenous fertilizing schedule.

## 2. Cross Disciplinary Forecast Methods

A second contribution of Interfere is the integration of dynamic *forecasting* methodologies from deep learning (LSTM, NHITS), applied mathematics (SINDy, Reservoir Computers) and social science (VAR). The Interfere "ForecastingMethod" class is expressive enough to describe, fit and predict with multivariate dynamic models and apply interventions to the states of the models during prediction. This cross disciplinary mix of techniques has the potential to produce new insights into the problem of intervention response prediction among others. For example, experiments using this package have revealed that cross validation error does not correlate with well with prediction error when LSTM and NHITS attempt to predict intervention response.

## 3. Comprehensive and Extensible Benchmarking

The third major contribution of Interfere is the collection of dynamic scenarios organized into the [Interfere Benchmark](https://drive.google.com/file/d/19_Ha-D8Kb1fFJ_iECU62eawbeuCpeV_g/view?usp=sharing). The Interfere Benchmark is a comprehensive and extensible set of dynamic scenarios that are conveniently available for testing methods that predict the effects of interventions. The benchmark set contains 60 intervention response scenarios for testing, each simulated with different levels of stochastic noise. Each scenario is housed in a JSON file, complete with full metadata annotation, documentation, versioning and commit hashes marking the commit of Interfere that was used to generate the data. The scenarios were reviewed by hand with some systems exposed to exogenous input to ensure that none of the key variables settle into a steady state. Additionally, all interventions were chosen in a manner such that the response of the target variable is a significant departure from its previous behavior.

The Interfere package enables researchers from various backgrounds to systematically study the problem of predicting intervention response on simulated data from a wide range of disciplines. It thereby facilitates future progress towards correctly anticipating how complex systems will respond in new, never before seen scenarios.

# Related Software and Mathematical Foundations

## Predictive Methods

The Interfere package draws from the Nixtla open source ecosystem for time series forecasting. We implemented intervention support for LSTM and NHITS from the NeuralForecast package, and for ARIMA from the StatsForecast package [@olivares2022library_neuralforecast; @garza2022statsforecast]. We followed Nixtla's example for cross validation and hyperparameter optimization approaches. We integrated predictive methods from the PySINDy [@kaptanoglu2022] and StatsModels [@seabold2010statsmodels] packages. We also include ResComp, a reservoir computing method for global forecasts from [@harding_global_2024]. Hyperparameter optimization is designed around the Optuna framework [@akiba2019optuna].

While other forecasting methods exist, integrating a method with Interfere requires that the method is capable of (1) multivariate endogenous dynamic forecasting, (2) support for exogenous variables, and (3) support for flexible length forecast windows or recursive predictions. Few forecasting methods meet these criteria, and it is our hope that this package can encourage the development of additional methods.

## Dynamic Models

The table below list the dynamic models that are currently implemented in the Interfere package, plus attributions. These dynamic models in were implemented directly from mathematical descriptions except for two, "Hodgkin Huxley Pyclustering" and "Stuart Landau Kuramoto" which adapt existing simulations from the PyClustering package [@novikov2019].


# Acknowledgements

The work described here was supported by an NSF Graduate Research Fellowship (DJP) and by award W911NF2510049 from the Army Research Office. The content is solely the responsibility of the authors and does not necessarily represent the official views of any agency supporting this research.

# References
