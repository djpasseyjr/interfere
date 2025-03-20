---
title: 'Interfere: Intervention Response Simulation and Prediction for Stochastic Non-Linear Dynamics'
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
    equal-contrib: true
    affiliation: 1
  - name: Peter J. Mucha
    orcid: 0000-0002-0648-7230
    equal-contrib: true
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
The vision of Interfere is simple: What if we used high quality scientific models to benchmark our causal prediction tools? When attempting to infer causal relationships from data, randomized experimental data and counterfactuals are key, but obtaining such datasets is expensive and difficult. Across many fields, like economics, neuroscience, ecology, systems biology and others, mechanistic models are developed to simulate scenarios and predict the response of systems to interventions. Because these models are painstaking calibrated with the real world, they have the ability to generate synthetic counterfactual data containing complexity characteristics of the real processes they emulate. With this vision in mind, Interfere offers the first steps towards such a vision: (1) A general interface for simulating the effect of interventions on dynamic simulation models, (2) a suite of predictive methods and cross validation tools, and (3) an initial benchmark set of dynamic counterfactual scenarios. 

# Statement of need
Over the past twenty years we've seen an emergence of multiple frameworks for identifying causal relationships [@imbens_causal_2015], [@pearl_causality_2009], [@wieczorek_information_2019]. The most influential frameworks are probabilistic and while it is not a requirement of the frameworks, in practice, a static linear relationship is usually assumed. However, when attempting to anticipate the response of complex systems in the medium and long term, static linear models are insufficient. Thus there is a need for causal models with more complexity. (For example, static linear models cannot predict scenarios where things get worse before they get better.) Currently, there are very few techniques that are able to fit non-linear dynamic causal models to data. We see an opportunity to bring the insights from recent breakthroughs in causal inference to the world of mechanistic modeling.  In order to facilitate this cross pollination, we chose a key causal problem, predicting how a complex system responds to a previously unobserved intervention in the medium and long term, and designed the Interfere package as a focal point for building and benchmarking tools that attempt to solve this problem.


![Comparing deterministic and stochastic systems.\label{fig:det_vs_stoch}](images/det_v_stoch.png)

# Acknowledgements
This work was supported by NSF GRFP.

# References