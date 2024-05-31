# interfere.py

***When can we predict how complex systems will respond to interventions?***

This package contains classes and functions for modeling and predicting the response of
dynamic nonlinear multivariate stochastic systems to unobserved exogenous interventions. At
its essence, this is question about causality: how will a dynamic system react to
new scenarios? The code here is designed to study and explore this question.

For example, the API defined in this package was used to simulate the effect of
stochasticity on the ability to forecast intervention response as described in
the following figure.

![ (Figure of stochastic and deterministic systems responding to interventions.)](https://github.com/djpasseyjr/interfere/blob/c7090043aec4a984a45517794d266df4eb105f79/images/det_v_stoch.png?raw=true)

## Install

To install, run

```
pip install git+https://github.com/djpasseyjr/interfere

```

Alternatively, clone the repo navigate inside and run

```
pip install .
```

## Overview

In ecology, economics, social sciences, systems biology and many other disciplines, we are faced with the question, *how should we intervene in order to produce a desired outcome?*. The systems where we wish to intervene often exhibit nonlinearity, feedback, time delays and accumulation that combine to produce behavior that is difficult to predict.

This becomes most poigniant when the interventions we are curious about have never been observed. It leads us to the question, *what can the observed data tell us (if anything) about how the system will respond in a new scenario?*

This repository contains dynamic models with intervention machinery built in allowing users to generate *counterfactual pairs*, (1) the natural, observational time series and (2) the counterfactual, *what would have happened* if a certain intervention was applied.

In connection with this, this repository contains inference methods that attempt use the observational data to predict the system's response to the intervention.
