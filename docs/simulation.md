# Simulation

Simulation in Interfere brings mathematical models to life, offering a unified framework for producing deterministic, stochastic, and hybrid trajectories under controlled interventions. Whether you're exploring classic ODE behavior or complex SDE landscapes, Interfere handles the heavy lifting—time management, noise, and interventions—so you can focus on analysis.

At the heart of Interfere's simulation engine is the abstract `DynamicModel` class (in `interfere/base.py`), which provides:

- **Argument Validation**: Ensures `t`, `prior_states`, and `prior_t` shapes are consistent.
- **Time Inference**: Infers or checks historic time points (`prior_t`) if not supplied.
- **Noise Injection**: Builds and injects system (stochastic) and measurement noise.
- **Intervention Hooks**: Manages optional application of exogenous intervention functions.

Subclasses implement the protected `_simulate()` method to define the specific propagation rules.

With these foundational services—validation, time inference, noise injection, and intervention hooks—Interfere enables seamless simulation of everything from classic ODEs to complex SDEs and discrete-time maps.

In `interfere/dynamics/base.py`, three specialized simulation bases implement `_simulate()`:

- **OrdinaryDifferentialEquation**: Integrates ODEs by defining a `dXdt(x, t)` derivative function and calling `scipy.integrate.odeint`, applies interventions at each step, and adds measurement noise.
- **StochasticDifferentialEquation**: Advances SDEs using an Euler–Maruyama scheme, computing deterministic `drift(x, t)` and stochastic `noise(x, t) dW`, supports custom Wiener increments, applies interventions, and adds measurement noise.
- **DiscreteTimeDynamics**: Steps discrete-time maps by repeatedly calling a `step(x, t)` function, applying optional interventions after each step.

Below is a table of every dynamic model provided in `interfere/dynamics` (as imported in its `__init__.py`), showing each model name, its direct base class, and a concise description.

## simulate() Method

The `simulate` method is the primary interface for generating time-series trajectories of a `DynamicModel`. It handles initial conditions, optional interventions, stochastic process noise, and measurement noise, delivering a complete counterfactual or baseline trajectory.

```python
simulate(
    self,
    t: np.ndarray,
    prior_states: np.ndarray,
    prior_t: Optional[np.ndarray] = None,
    intervention: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    rng: np.random.RandomState = DEFAULT_RANGE,
    **kwargs
) -> np.ndarray
```

Parameters:

- **t** (`np.ndarray` of shape `(n,)`): Strictly increasing time points for simulation ('), requiring at least two entries for integration.
- **prior_states** (`np.ndarray`): Initial condition(s). Can be a 1D array of length `dim` or a 2D array `(p, dim)` representing `p` historic observations.
- **prior_t** (`Optional[np.ndarray]` of shape `(p,)`): Time stamps for each row of `prior_states`. If omitted and `t` is uniform, historic times are inferred as evenly spaced before `t[0]`.
- **intervention** (`Optional[Callable[[x: np.ndarray, t: float], np.ndarray]]`): A function to impose exogenous control at each step. Defaults to no intervention.
- **rng** (`numpy.random.RandomState`): RNG used for generating process (`sigma`) and measurement noise.
- **kwargs**: Additional subclass-specific options (e.g., `dW` for custom Wiener increments in SDEs).

Returns:

- **np.ndarray** `(n, dim)`: Simulated state trajectory over `t`, incorporating any interventions and noise effects.

| Model                                       | Base Class                         | Description                                          |
|---------------------------------------------|------------------------------------|------------------------------------------------------|
| `ArithmeticBrownianMotion`                  | StochasticDifferentialEquation     | Brownian motion with linear drift and constant diffusion |
| `GeometricBrownianMotion`                   | StochasticDifferentialEquation     | Geometric Brownian motion with multiplicative noise     |
| `OrdinaryDifferentialEquation`              | DynamicModel                       | Abstract base for continuous-time ODE integrators      |
| `StochasticDifferentialEquation`            | DynamicModel                       | Abstract base for SDE simulators                       |
| `LinearSDE`                                 | StochasticDifferentialEquation     | Linear stochastic differential equation                |
| `DampedOscillator`                          | StochasticDifferentialEquation     | Second-order damped oscillator modeled as SDE          |
| `imag_roots_2d_linear_sde`                  | LinearSDE (factory)                | 2D linear SDE with purely imaginary eigenvalues        |
| `imag_roots_4d_linear_sde`                  | LinearSDE (factory)                | 4D linear SDE with purely imaginary eigenvalues        |
| `attracting_fixed_point_4d_linear_sde`      | LinearSDE (factory)                | 4D SDE with an attracting fixed point                  |
| `CoupledMapLattice`                         | DiscreteTimeDynamics               | Deterministic coupled map lattice                      |
| `StochasticCoupledMapLattice`               | CoupledMapLattice                  | Coupled map lattice with stochastic fluctuations       |
| `coupled_logistic_map`                      | StochasticCoupledMapLattice (factory) | Factory: stochastic logistic map lattice          |
| `coupled_map_1dlattice_chaotic_brownian`    | StochasticCoupledMapLattice (factory) | Factory: chaotic Brownian CML                   |
| `coupled_map_1dlattice_chaotic_traveling_wave` | StochasticCoupledMapLattice (factory) | Factory: chaotic traveling-wave CML            |
| `coupled_map_1dlattice_defect_turbulence`   | StochasticCoupledMapLattice (factory) | Factory: defect turbulence CML                  |
| `coupled_map_1dlattice_frozen_chaos`        | StochasticCoupledMapLattice (factory) | Factory: frozen chaos CML                     |
| `coupled_map_1dlattice_pattern_selection`   | StochasticCoupledMapLattice (factory) | Factory: pattern selection CML                |
| `coupled_map_1dlattice_spatiotemp_chaos`    | StochasticCoupledMapLattice (factory) | Factory: spatiotemporal chaos CML               |
| `coupled_map_1dlattice_spatiotemp_intermit1`| StochasticCoupledMapLattice (factory) | Factory: intermittency type 1 CML              |
| `coupled_map_1dlattice_spatiotemp_intermit2`| StochasticCoupledMapLattice (factory) | Factory: intermittency type 2 CML              |
| `coupled_map_1dlattice_traveling_wave`      | StochasticCoupledMapLattice (factory) | Factory: traveling-wave CML                  |
| `Kuramoto`                                  | StochasticDifferentialEquation     | Coupled oscillator network                           |
| `KuramotoSakaguchi`                         | StochasticDifferentialEquation     | Kuramoto variant with phase frustration               |
| `LotkaVoltera`                              | OrdinaryDifferentialEquation       | Predator–prey ODE model                              |
| `LotkaVolteraSDE`                           | StochasticDifferentialEquation     | Stochastic predator–prey model                       |
| `MichaelisMenten`                           | StochasticDifferentialEquation     | Enzyme kinetics reaction network                     |
| `MutualisticPopulation`                     | OrdinaryDifferentialEquation       | Interacting mutualistic species model                |
| `SIS`                                       | StochasticDifferentialEquation     | Susceptible–Infected–Susceptible epidemiological model |
| `OrnsteinUhlenbeck`                         | StochasticDifferentialEquation     | Mean-reverting Ornstein–Uhlenbeck process            |
| `VARMADynamics`                             | DynamicModel                       | Vector autoregressive moving-average dynamics        |
| `WilsonCowan`                               | OrdinaryDifferentialEquation       | Wilson–Cowan neural population model                 |
| `Belozyorov3DQuad`                          | StochasticDifferentialEquation     | 3D quadratic chaotic system                          |
| `Liping3DQuadFinance`                       | StochasticDifferentialEquation     | 3D quadratic system for financial modeling           |
| `Lorenz`                                    | StochasticDifferentialEquation     | Lorenz chaotic attractor                             |
| `Rossler`                                   | StochasticDifferentialEquation     | Rössler attractor                                     |
| `Thomas`                                    | StochasticDifferentialEquation     | Thomas chaotic attractor                              |
| `MooreSpiegel`                              | StochasticDifferentialEquation     | Moore–Spiegel chaotic oscillator                      |
| `PlantedTankNitrogenCycle`                  | OrdinaryDifferentialEquation       | Aquatic nitrogen cycle model                         |
| `GenerativeForecaster`                      | DynamicModel                       | Generates trajectories using fitted forecasters      |
| `generative_lorenz_VAR_forecaster`          | GenerativeForecaster (factory)     | Factory: VAR-based Lorenz forecaster                 |
| `generative_cml_SINDy_forecaster`           | GenerativeForecaster (factory)     | Factory: SINDy-based CML forecaster                  |
| `StandardNormalNoise`                       | DiscreteTimeDynamics               | IID Gaussian noise generator                         |
| `StandardCauchyNoise`                       | DiscreteTimeDynamics               | IID Cauchy noise generator                           |
| `StandardExponentialNoise`                  | DiscreteTimeDynamics               | IID exponential noise generator                      |
| `StandardGammaNoise`                        | DiscreteTimeDynamics               | IID gamma noise generator                            |
| `StandardTNoise`                            | DiscreteTimeDynamics               | IID Student's t noise generator                      |