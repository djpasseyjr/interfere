import interfere
from interfere.dynamics.decoupled_noise_dynamics import UncorrelatedNoise
import numpy as np
import pytest
from scipy import integrate
import sdeint
from statsmodels.tsa.vector_ar.util import varsim

from interfere.dynamics import (
    StandardNormalNoise,
    StandardCauchyNoise,
    StandardExponentialNoise,
    StandardGammaNoise,
    StandardTNoise,
    coupled_map_1dlattice_chaotic_brownian,
    coupled_map_1dlattice_chaotic_traveling_wave,
    coupled_map_1dlattice_defect_turbulence,
    coupled_map_1dlattice_frozen_chaos,
    coupled_map_1dlattice_pattern_selection,
    coupled_map_1dlattice_spatiotemp_chaos,
    coupled_map_1dlattice_spatiotemp_intermit1,
    coupled_map_1dlattice_spatiotemp_intermit2,
    coupled_map_1dlattice_traveling_wave
)

COUPLED_MAP_LATTICES = [
    coupled_map_1dlattice_chaotic_brownian,
    coupled_map_1dlattice_chaotic_traveling_wave,
    coupled_map_1dlattice_defect_turbulence,
    coupled_map_1dlattice_frozen_chaos,
    coupled_map_1dlattice_pattern_selection,
    coupled_map_1dlattice_spatiotemp_chaos,
    coupled_map_1dlattice_spatiotemp_intermit1,
    coupled_map_1dlattice_spatiotemp_intermit2,
    coupled_map_1dlattice_traveling_wave
]

def check_simulate_method(
        model: interfere.base.DynamicModel,
        x0: np.ndarray = None,
    ):
    """Checks that model.simulate() runs and interventions occur appropriately.
    
    Args:
        model (interfere.base.DynamicModel): An initialized model.
        x0: Optional initial condition for simulate.
    """
    n = model.dim
    m = 1000
    rng = np.random.default_rng(10)
    if x0 is None:
        x0 = np.random.rand(n)
    t = np.linspace(0, 10, m)
    # Save original measurement noise
    orig_measurement_noise = model.measurement_noise_std

    # Adjust time scale for discrete time models
    if isinstance(model, interfere.dynamics.base.DiscreteTimeDynamics):
        t = np.arange(100)
        m = 100

    # For non noise models, add measurement noise:
    if not isinstance(model, UncorrelatedNoise):
        model.measurement_noise_std = 0.2 * np.ones(n)

    # Make intervention
    interv_idx = 0
    interv_const = 0.5
    g = interfere.perfect_intervention(interv_idx, interv_const)

    # Check output shape
    rng = np.random.default_rng(10)
    X = model.simulate(t, x0, rng=rng)
    assert X.shape == (m, n), (
        f"Output is the wrong shape"" for {model}.")

    # Check initial condition
    if x0.ndim == 1:
        assert np.allclose(X[0], x0), (
            f"Initial condition is incorrect for {model}.")
        
    elif x0.ndim == 2:
        p, _ = x0.shape
        assert np.allclose(X[:p, :], x0), (
            f"Initial condition is incorrect for {model}.")


    # Check that random state works correctly
    rng = np.random.default_rng(10)
    X_rerun = model.simulate(t, x0, rng=rng)
    assert np.all(X == X_rerun), (
        f"Random state does not preserve noise for {model}.")
    
    # Check that model is not deterministic
    X_new_realization = model.simulate(t, x0, rng=rng)
    assert not np.all(X == X_new_realization)

    # Apply an intervention
    rng = np.random.default_rng(10)
    X_do = model.simulate(t, x0, intervention=g, rng=rng)
    assert X_do.shape == (m, n), (
        f"Incorrect output size after intervention for {model}.")
    
    assert np.abs(np.mean(X_do[:, interv_idx]) - interv_const) < 0.1, (
        f"Intervention is incorrect for {model}.")

    # Make sure that random state works for interventions
    rng = np.random.default_rng(10)
    X_do_rerun = model.simulate(t, x0, intervention=g, rng=rng)
    assert np.allclose(X_do, X_do_rerun), (f"Random state does not preserve "
                                          "values after intervention for "
                                          " {model}.")
    

    # Reset to original measurement noise
    model.measurement_noise_std = orig_measurement_noise

    return X, X_do

def test_stochastic_array_builder():
    beloz = interfere.dynamics.Belozyorov3DQuad()

    # Test float argument.
    sigma = beloz.build_stochastic_noise_matrix(3)
    assert np.all(sigma == sigma * np.eye(3))
    assert sigma.shape == (beloz.dim, beloz.dim)

    # Test 1D array argument
    x = np.random.rand(3)
    sigma = beloz.build_stochastic_noise_matrix(x)
    assert np.all(sigma == np.diag(x))
    assert sigma.shape == (beloz.dim, beloz.dim)

    # Test 2D array argument
    x = np.random.rand(3, 3)
    sigma = beloz.build_stochastic_noise_matrix(x)
    assert np.all(sigma == x)
    assert sigma.shape == (beloz.dim, beloz.dim)

    with pytest.raises(
        ValueError, match="float or a 1 or 2 dimensional numpy array."):
        beloz.build_stochastic_noise_matrix(np.random.rand(3, 3, 3))

    with pytest.raises(
        ValueError, match="Pass a float or `sigma` with shape"
    ):
        beloz.build_stochastic_noise_matrix(np.random.rand(4))

    with pytest.raises(
        ValueError, match="Pass a float or `sigma` with shape"
    ):
        beloz.build_stochastic_noise_matrix(np.random.rand(2, 2))


def test_lotka_voltera():
    # Initialize interfere.LotkaVoltera model.
    n = 10
    r = np.random.rand(n)
    k = np.ones(n)
    A = np.random.rand(n, n) - 0.5

    interv_idx = n - 1
    interv_const = 1.0
    model = interfere.dynamics.LotkaVoltera(r, k, A)

    # Make two kinds of interventions
    perf_interv = interfere.perfect_intervention(interv_idx, interv_const)
    sin_interv = interfere.signal_intervention(interv_idx, np.sin)

    # Create ground truth systems with the interventions built in
    def perf_int_true_deriv(x, t):
        x[interv_idx] = interv_const
        dx = r * x * ( 1 - (x + A @ x) / k)
        dx[interv_idx] = 0.0
        return dx

    def sin_int_true_deriv(x, t):
        x[interv_idx] = np.sin(t)
        dx = r * x *( 1 - (x + A @ x) / k)
        dx[interv_idx] = np.cos(t)
        return dx

    # Set initial condition to match intervention
    x0 = np.random.rand(n)
    x0[interv_idx] = interv_const
    t = np.linspace(0, 2, 1000)

    # Test that for both interventions, the interfere API
    # correctly matches the ground truth system.
    true_perf_X = integrate.odeint(perf_int_true_deriv, x0, t)
    interfere_perf_X = model.simulate(t, x0, intervention=perf_interv)
    assert np.allclose(true_perf_X, interfere_perf_X)

    x0[interv_idx] = np.sin(t[0])
    true_sin_X = integrate.odeint(sin_int_true_deriv, x0, t)
    interfere_sin_X = model.simulate(t, x0, intervention=sin_interv)
    assert np.allclose(true_sin_X, interfere_sin_X)

    # Standard checks for intervene.base.DynamicModel objects
    model = interfere.dynamics.LotkaVolteraSDE(r, k, A, sigma=1.0)
    check_simulate_method(model)


def test_ornstein_uhlenbeck_and_sde_integrator():
    seed = 11
    rng = np.random.default_rng(seed)
    n = 3
    theta = rng.random((n, n)) - 0.5
    mu = np.ones(n)
    sigma = rng.random((n, n))- 0.5

    model = interfere.dynamics.OrnsteinUhlenbeck(theta, mu, sigma)
    # Standard checks for intervene.base.DynamicModel objects
    check_simulate_method(model)

    x0 = np.random.rand(n)
    tspan = np.linspace(0, 10, 1000)
    dt = (tspan[-1] - tspan[0]) / len(tspan)

    # Initialize the Weiner increments
    dW = np.random.normal(0, np.sqrt(dt), (len(tspan) - 1, n))

    # Check that the model.simulate API Euler Maruyama integrator is correct
    Xtrue = sdeint.itoEuler(model.drift, model.noise, x0, tspan, dW = dW)
    Xsim = model.simulate(tspan, x0, dW=dW)
    assert np.mean((Xtrue - Xsim) ** 2) < 0.01

    # Check that using the same generator corresponds exactly with sdeint
    seed = 11
    rng = np.random.default_rng(seed)
    Xtrue = sdeint.itoEuler(model.drift, model.noise, x0, tspan, generator=rng)

    seed = 11
    rng = np.random.default_rng(seed)
    Xsim = model.simulate(tspan, x0, rng=rng)

    assert np.mean((Xtrue - Xsim) ** 2) < 0.01

    # Construct parameters of the true intervened system
    theta_perf_inter = model.theta.copy()
    sigma_perf_inter = model.sigma.copy()

    theta_perf_inter[0, :] = 0
    sigma_perf_inter[0, :] = 0

    # True perfect intervention noise and drift functions
    perf_inter_drift = lambda x, t: theta_perf_inter @ (model.mu - x)
    perf_inter_noise = lambda x, t: sigma_perf_inter

    # Make the intervention function
    interv_idx = 0
    interv_const = 1.0
    intervention = interfere.perfect_intervention(interv_idx, interv_const)

    # Compare the true perfect intervention system to the true one.
    rng = np.random.default_rng(seed)
    X_perf_inter = sdeint.itoEuler(
        perf_inter_drift,
        perf_inter_noise,
        intervention(x0, 0),
        tspan,
        generator=rng,
        dW=dW
    )

    rng = np.random.default_rng(seed)
    X_perf_inter_sim = model.simulate(
        tspan,
        x0,
        intervention=intervention,
        rng=rng,
        dW=dW
    )

    # Check that the intervened variable is constant
    assert np.all(X_perf_inter_sim[:, interv_idx] == interv_const)

    # Check that the simulations match
    assert np.mean((X_perf_inter - X_perf_inter_sim) ** 2) < 0.01


def test_coupled_logistic_map():
    rng = np.random.default_rng(10)
    A = rng.random((10, 10)) < 0.5
    model = interfere.dynamics.coupled_logistic_map(A)
    # Standard checks for intervene.base.DynamicModel objects
    check_simulate_method(model)


def test_coupled_map_lattice():

    ndims = 4
    for cml in COUPLED_MAP_LATTICES:
        model = cml(ndims)
        # Standard checks for intervene.base.DynamicModel objects
        check_simulate_method(model)


def test_stochastic_coupled_map_lattice():

    ndims = 4

    for cml in COUPLED_MAP_LATTICES:
        model = cml(ndims, sigma=0.01)
        # Standard checks for intervene.base.DynamicModel objects
        check_simulate_method(model)


def test_normal_noise():
    rng = np.random.default_rng(11)
    model = interfere.dynamics.StandardNormalNoise(5)
    # Standard checks for intervene.base.DynamicModel objects
    X, X_do = check_simulate_method(model)

    # Check that the normal distribution works as expected
    assert np.all(np.abs(np.mean(X[1:, :], axis=0)) < 0.1)
    assert np.all(np.abs(np.std(X[1:, :], axis=0) - 1) < 0.1)

    # Check that the random state generated reproducible noise
    assert np.all(X_do[:, 1:] == X[:, 1:])


def test_noise_dynamics():
    noise_models = [StandardNormalNoise, StandardCauchyNoise, 
        StandardExponentialNoise, StandardGammaNoise, StandardTNoise]
    
    f = interfere.perfect_intervention(0, 30.0)
    for model_class in noise_models:
        dim = 5
        model = model_class(dim)
        # Standard checks for intervene.base.DynamicModel objects
        check_simulate_method(model)   


def test_arithmetic_brownian_motion():
    n = 1000
    m = 1000
    mu = np.ones(n) * -1
    sigma = np.ones(n) * 0.1
    model = interfere.dynamics.ArithmeticBrownianMotion(mu=mu, sigma=sigma)
    # Standard checks for intervene.base.DynamicModel objects
    X, X_do =check_simulate_method(model)

    # Make sure that the intervention makes a difference but only on one signal.
    assert np.any(X_do != X)
    assert np.all(X_do[:, 1:] == X[:, 1:])


def test_geometric_brownian_motion():
    n = 1000
    m = 1000
    mu = np.ones(n) * -1
    sigma = np.ones(n) * 0.1
    model = interfere.dynamics.GeometricBrownianMotion(mu=mu, sigma=sigma)
    
    # Standard checks for intervene.base.DynamicModel objects
    check_simulate_method(model)

    # Run additional checks

    rng = np.random.default_rng(27)
    time_points = np.linspace(0, 10, m)
    x0 = np.ones(n)
    dW = np.random.randn(m, n)
    X = model.simulate(time_points, x0, rng=rng, dW=dW)

    assert X.shape == (m, n)

    # Test that expectation matches the analytic expectation
    diff = np.mean(X, axis=1)  - (np.exp(mu[0] * time_points) * x0[0])
    assert np.all(np.abs(diff) < 0.25)

    f = interfere.perfect_intervention(0, 10)
    Xdo = model.simulate(time_points, x0, intervention=f, rng=rng, dW=dW)

    assert np.any(Xdo != X)
    assert np.all(Xdo[:, 1:] == X[:, 1:])
    assert np.all(Xdo[:, 0] == 10)

@pytest.mark.slow
def test_varma():
    seed = 1
    rs = np.random.RandomState(seed)

    # Initialize a random VAR model
    A1 = rs.rand(3, 3) - 0.5
    A2 = rs.rand(3, 3) - 0.5
    coefs = np.stack([A1, A2])
    mu = np.zeros(3)
    Z = rs.rand(3, 3)
    sigma = Z * Z.T
    steps = 101
    initial_vals = np.ones((2, 3))
    nsims = 10000

    # Simulate it
    true_var_sim = varsim(
        coefs,
        mu,
        sigma,
        steps=steps,
        initial_values=initial_vals,
        seed=seed,
        nsimulations=nsims,
    )

    # Initialize a VARMA model with no moving average component
    model = interfere.dynamics.VARMA_Dynamics(
        phi_matrices=coefs,
        theta_matrices=[np.zeros((3,3))],
        sigma=sigma
    )

    t = np.arange(steps)

    varma_sim = np.stack([
        model.simulate(t, initial_vals)
        for i in range(nsims)
    ], axis=0)
    # Average over the 10000 simulations to compute the expected trajectory.
    # Make sure it is equal for both models.
    assert np.all(
        np.abs(np.mean(true_var_sim - varma_sim, axis=0)) < 0.2
    )

def test_varima_standard_checks():
    rng = np.random.default_rng(10)
    coef_matrices = [rng.random((3, 3)) - 0.5 for i in range(5)]
    Z = rng.random((3, 3))
    sigma = Z * Z.T

    model = interfere.dynamics.VARMA_Dynamics(
        phi_matrices=coef_matrices[:2],
        theta_matrices=coef_matrices[2:],
        sigma=sigma
    )
    check_simulate_method(model, x0=rng.random((3, 3)))


def test_kuramoto():
    rng = np.random.default_rng(10)
    omega = rng.random(10)
    K = 0.7
    A = rng.random((10, 10)) < .3
    sigma=0.1
    rho = np.random.rand(10)

    # Run standard checks for interfere.base.DynamicModel objects.
    model = interfere.dynamics.Kuramoto(omega, K, A, sigma)
    check_simulate_method(model)
    model = interfere.dynamics.KuramotoSakaguchi(omega, K, A, A, sigma)
    check_simulate_method(model)
    model = interfere.dynamics.StuartLandauKuramoto(omega, rho, K, sigma)
    check_simulate_method(model)


def test_hodgkin_huxley():
    stimulus = [0, 0, 0, 15, 15, 15, 25, 25, 25, 40]
    sigma = 0.1
    model = interfere.dynamics.HodgkinHuxleyPyclustering(stimulus, sigma)
    check_simulate_method(model)


def test_lorenz():
    model = interfere.dynamics.Lorenz()
    check_simulate_method(model)


def test_rossler():
    model = interfere.dynamics.Rossler()
    check_simulate_method(model)


def test_thomas():
    model = interfere.dynamics.Thomas()
    check_simulate_method(model)


def test_aquarium():
    model = interfere.dynamics.PlantedTankNitrogenCycle()
    check_simulate_method(model)

    