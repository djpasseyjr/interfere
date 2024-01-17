import interfere
import numpy as np
from scipy import integrate
import sdeint

from interfere.dynamics import (
    StandardNormalNoise,
    StandardCauchyNoise,
    StandardExponentialNoise,
    StandardGammaNoise,
    StandardTNoise
)

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
        dx = r * x *( 1 - x / k + A @ (x / k))
        dx[interv_idx] = 0.0
        return dx

    def sin_int_true_deriv(x, t):
        x[interv_idx] = np.sin(t)
        dx = r * x *( 1 - x / k + A @ (x / k))
        dx[interv_idx] = np.cos(t)
        return dx

    # Set initial condition to match intervention
    x0 = np.random.rand(n)
    x0[interv_idx] = interv_const
    t = np.linspace(0, 2, 1000)

    # Test that for both interventions, the interfere API
    # correctly matches the ground truth system.
    true_perf_X = integrate.odeint(perf_int_true_deriv, x0, t)
    interfere_perf_X = model.simulate(x0, t, perf_interv)
    assert np.allclose(true_perf_X, interfere_perf_X)

    x0[interv_idx] = np.sin(t[0])
    true_sin_X = integrate.odeint(sin_int_true_deriv, x0, t)
    interfere_sin_X = model.simulate(x0, t, sin_interv)
    assert np.allclose(true_sin_X, interfere_sin_X)


def test_ornstein_uhlenbeck():
    seed = 11
    rng = np.random.default_rng(seed)
    n = 3
    theta = rng.random((n, n)) - 0.5
    mu = np.ones(n)
    sigma = rng.random((n, n))- 0.5

    model = interfere.dynamics.OrnsteinUhlenbeck(theta, mu, sigma)

    x0 = np.random.rand(n)
    tspan = np.linspace(0, 10, 1000)
    dt = (tspan[-1] - tspan[0]) / len(tspan)

    # Initialize the Weiner increments
    dW = np.random.normal(0, np.sqrt(dt), (len(tspan) - 1, n))

    # Check that the model.simulate API Euler Maruyama integrator is correct
    Xtrue = sdeint.itoEuler(model.drift, model.noise, x0, tspan, dW = dW)
    Xsim = model.simulate(x0, tspan, dW=dW)
    assert np.mean((Xtrue - Xsim) ** 2) < 0.01

    # Check that using the same generator corresponds exactly with sdeint
    seed = 11
    rng = np.random.default_rng(seed)
    Xtrue = sdeint.itoEuler(model.drift, model.noise, x0, tspan, generator=rng)

    seed = 11
    rng = np.random.default_rng(seed)
    Xsim = model.simulate(x0, tspan, rng=rng)

    assert np.mean((Xtrue - Xsim) ** 2) < 0.01

    # Construct parameters of the true intervened system
    theta_perf_inter = model.theta.copy()
    sigma_perf_inter = model.Sigma.copy()

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
        x0,
        tspan,
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
    model = interfere.dynamics.CoupledLogisticMaps(A)
    x0 = rng.random(10)
    t = range(100)
    X = model.simulate(x0, t)
    assert X.shape == (100, 10)

    # Make an intervention function
    interv_idx = 0
    interv_const = 1.0
    intervention = interfere.perfect_intervention(interv_idx, interv_const)
    X_do = model.simulate(x0, t, intervention)
    assert np.all(X_do[:, interv_idx] == interv_const)

def test_normal_noise():
    rng = np.random.default_rng(11)
    model = interfere.dynamics.StandardNormalNoise(5)
    X = model.simulate(np.ones(5), np.arange(1000), rng=rng)

    # Check initial condition
    assert np.all(X[0] == 1)

    # Check that the normal distribution works as expected
    assert np.all(np.abs(np.mean(X[1:, :], axis=0)) < 0.1)
    assert np.all(np.abs(np.std(X[1:, :], axis=0) - 1) < 0.1)

    f = interfere.perfect_intervention(0, 30.0)
    rng = np.random.default_rng(11)
    X_do = model.simulate(np.ones(5), np.arange(1000), f, rng=rng)

    # Check that the intervention was applied
    assert np.all(X_do[:, 0] == 30.0)

    # Check that the random state generated reproducible noise
    assert np.all(X_do[:, 1:] == X[:, 1:])


def test_noise_dynamics():
    noise_models = [StandardNormalNoise, StandardCauchyNoise, 
        StandardExponentialNoise, StandardGammaNoise, StandardTNoise]
    
    f = interfere.perfect_intervention(0, 30.0)
    for model_class in noise_models:
        dim = 5
        model = model_class(dim)

        rng = np.random.default_rng(11)
        X_do = model.simulate(np.ones(dim), np.arange(1000), f, rng=rng)

        # Check that the intervention was applied
        assert np.all(X_do[:, 0] == 30.0)

        # Check that the array is the right size
        assert X_do.shape == (1000, dim)

        # Check that random state is working properly
        rng = np.random.default_rng(11)
        X_do2 = model.simulate(np.ones(dim), np.arange(1000), f, rng=rng)
        assert np.all(X_do == X_do2)


def test_arithmetic_brownian_motion():
    n = 1000
    m = 1000
    mu = np.ones(n) * -1
    sigma = np.ones(n) * 0.1
    model = interfere.dynamics.ArithmeticBrownianMotion(mu=mu, sigma=sigma)

    rng = np.random.default_rng(27)
    time_points = np.linspace(0, 10, m)
    x0 = np.ones(n)
    dW = np.random.randn(m, n)
    X = model.simulate(x0, time_points, rng=rng, dW=dW)

    assert X.shape == (m, n)

    # Test that expectation matches the analytic expectation
    diff = np.mean(X, axis=1)  - (mu[0] * time_points + x0[0])
    assert np.all(np.abs(diff) < 0.2)

    f = interfere.perfect_intervention(0, 10)
    Xdo = model.simulate(x0, time_points, f, rng=rng, dW=dW)

    assert np.any(Xdo != X)
    assert np.all(Xdo[:, 1:] == X[:, 1:])
    assert np.all(Xdo[:, 0] == 10)


def test_geometric_brownian_motion():
    n = 1000
    m = 1000
    mu = np.ones(n) * -1
    sigma = np.ones(n) * 0.1
    model = interfere.dynamics.GeometricBrownianMotion(mu=mu, sigma=sigma)

    rng = np.random.default_rng(27)
    time_points = np.linspace(0, 10, m)
    x0 = np.ones(n)
    dW = np.random.randn(m, n)
    X = model.simulate(x0, time_points, rng=rng, dW=dW)

    assert X.shape == (m, n)

    # Test that expectation matches the analytic expectation
    diff = np.mean(X, axis=1)  - (np.exp(mu[0] * time_points) * x0[0])
    assert np.all(np.abs(diff) < 0.2)

    f = interfere.perfect_intervention(0, 10)
    Xdo = model.simulate(x0, time_points, f, rng=rng, dW=dW)

    assert np.any(Xdo != X)
    assert np.all(Xdo[:, 1:] == X[:, 1:])
    assert np.all(Xdo[:, 0] == 10)