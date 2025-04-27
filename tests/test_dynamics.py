import interfere
from interfere.dynamics.decoupled_noise_dynamics import UncorrelatedNoise
from interfere.base import DynamicModel

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

from interfere.dynamics.generative_forecasters import (
    generative_lorenz_VAR_forecaster,
    generative_cml_SINDy_forecaster
)

from sample_models import (
    lotka_voltera_model,
    ornstein_uhlenbeck_model,
    coupled_logistic_model,
    arithmetic_brownian_motion_model,
    geometric_brownian_motion_model,
    varima_model,
    kuramoto_model,
    kuramoto_sakaguchi_model,
    stuart_landau_kuramoto_model,
    hodgkin_huxley_model,
    michaelis_menten_model,
    mutualistic_population_model,
    sis_model,
    wilson_cowan_model
)

SEED = 11
MAX_PRIOR_OBS = 5
MEASUREMENT_NOISE_MAG = 0.1
STOCHASTIC_NOISE_MAG = 0.02

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

CML_TEST_DIM = 4
CML_MODELS = [
   cml(CML_TEST_DIM) for cml in COUPLED_MAP_LATTICES
]

NOISE_DYNAMICS = [
    StandardNormalNoise,
    StandardCauchyNoise, 
    StandardExponentialNoise,
    StandardGammaNoise,
    StandardTNoise
]

NOISE_DIM = 3
NOISE_MODELS = [
    nd(NOISE_DIM) for nd in NOISE_DYNAMICS
]
        

MODELS = [
    generative_cml_SINDy_forecaster(),
    generative_lorenz_VAR_forecaster(),
    interfere.dynamics.PlantedTankNitrogenCycle(),
    interfere.dynamics.Thomas(),
    interfere.dynamics.Rossler(),
    interfere.dynamics.Lorenz(),
    interfere.dynamics.MooreSpiegel(T=40),
    lotka_voltera_model(),
    ornstein_uhlenbeck_model(),
    coupled_logistic_model(),
    arithmetic_brownian_motion_model(),
    geometric_brownian_motion_model(),
    varima_model(),
    kuramoto_model(),
    kuramoto_sakaguchi_model(),
    stuart_landau_kuramoto_model(),
    hodgkin_huxley_model(),
    michaelis_menten_model(),
    mutualistic_population_model(),
    sis_model(),
    wilson_cowan_model(),
]

@pytest.mark.parametrize("model", MODELS + CML_MODELS)
class TestSimulate:


    def make_test_data(self, model: DynamicModel):
        """Makes test data that matches the model specs.

        Args:
            model (DynamicModel): An instance of a dynamic model
                to test.
        """
        n = model.dim
        m = 1000
        rng = np.random.default_rng(SEED)
        x0 = rng.random((MAX_PRIOR_OBS, n))
        t = np.linspace(0, 10, m)

        # Adjust time scale for discrete time models
        if isinstance(model, interfere.dynamics.base.DiscreteTimeDynamics):
            t = np.arange(100)
            m = 100

        # Make intervention
        interv_idx = 0
        interv_const = 0.1
        interv = interfere.PerfectIntervention(interv_idx, interv_const)

        return m, n, t, x0, interv, rng


    def test_output_shape(self, model: DynamicModel):
        """Tests that simulate produces the correct output shape.

        Args:
            model (DynamicModel): An instance of a dynamic model
                to test.

        Notes:
            Args are supplied via the pytest.mark.parametrize decorator.
        """
        m, n, t, x0, interv, rng = self.make_test_data(model)
        X = model.simulate(t, x0, rng=rng)

        assert X.shape == (m, n), (
            f"Output is the wrong shape for {type(model)}.")


    def test_initial_condition(self, model: DynamicModel):
        """Tests that simulate produces the correct initial condition.

        Args:
            model (DynamicModel): An instance of a dynamic model
                to test.

        Notes:
            Args are supplied via the pytest.mark.parametrize decorator.
        """
        
        m, n, t, x0, interv, rng = self.make_test_data(model)
        X = model.simulate(t, x0, rng=rng)

        if x0.ndim == 1:
            assert np.allclose(X[0], x0), (
                f"Initial condition is incorrect for {type(model)}.")
            
        elif x0.ndim == 2:
            assert np.allclose(X[0, :], x0[-1, :]), (
                f"Initial condition is incorrect for {type(model)}.")


    def test_noise_preservation(self, model: DynamicModel):
        """Tests that random state works correctly.

        Args:
            model (DynamicModel): An instance of a dynamic model
                to test.

        Notes:
            Args are supplied via the pytest.mark.parametrize decorator.
        """

        m, n, t, x0, interv, rng = self.make_test_data(model)

        # Add measurement_noise
        old_measurement_noise = model.measurement_noise_std
        model.measurement_noise_std = MEASUREMENT_NOISE_MAG *np.ones(n)

        rng = np.random.default_rng(SEED)
        X = model.simulate(t, x0, rng=rng)
        
        rng = np.random.default_rng(SEED)
        X_rerun = model.simulate(t, x0, rng=rng)

        # Remove measurement noise.
        model.measurement_noise_std = old_measurement_noise

         # Added a tolerance because Stuart Landau model has some round off
        # error. 
        assert np.max(X - X_rerun) < 1e-15, (
            f"Random state does not preserve noise for {type(model)}.")
        
    
    def test_not_deterministic(self, model: DynamicModel):
        """Tests that measurement noise is random.

        Args:
            model (DynamicModel): An instance of a dynamic model
                to test.

        Notes:
            Args are supplied via the pytest.mark.parametrize decorator.
        """
        
        m, n, t, x0, interv, rng = self.make_test_data(model)
        
        # Add measurement_noise.
        old_measurement_noise = model.measurement_noise_std
        model.measurement_noise_std = MEASUREMENT_NOISE_MAG *np.ones(n)

        X = model.simulate(t, x0, rng=rng)
    
        # Check that model is not deterministic.
        X_new_realization = model.simulate(t, x0, rng=rng)

        # Remove measurment noise.
        model.measurement_noise_std = old_measurement_noise

        assert not np.all(X == X_new_realization), (
            f"{type(model)} has deterministic measurement noise."
        )

    
    def test_intervention(self, model: DynamicModel):
        """Tests that interventions function correctly.

        Args:
            model (DynamicModel): An instance of a dynamic model
                to test.

        Notes:
            Args are supplied via the pytest.mark.parametrize decorator.
        """
        
        m, n, t, x0, interv, rng = self.make_test_data(model)

        # Add measurement_noise.
        old_measurement_noise = model.measurement_noise_std
        model.measurement_noise_std = MEASUREMENT_NOISE_MAG *np.ones(n)

        # Apply an intervention.
        rng = np.random.default_rng(SEED)
        X_do = model.simulate(t, x0, intervention=interv, rng=rng)

        # Remove measurement noise.
        model.measurement_noise_std = old_measurement_noise

        assert X_do.shape == (m, n), (
            f"Incorrect output size after intervention for {type(model)}.")
        
        _, interv_states = interv.split_exog(X_do)
        assert np.isclose(
            np.mean(interv_states), interv.constants[0], atol=0.1), (
            f"Intervention is incorrect for {type(model)}.")


    def test_intervention_random_state(self, model: DynamicModel):
        """Tests that intervention function works with random noise.

        Args:
            model (DynamicModel): An instance of a dynamic model
                to test.

        Notes:
            Args are supplied via the pytest.mark.parametrize decorator.
        """

        m, n, t, x0, interv, rng = self.make_test_data(model)

        # Add measurement_noise.
        old_measurement_noise = model.measurement_noise_std
        model.measurement_noise_std = MEASUREMENT_NOISE_MAG *np.ones(n)

        # Apply an intervention
        rng = np.random.default_rng(SEED)
        X_do = model.simulate(t, x0, intervention=interv, rng=rng)

        # Make sure that random state works for interventions
        rng = np.random.default_rng(SEED)
        X_do_rerun = model.simulate(t, x0, intervention=interv, rng=rng)

        # Remove measurement noise.
        model.measurement_noise_std = old_measurement_noise

        assert np.allclose(X_do, X_do_rerun), (
            "Random state does not preserve values after intervention for "
            f" {type(model)}."
        )       


    def test_is_stochastic(self, model: DynamicModel):
        """Tests that model is stochastic.

        Args:
            model (DynamicModel): An instance of a dynamic model
                to test.

        Notes:
            Args are supplied via the pytest.mark.parametrize decorator.
        """
        m, n, t, x0, interv, rng = self.make_test_data(model)

        # Turn off measurement_noise.
        old_measurement_noise = model.measurement_noise_std
        model.measurement_noise_std = np.zeros(n)

        # Add stochastic matrix.
        old_sigma = model.sigma
        model.sigma = STOCHASTIC_NOISE_MAG * np.eye(n)

        X = model.simulate(t, x0, rng=rng)
        X_rerun = model.simulate(t, x0, rng=rng)

        # Remove stochastic matrix.
        model.sigma = old_sigma
        model.measurement_noise_std = old_measurement_noise

        assert not np.all(X == X_rerun), (
            f"{type(model)} is not behaving stochastically."
        )


    def test_range_preserves_stochastic(self, model: DynamicModel):
        """Tests that range preserves stochasticity.

        Args:
            model (DynamicModel): An instance of a dynamic model
                to test.

        Notes:
            Args are supplied via the pytest.mark.parametrize decorator.
        """
        m, n, t, x0, interv, rng = self.make_test_data(model)

        # Turn off measurement_noise.
        old_measurement_noise = model.measurement_noise_std
        model.measurement_noise_std = np.zeros(n)

        # Add stochastic matrix.
        old_sigma = model.sigma
        model.sigma = STOCHASTIC_NOISE_MAG * np.eye(n)

        rng = np.random.default_rng(SEED)
        X = model.simulate(t, x0, rng=rng)

        rng = np.random.default_rng(SEED)
        X_rerun = model.simulate(t, x0, rng=rng)

        # Remove stochastic matrix.
        model.sigma = old_sigma
        model.measurement_noise_std = old_measurement_noise

        # Added a tolerance because Stuart Landau model has some round off
        # error. 
        assert np.max(X - X_rerun) < 1e-15, (
            f"{type(model)} is not generating reproducible stochasticity."
        )


    def test_intervention_is_stochastic(self, model: DynamicModel):
        """Tests that model is stochastic.

        Args:
            model (DynamicModel): An instance of a dynamic model
                to test.

        Notes:
            Args are supplied via the pytest.mark.parametrize decorator.
        """
        m, n, t, x0, interv, rng = self.make_test_data(model)

        # Turn off measurement_noise.
        old_measurement_noise = model.measurement_noise_std
        model.measurement_noise_std = np.zeros(n)

        # Add stochastic matrix.
        old_sigma = model.sigma
        model.sigma = STOCHASTIC_NOISE_MAG * np.eye(n)

        X = model.simulate(t, x0, intervention=interv, rng=rng)
        X_rerun = model.simulate(t, x0, intervention=interv, rng=rng)

        # Remove stochastic matrix.
        model.sigma = old_sigma
        model.measurement_noise_std = old_measurement_noise

        assert not np.all(X == X_rerun), (
            f"{type(model)} is not behaving stochastically with interventions."
        )

    
    def test_stochastic_init(self, model: DynamicModel):
        """Tests that the stochastic matrix initializer works correctly."""
        n = model.dim
        true_sigma = np.eye(n)

        for sigma in [1, np.ones(n), np.eye(n)]:
            # Reinitialize model with the same args but change sigma
            ini = model.__init__

            # Grab arg names.
            kwargs = {
                kw: getattr(model, kw)
                for kw in ini.__code__.co_varnames[1:ini.__code__.co_argcount]
                if hasattr(model, kw)
            }

            new_model = type(model)(**{**kwargs, "sigma": sigma})
            assert np.all(new_model.sigma == true_sigma), (
                f"Stochastic initialization failed for {type(model).__name__}")


def test_stochastic_array_builder():
    beloz = interfere.dynamics.Belozyorov3DQuad()

    # Test float argument.
    sigma = beloz.build_stochastic_noise_matrix(3, 3)
    assert np.all(sigma == sigma * np.eye(3))
    assert sigma.shape == (beloz.dim, beloz.dim)

    # Test 1D array argument
    x = np.random.rand(3)
    sigma = beloz.build_stochastic_noise_matrix(x, 3)
    assert np.all(sigma == np.diag(x))
    assert sigma.shape == (beloz.dim, beloz.dim)

    # Test 2D array argument
    x = np.random.rand(3, 3)
    sigma = beloz.build_stochastic_noise_matrix(x, 3)
    assert np.all(sigma == x)
    assert sigma.shape == (beloz.dim, beloz.dim)

    with pytest.raises(
        ValueError, match="float or a 1 or 2 dimensional numpy array."):
        beloz.build_stochastic_noise_matrix(np.random.rand(3, 3, 3), 3)

    with pytest.raises(
        ValueError, match="Pass a float or `sigma` with shape"
    ):
        beloz.build_stochastic_noise_matrix(np.random.rand(4), 3)

    with pytest.raises(
        ValueError, match="Pass a float or `sigma` with shape"
    ):
        beloz.build_stochastic_noise_matrix(np.random.rand(2, 2), 3)


def test_lotka_voltera():
    """Tests that interfere.dynamics.LotkaVoltera simulation corresponds to a
    ground truth ODE with interventions built in.
    """
    # Initialize interfere.LotkaVoltera model.
    rng = np.random.default_rng(SEED)
    n = 10
    r = rng.random(n)
    k = np.ones(n)
    A = rng.random((n, n)) - 0.5

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
    x0 = rng.random(n)
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


def test_ornstein_uhlenbeck_and_sde_integrator():
    """Uses Ornstein Uhlenback model to test interfere's SDE inegrator against
    the sdeint integrator.
    """
    seed = 11
    rng = np.random.default_rng(seed)
    n = 3
    theta = rng.random((n, n)) - 0.5
    mu = np.ones(n)
    sigma = rng.random((n, n))- 0.5

    model = interfere.dynamics.OrnsteinUhlenbeck(theta, mu, sigma)

    x0 = rng.random(n)
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


def test_normal_noise():
    """Tests that the normal distribution parameters are correct and that the
    generated noise is reproducible.
    """
    model = interfere.dynamics.StandardNormalNoise(5)
    m, n, t, x0, interv, rng = TestSimulate().make_test_data(model)

    X = model.simulate(t, x0, rng=np.random.default_rng(SEED))
    X_do = model.simulate(t, x0, rng=np.random.default_rng(SEED))

    # Check that the normal distribution works as expected
    assert np.all(np.abs(np.mean(X[1:, :], axis=0)) < 0.1)
    assert np.all(np.abs(np.std(X[1:, :], axis=0) - 1) < 0.1)

    # Check that the random state generated reproducible noise
    assert np.all(X_do[:, 1:] == X[:, 1:])


def test_geometric_brownian_motion():
    """Tests that simulated expectation matches analytic expectation for
    geometric brownian motion.
    """
    
    n = 1000
    m = 1000
    mu = np.ones(n) * -1
    sigma = np.ones(n) * 0.1
    model = interfere.dynamics.GeometricBrownianMotion(mu=mu, sigma=sigma)

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
    """Tests that multiple simulations of the interfere VARMA matches
    statsmodels.tsa.vector_ar.util.varsim when averaged.
    """
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
    lags = 2
    initial_vals = np.ones((lags, 3))
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
    model = interfere.dynamics.VARMADynamics(
        phi_matrices=coefs,
        theta_matrices=[np.zeros((3,3))],
        sigma=sigma
    )

    t = np.arange(steps - lags + 1)

    varma_sim = np.stack([
        model.simulate(t, initial_vals)
        for i in range(nsims)
    ], axis=0)
    # Average over the 10000 simulations to compute the expected trajectory.
    # Make sure it is equal for both models.
    assert np.all(
        np.abs(np.mean(true_var_sim[:, lags - 1:, :] - varma_sim, axis=0)) < 0.2
    ), ("Average of interfere VARMA model runs does not match statsmodels sim.")