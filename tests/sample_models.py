"""Contains sample models of many kinds for testing.
"""

import interfere
import numpy as np
SEED = 10
RNG = np.random.default_rng(SEED)


def lotka_voltera_model(
) -> interfere.dynamics.LotkaVolteraSDE:
    """Creates a lotka-voltera model.
    
    Returns:
        An instance of interfere.dynamics.LotkaVolteraSDE.
    """
    n = 10
    r = RNG.random(n)
    k = np.ones(n)
    A = RNG.random((n, n)) - 0.5
    model = interfere.dynamics.LotkaVolteraSDE(r, k, A, sigma=1.0)
    return model


def ornstein_uhlenbeck_model(
) -> interfere.dynamics.OrnsteinUhlenbeck:
    """Creates an ornstein uhlenbeck model.
    
    Returns:
        An instance of interfere.dynamics.OrnsteinUhlenbeck.
    """
    n = 3
    theta = RNG.random((n, n)) - 0.5
    mu = np.ones(n)
    sigma = RNG.random((n, n))- 0.5

    model = interfere.dynamics.OrnsteinUhlenbeck(theta, mu, sigma)
    return model


def coupled_logistic_model(
) -> interfere.dynamics.StochasticCoupledMapLattice:
    """Creates an coupled logistic map model.
    
    Returns:
        An instance of interfere.dynamics.StochasticCoupledMapLattice.
    """
    
    A = RNG.random((10, 10)) < 0.5
    model = interfere.dynamics.coupled_logistic_map(A)
    return model


def arithmetic_brownian_motion_model(
) -> interfere.dynamics.ArithmeticBrownianMotion:
    """Creates an arithmetic brownian motion model.
    
    Returns:
        An instance of interfere.dynamics.ArithmeticBrownianMotion.
    """
    n = 1000
    m = 1000
    mu = np.ones(n) * -1
    sigma = np.ones(n) * 0.1
    model = interfere.dynamics.ArithmeticBrownianMotion(mu=mu, sigma=sigma)
    return model


def geometric_brownian_motion_model(
) -> interfere.dynamics.GeometricBrownianMotion:
    """Creates a geometric brownian motion model.
    
    Returns:
        An instance of interfere.dynamics.GeometricBrownianMotion.
    """
    n = 1000
    m = 1000
    mu = np.ones(n) * -1
    sigma = np.ones(n) * 0.1
    model = interfere.dynamics.GeometricBrownianMotion(mu=mu, sigma=sigma)
    return model


def varima_model() -> interfere.dynamics.VARMADynamics:
    """Initializes a VARIMA model.
    
    Returns:
        An instance of interfere.dynamics.VARMADynamics.
    """
    coef_matrices = [RNG.random((3, 3)) - 0.5 for i in range(5)]
    Z = RNG.random((3, 3))
    sigma = Z * Z.T

    model = interfere.dynamics.VARMADynamics(
        phi_matrices=coef_matrices[:2],
        theta_matrices=coef_matrices[2:],
        sigma=sigma
    )
    return model


def kuramoto_model() -> interfere.dynamics.Kuramoto:
    """Initializes a Kuramoto model.
    
    Returns:
        An instance of interfere.dynamics.KuramotoSakaguchi.
    """
    omega = RNG.random(10)
    K = 0.7
    A = RNG.random((10, 10)) < .3
    sigma=0.1
    return interfere.dynamics.Kuramoto(omega, K, A, sigma)


def kuramoto_sakaguchi_model() -> interfere.dynamics.KuramotoSakaguchi:
    """Initializes a Kuramoto-Sakaguchi model.
    
    Returns:
        An instance of interfere.dynamics.KuramotoSakaguchi.
    """
    omega = RNG.random(10)
    K = 0.7
    A = RNG.random((10, 10)) < .3
    sigma=0.1
    return interfere.dynamics.KuramotoSakaguchi(omega, K, A, A, sigma)


def stuart_landau_kuramoto_model() -> interfere.dynamics.StuartLandauKuramoto:
    """Initializes a Stuart-Landau-Kuramoto model.

    Returns
        An instance of interfere.dynamics.StuartLandauKuramoto.
    """
    omega = RNG.random(10)
    K = 0.7
    A = RNG.random((10, 10)) < .3
    sigma=0.1
    rho = RNG.random(10)
    return interfere.dynamics.StuartLandauKuramoto(omega, rho, K, sigma)


def hodgkin_huxley_model() ->  interfere.dynamics.HodgkinHuxleyPyclustering:
    """Initializes a hodgking huxley model.

    Returns:
        An instance of interfere.dynamics.HodgkinHuxleyPyclustering
    """
    return interfere.dynamics.HodgkinHuxleyPyclustering(
        [0, 0, 0, 15, 15, 15, 25, 25, 25, 40], sigma=0.1)
