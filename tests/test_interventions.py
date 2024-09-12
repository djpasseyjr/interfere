import interfere
import numpy as np

def test_perfect_intervention():
    rng = np.random.default_rng(2)

    g = interfere.PerfectIntervention(2, 0.3)
    assert np.all(
        g(np.array([0.1, 0.2, 0.0]), 0) == np.array([0.1, 0.2, 0.3])
    )
    X = rng.random((10, 5))
    X_en, X_ex = g.split_exogeneous(X)
    assert np.all(
        X_en == np.hstack([X[:, :2], X[:, 3:]])
    )
    assert np.all(X_ex == X[:, 2:3])

    g = interfere.PerfectIntervention([0, 1], [0, 0])
    assert np.all(
        g(np.array([0.1, 0.2, 0.0]), 0.1) == np.zeros(3)
    )

    assert np.all(
        g.eval_at_times(np.array([0, 1, 2])) == np.zeros((3, 2))
    )
    X = rng.random((10, 5))
    X_en, X_ex = g.split_exogeneous(X)
    assert np.all(X_en == X[:, 2:])
    assert np.all(X_ex == X[:, :2])


def test_signal_intervention():
    g = interfere.SignalIntervention(1, np.sin)
    x = np.array([1.1, 2, -1.2])
    assert np.allclose(g(x, 0), np.array([1.1, 0.0, -1.2]))
    assert np.allclose(g(x, np.pi/2), np.array([1.1, 1.0, -1.2]))

    g = interfere.SignalIntervention(2, lambda t: t ** 2)
    assert np.allclose(g(x, 1.0), np.array([1.1, 2.0, 1.0]))
    assert np.allclose(g(x, -2.0), np.array([1.1, 2.0, 4.0]))

    g = interfere.SignalIntervention([1, 2], [np.sin, lambda t: t ** 2])
    assert np.allclose(g(x, 0.0), np.array([1.1, 0.0, 0.0]))
    assert np.allclose(g(x, np.pi/2), np.array([1.1, 1.0, (np.pi/2)**2]))


def test_identity_intervention():

    # Test identity intervention for discrete stochastic dynamics.
    model = interfere.dynamics.coupled_map_1dlattice_chaotic_brownian(sigma=0.1)
    x0 = np.random.rand(10)
    t = np.arange(100)
    rng = np.random.default_rng(11)

    X = model.simulate(t, x0, rng=rng)

    rng = np.random.default_rng(11)
    X_ident = model.simulate(
        t, x0,
        intervention=interfere.interventions.IdentityIntervention(),
        rng=rng
    )
    
    assert np.all(X == X_ident)

    # Test identity intervention for continuous stochastic dynamics.
    model = interfere.dynamics.Belozyorov3DQuad(sigma=0.01)
    x0 = np.random.rand(3) * 0.1
    t = np.linspace(0, 10, 1001)

    rng = np.random.default_rng(11)
    X = model.simulate(t, x0, rng=rng)

    rng = np.random.default_rng(11)
    X_ident = model.simulate(
        t, x0,
        intervention=interfere.interventions.IdentityIntervention(),
        rng=rng
    )
    
    assert np.all(X == X_ident)
