import interfere
import numpy as np
import pytest


def test_perfect_intervention():
    rng = np.random.default_rng(2)

    g = interfere.PerfectIntervention(2, 0.3)
    assert np.all(
        g(np.array([0.1, 0.2, 0.0]), 0) == np.array([0.1, 0.2, 0.3])
    )
    X = rng.random((10, 5))
    X_en, X_ex = g.split_exog(X)
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
    X_en, X_ex = g.split_exog(X)
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


def test_intervention_equality():
    """Tests that interventions are correctly labeled as equal or unequal."""
    h = interfere.PerfectIntervention(0, 0)
    g = interfere.PerfectIntervention(0, 0.0)

    assert h == g, (
        "PerfectIntervention.__eq__ method should have found the two "
        "interventions equal."
    )

    h = interfere.PerfectIntervention([0], 0)
    g = interfere.PerfectIntervention(0, 0.0)
    
    assert h == g, (
        "PerfectIntervention.__eq__ method should have found the two "
        "interventions equal."
    )

    h = interfere.PerfectIntervention([0], 0)
    g = interfere.PerfectIntervention(0, [0.0])
    
    assert h == g, (
        "PerfectIntervention.__eq__ method should have found the two "
        "interventions equal."
    )

    h = interfere.PerfectIntervention([0], [0])
    g = interfere.PerfectIntervention(0, 0.0)
    
    assert h == g, (
        "PerfectIntervention.__eq__ method should have found the two "
        "interventions equal."
    )

    h = interfere.PerfectIntervention([0, 1], [0, 1])
    g = interfere.PerfectIntervention([0, 1], [0.0, 1])
    
    assert h == g, (
        "PerfectIntervention.__eq__ method should have found the two "
        "interventions equal."
    )

    h = interfere.PerfectIntervention([1], 0)
    g = interfere.PerfectIntervention(0, 0.0)
    
    assert h != g, (
        "PerfectIntervention.__eq__ method should not have found the two "
        "interventions equal."
    )

    h = interfere.PerfectIntervention(0, [0])
    g = interfere.PerfectIntervention(0, 1.0)
    
    assert h != g, (
        "PerfectIntervention.__eq__ method should not have found the two "
        "interventions equal."
    )

    h = interfere.PerfectIntervention([1, 2], [0, 1])
    g = interfere.PerfectIntervention([1, 2], [0.0, 2])
    
    assert h != g, (
        "PerfectIntervention.__eq__ method should not have found the two "
        "interventions equal."
    )

    h = interfere.PerfectIntervention([1, 3], [0, 1])
    g = interfere.PerfectIntervention([1, 2], [0.0, 1])
    
    assert h != g, (
        "PerfectIntervention.__eq__ method should not have found the two "
        "interventions equal."
    )

def test_split_exog():
    """Tests that splitting exog works as expected.
    """
    interv = interfere.PerfectIntervention([2,3], [1.0, 2.0])

    with pytest.raises(ValueError, match=(
            f"Array must be two dimensional."
        )):
        interv.split_exog(np.ones(10))

    with pytest.raises(ValueError, match=(
            f"Some intervention indexes are too big."
        )):
        interv.split_exog(np.ones((10, 2)))

    with pytest.raises(ValueError, match=(
            f"Some intervention indexes are too big."
        )):
        interv.split_exog(np.ones((10, 2)))

    interv = interfere.PerfectIntervention([0,1], [1.0, 2.0])
    X = np.ones((4, 2))
    en, ex = interv.split_exog(X)
    assert en is None, (
        "split_exog() returned endogenous when all states were exogenous."
        f"\n\tX.shape = {X.shape}"
        f"\n\tendog_X = {en}"
    )

    interv = interfere.PerfectIntervention([], [])
    X = np.ones((4, 2))
    en, ex = interv.split_exog(X)
    assert ex is None, (
        "split_exog() returned exogenous when all states were endog."
        f"\n\tX.shape = {X.shape}"
        f"\n\tendog_X = {en}"
    )


def test_combine_exog():
    """Tests that combining exog works as expected."""

    interv = interfere.PerfectIntervention(0, 1.0)
    ident_interv = interfere.IdentityIntervention()

    with pytest.raises(ValueError, match=("Both endo_X and exog_X are None.")):
        interv.combine_exog(None, None)

    with pytest.raises(ValueError, match=(
        "Exogenous states was None but intervention expects")):
        interv.combine_exog(np.ones((4, 2)), None)

    with pytest.raises(ValueError, match=(
        "Exogenous states provided but intervention does not expect"
        " exogenous states"
    )):
        ident_interv.combine_exog(np.ones((4, 2)), np.ones((4, 1)))

    with pytest.raises(ValueError, match=(
        "Endogenous states was None but intervention expects "
        "endogenous states."
    )):
        interv2 = interfere.PerfectIntervention([1, 2], [.5, .5])
        interv2.combine_exog(None, np.ones((4, 2)))

    with pytest.raises(ValueError, match=(
        "Endogenous and exogenous arrays must have the same number of ")):
        interv.combine_exog(np.ones((4, 2)), np.ones((3, 1)))

    with pytest.raises(ValueError, match=(
        "Wrong number of exogenous signals passed to intervention.")):
        interv.combine_exog(np.ones((4, 2)), np.ones((4, 2)))

    with pytest.raises(ValueError, match=(
        "Wrong number of exogenous signals passed to intervention.")):
        interv.combine_exog(np.ones((4, 2)), np.ones((4, 0)))


    X = np.random.rand(4, 1)
    assert np.allclose(ident_interv.combine_exog(X, None), X)
    assert np.allclose(interv.combine_exog(None, X), X)
    

def test_eval_at_times():
    t = np.arange(0, 1, 0.05)
    x0 = np.ones(3)
    model = interfere.dynamics.Lorenz()
    X1 = model.simulate(t, x0, intervention=interfere.IdentityIntervention())
    assert X1.shape == (len(t), 3)

    X2 = model.simulate(t, x0, intervention=interfere.PerfectIntervention(0, 5))
    assert np.all(X2[:, 0] == 5)
