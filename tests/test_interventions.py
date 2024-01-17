import interfere
import numpy as np

def test_perfect_intervention():
    g = interfere.perfect_intervention(2, 0.3)
    assert g([0.1, 0.2, 0.0], 0) == [0.1, 0.2, 0.3]
    g = interfere.perfect_intervention([0, 1], [0, 0])
    assert g([0.1, 0.2, 0.0], 0.1) == [0, 0, 0]

def test_signal_intervention():
    g = interfere.signal_intervention(1, np.sin)
    x = np.array([1.1, 2, -1.2])
    assert np.allclose(g(x, 0), np.array([1.1, 0.0, -1.2]))
    assert np.allclose(g(x, np.pi/2), np.array([1.1, 1.0, -1.2]))

    g = interfere.signal_intervention(2, lambda t: t ** 2)
    assert np.allclose(g(x, 1.0), np.array([1.1, 2.0, 1.0]))
    assert np.allclose(g(x, -2.0), np.array([1.1, 2.0, 4.0]))