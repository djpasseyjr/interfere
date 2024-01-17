from interfere.dynamics import attracting_fixed_point_4d_linear_sde
import numpy as np


def test_add_gaussian_noise():
    X = np.zeros((10_000, 3))

    seed = 12
    rng = np.random.default_rng(seed)

    stdevs = np.array([0.01, 1.0, 10])
    model = attracting_fixed_point_4d_linear_sde(0, stdevs)
    Xnoise = model.add_measurement_noise(X, rng)

    # Check that the standard deviations are correct.
    assert np.allclose(np.std(Xnoise, axis=0), stdevs, atol=0.1)

    rng = np.random.default_rng(seed)
    Xnoise2 = model.add_measurement_noise(X, rng)

    # Check that the range gives control over the randomness.
    assert np.all(Xnoise == Xnoise2)