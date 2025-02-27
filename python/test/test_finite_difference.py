import numpy as np
from IMPRESPython.finite_difference import radial_schrondinger_eq


def test_radial_schrondinger_eq():
    n_points = 1000
    r_min, r_max = 0.0, 20.0
    eigenvalues, eigenvectors = radial_schrondinger_eq(n_points=n_points, r_min=r_min, r_max=r_max)

    assert np.isclose(eigenvalues[0], 4922.649294495996)
    assert np.isclose(eigenvalues[1], 4924.440798564914)
    assert np.isclose(eigenvalues[-1], 957.3689864531476)

    # TODO Would also be interesting to test something physical
