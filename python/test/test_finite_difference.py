import numpy as np
from IMPRESPython.finite_difference import radial_schrondinger_eq


def test_radial_schrondinger_eq():
    n_points = 1000
    r_min, r_max = 0.0, 20.0
    eigenvalues, eigenvectors = radial_schrondinger_eq(n_points=n_points, r_min=r_min, r_max=r_max)
    assert np.isclose(eigenvalues[0], 4881.4012499)
    assert np.isclose(eigenvalues[1], 4883.67599238)
    assert np.isclose(eigenvalues[-1], -0.0500501002004008)

    # TODO Would also be interesting to test something physical
