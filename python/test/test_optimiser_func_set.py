"""Test optimiser functions and their derivatives."""

import numpy as np
from IMPRESPython.finite_difference import central_difference
from IMPRESPython.optimiser_func_set import derivative_rosenbrock, rosenbrock


def test_derivative_rosenbrock():
    """Test analytic gradient against finite difference"""
    # Some x-value that is not the global minimum, (1, 1)
    x = np.array([-1.2, 1.0])
    grad_fd = central_difference(rosenbrock, x)
    grad_analytic = derivative_rosenbrock(x)
    assert np.allclose(grad_analytic, grad_fd)

    x_min = np.array([1.0, 1.0])
    grad_fd = central_difference(rosenbrock, x_min)
    grad_analytic = derivative_rosenbrock(x_min)
    assert np.allclose(grad_analytic, grad_fd), "Global minimum"

