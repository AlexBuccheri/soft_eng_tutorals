"""Compute derivatives with finite difference"""

import sys
from collections.abc import Callable
from typing import Tuple

import numpy as np


def central_difference(
        f: Callable[[np.ndarray], float], x: np.ndarray
) -> np.ndarray:
    """Compute finite derivatives using central difference
    Note, oscillating functions can yield zero derivative.
    :param f:
    :param x:
    :return:
    """
    grad = np.empty_like(x)
    x_forward = np.copy(x)
    x_backward = np.copy(x)

    for i in range(len(x)):
        h_i = np.sqrt(sys.float_info.epsilon) * max(1.0, np.abs(x[i]))
        # Perturb only the i-th coordinate
        x_forward[i] += 0.5 * h_i
        x_backward[i] -= 0.5 * h_i
        grad[i] = (f(x_forward) - f(x_backward)) / h_i
        # Undo the ith displacement
        x_forward[i] -= 0.5 * h_i
        x_backward[i] += 0.5 * h_i
    return grad


def radial_schrondinger_eq(n_points=1000, r_min=0.0, r_max=20.0) -> Tuple[np.ndarray, np.ndarray]:
    """ Solve the 1D radial schrondinger eq in a spherical potential, for l=0, using finite differences.

    TODO Add the maths
    """
    # Could also set retstep=True and get from linspace
    # Line element
    dr = (r_max - r_min) / float(n_points - 1)

    # Create a grid
    r = np.linspace(r_min, r_max, num=n_points, endpoint=True)

    # Set singularity at r = 0 to the grid spacing
    r[0] = dr

    # The KE operator in the central difference approximation is
    # 1 \frac{1}{2} {d^2}{dr^2} u(r) = (-0.5 / (dr)**2) * [u(r_{i-1} - 2 u(r_{i} + u(r_{i+1}]
    coefficient = -0.5 / dr ** 2
    stencil_weights = np.array([1.0, -2.0, 1.0])

    # Set H according to the KE operator
    # Initialised with zeros as most elements never get touched: Sparse matrix
    ham = np.zeros(shape=(n_points, n_points))

    # First point, with one-sided difference
    # TODO Add expression
    ham[0, 0] = - 2 * coefficient
    ham[0, 1] = coefficient

    # Interior points. For point i, the central difference stencil defines values points
    # i-1, i and i+1
    for i in range(1, n_points-2):
        ham[i, i-1:i+2] += coefficient * stencil_weights[:]

    # Last point, with one-sided difference
    ham[-1, -2] = coefficient
    ham[-1, -1] = - 2 * coefficient

    # Add contribution from the Coulomb potential, -e/r, which is -1/r in atomic units
    for i in range(1, n_points-1):
        ham[i, i] += - 1 / r[i]

    # Diagonalise
    eigenvalues, eigenvectors = np.linalg.eig(ham)

    return eigenvalues, eigenvectors
