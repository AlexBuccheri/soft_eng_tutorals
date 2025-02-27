"""

"""
from typing import Tuple

import numpy as np


def construct_grid(n_points: int, r_min: float, r_max: float) -> tuple:
    """ Construct 1D radial grid
    :param n_points: Number of grid points
    :param r_min: Starting point
    :param r_max: End point

    """
    r, dr = np.linspace(r_min, r_max, num=n_points, endpoint=True, retstep=True)
    # Set singularity at r = 0 to the grid spacing
    r[0] = dr
    return r, dr


def construct_stencil(r: np.ndarray, dr: float) -> tuple:
    """ Construct stencil for central difference approximation of the KE operator.
    The KE operator in the central difference approximation is:
    1 \frac{1}{2} {d^2}{dr^2} u(r) = (-0.5 / (dr)**2) * [u(r_{i-1} - 2 u(r_{i} + u(r_{i+1}]
    """
    coefficient = -0.5 / dr ** 2
    stencil_weights = np.array([1.0, -2.0, 1.0])
    return coefficient, stencil_weights


def kinetic_energy_contribution(n_points: int, coefficient: float, stencil_weights: np.ndarray) -> np.ndarray:
    """ Kinetic energy operator contribution.
    The KE operator in the central difference approximation is:
    1 \frac{1}{2} {d^2}{dr^2} u(r) = (-0.5 / (dr)**2) * [u(r_{i-1} - 2 u(r_{i} + u(r_{i+1}]
    """
    ke = np.zeros(shape=(n_points, n_points))

    # First point, with one-sided difference
    # At H_{0, (-1, 0, 1)}, drop the weight corresponding to -1
    ke[0, 0:2] = coefficient * stencil_weights[1:]

    # Interior points. For point i, the central difference stencil defines values points
    # i-1, i and i+1
    for i in range(1, n_points - 1):
        ke[i, i - 1:i + 2] += coefficient * stencil_weights[:]

    # Last point, with one-sided difference
    # At H_{n_points-1, (n_points-2, n_points-1, n_points)}, drop the weight corresponding to index n_points
    ke[-1, -2:n_points] = coefficient * stencil_weights[0:2]

    return ke


def coulomb_energy_contribution(r: np.ndarray) -> np.ndarray:
    """ Coulomb potential contribution.
    The Coulomb potential is -1/r in atomic units.
    Assumes the singularity is handled in the definition of r.
    """
    n_points = r.size
    coul = np.zeros(shape=(n_points, n_points))
    for i in range(0, n_points):
        coul[i, i] += - 1 / r[i]
    return coul


class Eigenstates:
    def __init__(self, values, vectors):
        self.values = values
        self.vectors = vectors


def radial_schrondinger_eq(n_points=1000, r_min=0.0, r_max=20.0) -> Eigenstates:
    r""" Solve the 1D radial Schrödinger equation in a spherical potential, for l=0, using finite differences.

    The radial Schrödinger equation for an \(s\)-wave (\(l = 0\)) is:

    .. math::

        -\frac{1}{2} \frac{d^2 u(r)}{dr^2} + V(r) u(r) = E u(r).

    Using the central difference approximation:

        \frac{d^2 u}{dr^2} \approx \frac{u(r_{i+1}) - 2 u(r_i) + u(r_{i-1})}{\Delta r^2},

    we can represent this as a discretised equation:

    -\frac{1}{2} \frac{u(r_{i+1}) - 2 u(r_i) + u(r_{i-1})}{\Delta r^2} + V(r_i) u(r_i) = E u(r_i).

    The end points at 0 and n_points are approximated as:

    \frac{d^2 u}{dr^2} \approx \frac{-2 u(r_0) + u(r_1)} {\Delta r^2},

    and

    \frac{d^2 u}{dr^2} \approx \frac{u(r_{N-1}) - 2 u(r_N)} {\Delta r^2}.

    :param n_points: Number of grid points.
    :param r_min: Start of the radial grid.
    :param r_max: End of the radial grid.
    :return Tuple of eigenvalues and eigenvectors.
    """
    r, dr = construct_grid(n_points, r_min, r_max)
    coefficient, stencil_weights = construct_stencil(r, dr)
    ham = kinetic_energy_contribution(n_points, coefficient, stencil_weights)
    ham += coulomb_energy_contribution(r)
    eigenvalues, eigenvectors = np.linalg.eig(ham)
    return Eigenstates(eigenvalues, eigenvectors)


def full_radial_schrondinger_eq(n_points=1000, r_min=0.0, r_max=20.0) -> Tuple[np.ndarray, np.ndarray]:
    r""" Solve the 1D radial Schrödinger equation in a spherical potential, for l=0, using finite differences.

    The radial Schrödinger equation for an \(s\)-wave (\(l = 0\)) is:

    .. math::

        -\frac{1}{2} \frac{d^2 u(r)}{dr^2} + V(r) u(r) = E u(r).

    Using the central difference approximation:

        \frac{d^2 u}{dr^2} \approx \frac{u(r_{i+1}) - 2 u(r_i) + u(r_{i-1})}{\Delta r^2},

    we can represent this as a discretised equation:

    -\frac{1}{2} \frac{u(r_{i+1}) - 2 u(r_i) + u(r_{i-1})}{\Delta r^2} + V(r_i) u(r_i) = E u(r_i).

    The end points at 0 and n_points are approximated as:

    \frac{d^2 u}{dr^2} \approx \frac{-2 u(r_0) + u(r_1)} {\Delta r^2},

    and

    \frac{d^2 u}{dr^2} \approx \frac{u(r_{N-1}) - 2 u(r_N)} {\Delta r^2}.

    :param n_points: Number of grid points.
    :param r_min: Start of the radial grid.
    :param r_max: End of the radial grid.
    :return Tuple of eigenvalues and eigenvectors.
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
    # At H_{0, (-1, 0, 1)}, drop the weight corresponding to -1
    ham[0, 0:2] = coefficient * stencil_weights[1:]

    # Interior points. For point i, the central difference stencil defines values points
    # i-1, i and i+1
    for i in range(1, n_points - 1):
        ham[i, i - 1:i + 2] += coefficient * stencil_weights[:]

    # Last point, with one-sided difference
    # At H_{n_points-1, (n_points-2, n_points-1, n_points)}, drop the weight corresponding to index n_points
    ham[-1, -2:n_points] = coefficient * stencil_weights[0:2]
    assert ham[-1, -1] == ham[n_points - 1, n_points - 1], "Confirm last element is n_points-1"

    # Add contribution from the Coulomb potential, -e/r, which is -1/r in atomic units
    # The singularity is approximated, as r[0] = dr
    for i in range(0, n_points):
        ham[i, i] += - 1 / r[i]

    # Diagonalise
    eigenvalues, eigenvectors = np.linalg.eig(ham)

    return eigenvalues, eigenvectors
