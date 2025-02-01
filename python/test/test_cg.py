""" Test conjugate gradient implementations
"""
import numpy as np
from scipy.sparse import diags
from scipy.sparse import linalg

# Absolute Import Path
from src import cg



def test_conjugate_gradient():

    # SPD matrix (5x5 Tridiagonal)
    A = np.array([
        [4, 1, 0, 0, 0],
        [1, 4, 1, 0, 0],
        [0, 1, 4, 1, 0],
        [0, 0, 1, 4, 1],
        [0, 0, 0, 1, 4]
    ])

    # Define b
    b = np.array([1, 2, 3, 4, 5])

    # Initial guess
    x0 = np.zeros_like(b)

    x_ref, info = linalg.cg(A, b, x0=x0)
    x, _ = cg.conjugate_gradient(A, x0, b)
    assert np.allclose(x, x_ref)


# def test_conjugate_gradient_sparse_matrix():
#     size = 1000
#     diagonal = 4 * np.ones(size)
#     off_diagonal = np.ones(size - 1)
#     A_sparse = diags([diagonal, off_diagonal, off_diagonal], [0, -1, 1], format='csr')


# def test_nonlinear_conjugate_gradient():
#
