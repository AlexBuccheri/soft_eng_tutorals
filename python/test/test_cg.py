"""Test conjugate gradient implementations"""

import numpy as np

# Absolute Import Path
from IMPRESPython import cg, cg_class
from IMPRESPython.optimiser_func_set import derivative_rosenbrock, rosenbrock

# from scipy.sparse import diags
from scipy.sparse import linalg


def test_conjugate_gradient():

    # SPD matrix (5x5 Tridiagonal)
    A = np.array(
        [
            [4, 1, 0, 0, 0],
            [1, 4, 1, 0, 0],
            [0, 1, 4, 1, 0],
            [0, 0, 1, 4, 1],
            [0, 0, 0, 1, 4],
        ]
    )

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


def test_nonlinear_conjugate_gradient():
    x_min = np.array([1.0, 1.0])
    # For x0 = np.array([-1.0, 0.8]), this was not converging
    x0 = np.array([0.2, 0.5])
    x, n_iter = cg.nonlinear_conjugate_gradient(
        rosenbrock, derivative_rosenbrock, x0, max_iter=2000, tol=1.0e-5
    )
    assert n_iter == 1269, "Converged before hit max iterations"
    assert np.allclose(x, x_min, atol=1.0e-4)


def test_bfgs_optimiser():
    x_min = np.array([1.0, 1.0])
    x0 = np.array([0.2, 0.5])

    x, n_iter = cg.bfgs_optimiser(
        rosenbrock, derivative_rosenbrock, x0, max_iter=3000, tol=1.0e-4
    )

    print(f"Func. {n_iter} iterations to get x_min", x)
    assert n_iter == 3000, "BFGS does not quite reach convergence"
    assert np.allclose(x, x_min, atol=1.0e-3)


def test_bfgs__obj():
    # Result from `test_bfgs_optimiser`
    x_ref = np.array([0.9999086, 0.99981565])
    x0 = np.array([0.2, 0.5])

    bfgs = cg_class.BFGS(
        rosenbrock, derivative_rosenbrock, x0, max_iter=3000, tol=1.0e-4
    )

    x, n_iter = bfgs.minimize()
    assert np.allclose(
        x, x_ref
    ), "Result should be the same as the function design"
