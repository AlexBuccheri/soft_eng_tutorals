"""Implementations of conjugate gradient

* Functional
* Abstract class
* Plugin
"""

from collections.abc import Callable
from typing import Tuple

import numpy as np
from src.linesearch import line_search_backtrack


def conjugate_gradient(
    A: np.ndarray, x0: np.ndarray, b: np.ndarray, tol=1.0e-8, n_iter=400
) -> Tuple[np.ndarray, int]:
    r"""Solve the linear system of equations

    \mathbf{A} \mathbf{x} = \mathbf{b}

    :param A: Positive semi-definite matrix
    :param x0: Initial guess at left-hand side vectors
    :param b: Known right-hand side solution
    :return: x: L.H.S solution
         info : Provides convergence information:
            0 : successful exit
           >0 : convergence to tolerance not achieved, number of iterations
    """
    # assert A positive definite

    x = x0
    r = b - A @ x
    if np.linalg.norm(r) < tol:
        return x, 0

    p = np.copy(r)

    for k in range(n_iter):
        alpha = (r.T @ r) / (p.T @ (A @ p))
        x = x + alpha * p
        r_next = r - alpha * (A @ p)
        if np.linalg.norm(r_next) < tol:
            return x, 0
        beta = (r_next.T @ r_next) / (r.T @ r)
        p = r_next + (beta * p)
        r = r_next

    return x, n_iter


def preconditioned_conjugate_gradient(
    A: np.ndarray,
    x0: np.ndarray,
    b: np.ndarray,
    M: np.ndarray,
    tol=1.0e-8,
    max_iter=400,
) -> Tuple[np.ndarray, int]:
    r"""Solve the linear system of equations

    \mathbf{A} \mathbf{x} = \mathbf{b}

    :param A: Positive semi-definite matrix
    :param x0: Initial guess at left-hand side vectors
    :param b: Known right-hand side solution
    :param preconditioner_func: Function defining the preconditioner
    :return: x: L.H.S solution
         info : Provides convergence information:
            0 : successful exit
           >0 : convergence to tolerance not achieved, number of iterations
    """
    # assert A positive definite

    x = x0
    r = b - A @ x
    if np.linalg.norm(r) < tol:
        return x, 0

    z = np.linalg.solve(M, r)
    p = np.copy(z)

    for k in range(max_iter):
        alpha = (r.T @ r) / (p.T @ (A @ p))
        x = x + alpha * p
        r_next = r - alpha * (A @ p)
        if np.linalg.norm(r_next) < tol:
            return x, 0
        beta = (r_next.T @ r_next) / (r.T @ r)
        p = r_next + (beta * p)
        r = r_next

    return x, max_iter


# Define a function that evaluates a vector to a float.
FuncType = Callable[[np.ndarray], float]

# Define a derivative that accepts a vector and returns a vector of the same length.
DerFuncType = Callable[[np.ndarray], np.ndarray]


def fletcher_reeves_coefficient(g: np.ndarray, g_next: np.ndarray) -> float:
    return np.dot(g_next, g_next) / np.dot(g, g)


def nonlinear_conjugate_gradient(
    f: FuncType, df: DerFuncType, x0: np.ndarray, max_iter=500, tol=1.0e-6
):
    """Non-linear conjugate gradient.

    Note: The current implementation is not that robust.
    Implement:
    * A more sophisticated line search
    * Introduce a restart mechanism, resetting the search direction to –g when appropriate
    * Change the definition of the coefficient beta.

    :param f:  Function to optimise
    :param df: Derivative of f
    :param x0: Initial guess for x
    :return: x: Vector that optimises (minimises) f(x)
    """
    # Initialise variable
    x = np.copy(x0)
    # Compute gradient
    g = df(x)
    # Initialise search direction
    d = np.copy(-g)
    # Initialise step size
    alpha = 1

    for k in range(0, max_iter):
        # Line search
        alpha = line_search_backtrack(f, df, x, d, alpha)

        # Update variable
        x += alpha * d

        # Compute new gradient
        g_next = df(x)

        # Compute conjugate coefficient
        beta = fletcher_reeves_coefficient(g, g_next)

        # Check convergence
        if np.linalg.norm(g_next) <= tol:
            return x, k

        # Update search direction
        g = g_next
        d = -g + beta * d

    return x, max_iter


def update_hessian(
    s: np.ndarray, y: np.ndarray, Hess: np.ndarray
) -> np.ndarray:
    """Update the Hessian in BFGS.

    Maintain the symmetry and positive definiteness of Hₖ
    TODO Add expression for this approximation update.

    :param s: Change in the iterates, x_{k+1} = x_k
    :param y: Change in the gradients, grad_{k+1} - grad_k
    :param Hess: Current approximate inverse Hessian
    :return: Updated approximate inverse Hessian.
    """
    n = np.size(s)
    assert np.size(y) == n, "s and y must have same size"
    assert Hess.shape == (n, n)
    # If this is not true, the line search is likely the issue
    assert np.dot(y, s) > 0, f"Expect y dot s > 0 {np.dot(y, s)}"

    p = 1.0 / (np.dot(y, s))
    id = np.eye(n, n)

    return (id - p * np.outer(s, y.T)) @ Hess @ (id - p * np.outer(y, s.T)) + (
        p * np.outer(s, s.T)
    )


def bfgs_optimiser(
    f: FuncType, df: DerFuncType, x0: np.ndarray, max_iter=1000, tol=1.0e-6
):
    """BFGS.

    Scaling of the Hessian in the first iteration:
    This scaling is motivated by the secant condition used in quasi-Newton methods, ensuring that
    Hess_0 has a reasonable initial curvature estimate. The fraction (add expression) approximates the
    inverse of the Hessian’s first step curvature.

    This is not needed if the function, f, is well-conditioned.

    :param f:  Function to optimise
    :param df: Derivative of f
    :param x0: Initial guess for x
    :param max_iter: Maximum number of iterations to use
    :param tol: Convergence threshold in the gradient
    :return: x: Vector that minimises f(x)
    """
    n = np.size(x0)
    # Initialise variable
    x = np.copy(x0)
    # Initialise approximate inverse Hessian
    Hess = np.eye(n, n)
    # Initialise gradient at x0
    grad = df(x)
    # Initialise search direction
    d = -Hess @ grad
    # Initialise step size
    alpha = 1

    for k in range(0, max_iter):
        # Line search
        alpha = line_search_backtrack(f, df, x, d, alpha)

        # save x_k in s
        s = np.copy(x)
        # Update current point, x_{k+1}
        x += alpha * d

        # Save gradient, grad_k, in y
        y = np.copy(grad)
        # Compute new gradient
        grad: np.ndarray = df(x)

        # Check convergence
        if np.linalg.norm(grad) <= tol:
            return x, k

        # Compute change vectors
        s = x - s
        y = grad - y

        # Scale with an estimate of the curvature of the function at the starting point
        # if k == 0:
        #     scaling_factor = np.dot(s, y) / (np.dot(y, y) + 1e-8)
        #     Hess *= scaling_factor

        # Update the inv Hessian approximation
        Hess = update_hessian(s, y, Hess)

        # Update search direction
        d = -Hess @ grad

    return x, max_iter
