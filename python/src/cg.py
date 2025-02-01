""" Implementations of conjugate gradient

* Functional
* Abstract class
* Plugin
"""
from typing import Tuple
from collections.abc import Callable

import numpy as np

from src.linesearch import line_search_backtrack


def conjugate_gradient(A: np.ndarray, x0: np.ndarray, b: np.ndarray, tol=1.e-8, n_iter=400) -> Tuple[np.ndarray, int]:
    r""" Solve the linear system of equations

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


def preconditioned_conjugate_gradient(A: np.ndarray,
                                      x0: np.ndarray,
                                      b: np.ndarray,
                                      M: np.ndarray,
                                      tol=1.e-8,
                                      n_iter=400) \
        -> Tuple[np.ndarray, int]:
    r""" Solve the linear system of equations

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


# Define a function that evaluates a vector to a float.
FuncType = Callable[[np.ndarray], float]

# Define a derivative that accepts a vector and returns a vector
DerFuncType = Callable[[np.ndarray], np.ndarray]


def fletcher_reeves_coefficient(g: np.ndarray, g_next: np.ndarray) -> float:
    return np.dot(g_next, g_next) / np.dot(g, g)


def nonlinear_conjugate_gradient(f: FuncType,
                                 df: FuncType,
                                 x0: np.ndarray,
                                 n_iter=100,
                                 tol=1.e-8):
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

    for k in range(0, n_iter):
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

    return x, n_iter


# Implement me
def bfgs_optimiser():
    return
