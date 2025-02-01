""" Implementations of conjugate gradient

* Functional
* Abstract class
* Plugin
"""
from typing import Tuple
from collections.abc import Callable

import numpy as np


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

def line_search_backtrack(f: FuncType,
                          df: FuncType,
                          x: np.ndarray,
                          d: np.ndarray,
                          alpha0: float,
                          reduction_factor=0.5,
                          p_decrease=1.e-4) -> float:
    """ Perform a line search using backtracking.

    Iteratively check if (x + alpha d) provides a sufficient decrease in the function value,
    as defined by the Armijo condition.

    :param f: Function
    :param df: Derivative of function f
    :param x: variable
    :param d: Search direction
    :param alpha0: Initial guess at step length, alpha. 1 is reasonable
    :param reduction_factor: Amount the step length is reduced per iteration
    :param p_decrease: Decrease factor
    :return: alpha: Step length
    """
    assert 0 < reduction_factor < 1, \
        "Reduction factor must be > 0 and < 1"

    assert 0 < p_decrease < 1, \
        "Decrease parameter must be > 0 and < 1"

    alpha = alpha0
    while f(x + alpha * d) > f(x) + (p_decrease * alpha * np.dot(df(x), d)):
        alpha *= reduction_factor

    return alpha


def fletcher_reeves_coefficient(g: np.ndarray, g_next: np.ndarray) -> float:
    return np.dot(g_next, g_next) / np.dot(g, g)


def nonlinear_conjugate_gradient(f: FuncType,
                                 df: FuncType,
                                 x0: np.ndarray,
                                 n_iter=100,
                                 tol=1.e-8) -> np.ndarray:
    """

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
            return x

        # Update search direction
        g = g_next
        d = -g + beta * d

    return x


# Implement me
def bfgs_optimiser():
    return
