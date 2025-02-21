"""Line search algorithms"""

from collections.abc import Callable

import numpy as np

# Define a function that evaluates a vector to a float.
FuncType = Callable[[np.ndarray], float]

# Define a derivative that accepts a vector and returns a vector
DerFuncType = Callable[[np.ndarray], np.ndarray]


def line_search_backtrack(
    f: FuncType,
    df: DerFuncType,
    x: np.ndarray,
    d: np.ndarray,
    alpha0: float,
    reduction_factor=0.5,
    p_decrease=1.0e-4,
) -> float:
    """Perform a line search using backtracking.

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
    assert 0 < reduction_factor < 1, "Reduction factor must be > 0 and < 1"

    assert 0 < p_decrease < 1, "Decrease parameter must be > 0 and < 1"

    alpha = alpha0
    while f(x + alpha * d) > f(x) + (p_decrease * alpha * np.dot(df(x), d)):
        alpha *= reduction_factor

    return alpha


def line_search_strong_wolfe_conditions(
    f: FuncType,
    df: FuncType,
    x: np.ndarray,
    d: np.ndarray,
    alpha0: float,
    reduction_factor=0.5,
    c1=1.0e-4,
    c2=0.9,
) -> float:
    """ """
    assert 0 < reduction_factor < 1, "Reduction factor must be > 0 and < 1"

    assert 0 < c1 < c2, "Decrease parameter, c1, must be > 0 and < c2"

    assert c1 < c2 < 1, "Second decrease parameter, c2, must be > c1 and < 1"

    # Constants
    g_dot_d = np.dot(df(x), d)
    f_x = f(x)

    # Initialise iterative values
    alpha = alpha0
    x_increment = x + alpha * d

    while f(x_increment) > f_x + (c1 * alpha * g_dot_d) and np.abs(
        np.dot(df(x_increment), d)
    ) > c2 * np.abs(g_dot_d):
        alpha *= reduction_factor
        x_increment = x + alpha * d

    return alpha
