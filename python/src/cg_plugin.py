""" Implement non-linear CG with a plugin design.
"""
import importlib
from collections.abc import Callable
from typing import Tuple, Optional, List

import numpy as np

# Define a function that evaluates a vector to a float.
FuncType = Callable[[np.ndarray], float]

# Define a derivative that accepts a vector and returns a vector of the same length.
DerFuncType = Callable[[np.ndarray], np.ndarray]


class NLCGFunctionsInterface:
    """Define the API of functions to supply to `nl_conjugate_gradient"""

    @staticmethod
    def initialise_search_direction(g: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        :param g: Gradient
        :param *args
        :param **kwargs For example with BFGS, this would be the initial approximation
        for the inverse Hessian, such as hess = np.eye(n, n)
        """
        ...

    @staticmethod
    def line_search(f: FuncType,
                    df: DerFuncType,
                    x: np.ndarray,
                    d: np.ndarray,
                    alpha0: float,
                    *args, **kwargs) -> float:
        """ Perform a line search to find an optimal step size, alpha.

        :param f: Function
        :param df: Derivative of function f
        :param x: variable
        :param d: Search direction
        :param alpha0: Initial guess at step length, alpha. 1 is reasonable
        :param *args
        :param **kwargs: Parameters used by the specific search conditions, such as
        the reduction factor and decrease factor.
        :return: alpha: Step length
        """
        ...

    @staticmethod
    def update_hessian_or_coefficient(g:np.ndarray, g_next:np.ndarray, *args, **kwargs) -> float | np.ndarray:
        """ In NL-CG update the coefficient beta. In BFGS, update the approximate inv Hessian.

        :param g: Input or starting gradient for step k
        :param g_next: Updated gradient for step k
        :param *args
        :param **kwargs: Algorithm-specific quantities. For BFGS for example, this might
        be x and x_next, as well as the initial Hessian for step k.
        :return Updated Hessian or coefficient
        """
        ...

    @staticmethod
    def update_search_direction(g_next, hess_or_beta) -> np.ndarray:
        """ Update the search direction

        :param g_next: Updated gradient for step k
        :param hess_or_beta Updated Hessian or beta coefficient.
        :return Updated search direction, d.
        """
        ...


def nl_conjugate_gradient_factory(module_name: str) -> Callable:
    """ Dynamically load a module and serve the corresponding non-linear CG function.

    :param module_name: Module containing specialised functions
    :return nl_conjugate_gradient: Function to perform non-linear CG.
    """
    cg_module = importlib.import_module(module_name)  # type: ignore

    def nl_conjugate_gradient(f: FuncType,
                              df: DerFuncType,
                              x0: np.ndarray,
                              max_iter=1000,
                              tol=1.e-6,
                              *args,
                              **kwargs) -> Tuple[np.ndarray, int]:
        """ Perform non-linear conjugate gradient

        When implementing bfgs for example, the initial guess for the approximate
        inverse Hessian should be passed as a kwarg.

        :param f:  Function to optimise
        :param df: Derivative of f
        :param x0: Initial guess for x
        :param max_iter: Maximum number of iterations to use
        :param tol: Convergence threshold in the gradient
        :return: x: Vector that minimises f(x)
        """
        # Initialise variable
        x = np.copy(x0)
        # Compute gradient
        g = df(x)
        # Initialise step size
        alpha = 1

        # Need hess and grad OR just grad
        d = cg_module.initialise_search_direction(g, *args, **kwargs)

        for k in range(0, max_iter):

            # Compute new step length
            alpha = cg_module.line_search(f, df, x, d, alpha, *args, **kwargs)

            # Compute new variable
            x_next = x + alpha * d
            kwargs['x_next'] = x_next

            # Compute new gradient
            g_next = df(x_next)

            # Update coefficient or approx inv Hessian
            hess_or_beta = cg_module.update_hessian_or_coefficient(g, g_next, *args, **kwargs)

            if np.linalg.norm(g_next) <= tol:
                return x_next, k

            # Update quantities
            d = cg_module.update_search_direction(g_next, hess_or_beta)
            x = x_next
            kwargs['x'] = x
            g = g_next

        return x, max_iter

    return nl_conjugate_gradient
