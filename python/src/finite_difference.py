"""Compute derivatives with finite difference"""

import sys
from collections.abc import Callable

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
