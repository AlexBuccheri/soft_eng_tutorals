"""Nonlinear CG functions, with more generic API"""

import numpy as np
from IMPRESPython import cg


def initialise_search_direction(g: np.ndarray, *args, **kwargs): ...


def line_search(
    f: cg.FuncType,
    df: cg.DerFuncType,
    x: np.ndarray,
    d: np.ndarray,
    alpha0: float,
    *args,
    **kwargs,
) -> float:
    """ """
    pass


def update_hessian_or_coefficient(
    g: np.ndarray, g_next: np.ndarray, *args, **kwargs
): ...


def update_search_direction(): ...
