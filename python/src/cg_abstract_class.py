from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import List, Optional, Tuple

import numpy as np

# Define a function that evaluates a vector to a float.
FuncType = Callable[[np.ndarray], float]

# Define a derivative that accepts a vector and returns a vector of the same length.
DerFuncType = Callable[[np.ndarray], np.ndarray]


class NLConjugateGradient(ABC):

    def __init__(
        self,
        f: FuncType,
        df: DerFuncType,
        x0: np.ndarray,
        max_iter: Optional[float] = 500,
        tol: Optional[float] = 1.0e-6,
        hooks: Optional[List[Callable]] = None,
    ):
        """Initialise variables prior to CG loop"""
        self.f = f
        self.df = df
        self.max_iter = max_iter
        self.tol = tol
        if hooks is None:
            self.hooks = []

        # Initialise variable
        self.x = np.copy(x0)
        # Compute gradient
        self.g = self.df(self.x)
        # Initialise step size
        self.alpha = 1
        # Zero the search direction
        self.d = 0
        # Zero coefficient or approx inv Hessian
        self.hess = 0
        # Zero iteration counter
        self.k = 0

    @abstractmethod
    def initialise_search_direction(self) -> float:
        """Initialise the search direction,"""
        pass

    @abstractmethod
    def line_search(self) -> float:
        """Perform a line search

        Describe more in more detail
        """
        pass

    @abstractmethod
    def update_hessian_or_coefficient(self):
        pass

    @abstractmethod
    def update_search_direction(self) -> np.ndarray:
        """Update the search direction, self.d
        This should use self.g_next, as the vectors x and gradient g
        are the last quantities to be updated, per iteration
        """
        pass

    def minimize(self) -> Tuple[np.ndarray, int]:
        """Minimize f(x) using non-linear CG"""
        self.d = self.initialise_search_direction()

        for k in range(0, self.max_iter):
            self.k = k

            # Compute new step length
            self.alpha = self.line_search()

            # Compute new variable
            self.x_next = self.x + self.alpha * self.d

            # Compute new gradient
            self.g_next = self.df(self.x_next)

            # Call any optional functions prior to updating the Hessian/coefficient
            for func in self.hooks:
                func()

            # Update coefficient or approx inv Hessian
            self.hess = self.update_hessian_or_coefficient()

            if np.linalg.norm(self.g_next) <= self.tol:
                return self.x_next, self.k

            # Update quantities
            self.d = self.update_search_direction()
            self.x = self.x_next
            self.g = self.g_next

        return self.x, self.max_iter
