"""

"""
import numpy as np

from src.cg_abstract_class import FuncType, DerFuncType, NLConjugateGradient
from src.linesearch import line_search_backtrack
from src.cg import update_hessian


class BFGS(NLConjugateGradient):

    def __init__(self, f: FuncType,
                 df: DerFuncType,
                 x0: np.ndarray,
                 max_iter=500,
                 tol=1.e-6):
        super().__init__(f, df,
                         x0,
                         max_iter= max_iter,
                         tol=tol)
        n = np.size(x0)
        # Initialise approximate inverse Hessian
        self.hess = np.eye(n, n)

    def initialise_search_direction(self) -> np.ndarray:
        return - self.hess @ self.g

    def line_search(self) -> float:
        return line_search_backtrack(self.f, self.df, self.x, self.d, self.alpha)

    def update_hessian_or_coefficient(self) -> np.ndarray:
        s = self.x_next - self.x
        y = self.g_next - self.g
        return update_hessian(s, y, self.hess)

    def update_search_direction(self) -> np.ndarray:
        return -self.hess @ self.g
