import numpy as np

from src.cg import fletcher_reeves_coefficient, update_hessian
from src.cg_abstract_class import DerFuncType, FuncType, NLConjugateGradient
from src.linesearch import line_search_backtrack


class NLCG(NLConjugateGradient):

    def __init__(
        self,
        f: FuncType,
        df: DerFuncType,
        x0: np.ndarray,
        max_iter=500,
        tol=1.0e-6,
    ):
        super().__init__(f, df, x0, max_iter=max_iter, tol=tol)

    def initialise_search_direction(self) -> np.ndarray:
        return -self.g

    def line_search(self) -> float:
        return line_search_backtrack(
            self.f, self.df, self.x, self.d, self.alpha
        )

    def update_hessian_or_coefficient(self) -> float:
        return fletcher_reeves_coefficient(self.g, self.g_next)

    def update_search_direction(self) -> np.ndarray:
        return -self.g_next + self.hess * self.d


# Note(Alex) See my chatbot project that uses a combination of an abstract class, free functions
# and the type constructor to remove the boilerplate.
class BFGS(NLConjugateGradient):

    def __init__(
        self,
        f: FuncType,
        df: DerFuncType,
        x0: np.ndarray,
        max_iter=500,
        tol=1.0e-6,
    ):
        super().__init__(f, df, x0, max_iter=max_iter, tol=tol)
        n = np.size(x0)
        # Initialise approximate inverse Hessian
        self.hess = np.eye(n, n)

    def initialise_search_direction(self) -> np.ndarray:
        return -self.hess @ self.g

    def line_search(self) -> float:
        return line_search_backtrack(
            self.f, self.df, self.x, self.d, self.alpha
        )

    def update_hessian_or_coefficient(self) -> np.ndarray:
        s = self.x_next - self.x
        y = self.g_next - self.g
        return update_hessian(s, y, self.hess)

    def update_search_direction(self) -> np.ndarray:
        return -self.hess @ self.g_next
