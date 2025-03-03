#ifndef IMPRES_CG_TEMPLATE_H
#define IMPRES_CG_TEMPLATE_H

#include "cg.h"

namespace optimiser {

    /**
     * @brief Minimizes an objective function using a conjugate gradient method.
     *
     * This function calls unqualified functions (such as init_search_direction, line_search, etc.)
     * so that, by ADL, the overloads defined in the same namespace as the objective function (f)
     * are used.
     *
     * Each customization function must take f (or another type from its namespace) as its first parameter.
     *
     * @tparam FuncType  Type of the objective function.
     * @tparam DerFuncType Type of the derivative (gradient) function.
     * @param f The objective function.
     * @param df The gradient function.
     * @param x0 The initial guess.
     * @param max_iter Maximum number of iterations.
     * @param tol Convergence tolerance.
     * @return A CGResult containing the final solution and iteration count.
     */
    template <typename FuncType, typename DerFuncType, typename VecType>
    CGResult minimize(const FuncType &f,
                      const DerFuncType &df,
                      const arma::vec &x0,
                      int max_iter,
                      double tol)
    {
        // Initialise variable
        VecType x = x0;
        // Initialise gradient
        VecType g = df(x);
        // Initialise step size
        double alpha = 1.0;

        // Pass f so that ADL finds the correct init_search_direction in f's namespace.
        VecType d = init_search_direction(f, x, g);
        VecType hess = init_hessian_or_coefficient(x);

        for (int k = 0; k < max_iter; ++k) {
            alpha = line_search(f, x, d, df, tol);
            const VecType x_next = x + alpha * d;
            const VecType g_next = df(x_next);
            hess = update_hessian(f, x, g, x_next, g_next, hess);
            if (norm(g_next) <= tol) {
                return CGResult{x_next, k};
            }
            d = update_search_direction(f, x, g, x_next, g_next, d, hess);
            x = x_next;
            g = g_next;
        }
        return CGResult{x, max_iter};
    }

} // namespace nonlinear_cg


#endif
