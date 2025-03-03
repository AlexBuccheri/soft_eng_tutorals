#ifndef IMPRES_CG_TEMPLATE_H
#define IMPRES_CG_TEMPLATE_H

#include "cg.h" // CGResult
#include "line_search.h" // LineSearchParam

namespace optimiser::templated {

    /**
     * @brief Minimizes an objective function using a conjugate gradient method.
     *
     * This function calls expects all function calls to be defined with the Traits class
     * as static members:
     *  init_search_direction
     *  init_hessian_or_coefficient
     *  line_search
     *  update_hessian
     *  update_search_direction
     *  FuncType, DerFuncType
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
    template <typename Traits, typename FuncType, typename DerFuncType, typename VecType>
    CGResult non_linear_conjugate_gradient(const FuncType &f,
                                           const DerFuncType &df,
                                           const VecType &x0,
                                           int max_iter,
                                           double tol,
                                           const armadillo::LineSearchParam & ls_params)
    {
        // Initialise variable
        VecType x = x0;
        // Initialise gradient
        VecType g = df(x);
        // Initialise step size
        double alpha = 1.0;

        // Pass f so that ADL finds the correct init_search_direction in f's namespace.
        auto hess = Traits::init_hessian_or_coefficient(x);
        // This requires a variadic signature to be generic
        VecType d = Traits::init_search_direction(hess, g);

        for (int k = 0; k < max_iter; ++k) {
            alpha = Traits::line_search(f, df, x, d, alpha, ls_params);
            // This operation is a pain using the STL functions because
            // it requires * and + to be overloaded. Easier to also abstract
            // the definition of x_next to a function
            const VecType x_next = x + alpha * d;
            const VecType g_next = df(x_next);
            // This requires a variadic signature to be generic
            hess = Traits::update_hessian(x, g, x_next, g_next, hess);
            if (norm(g_next) <= tol) {
                return CGResult{x_next, k};
            }
            // This requires a variadic signature to be generic
            d = Traits::update_search_direction(hess, g_next);
            x = x_next;
            g = g_next;
        }
        return CGResult{x, max_iter};
    }

} // namespace nonlinear_cg


#endif
