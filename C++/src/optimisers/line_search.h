#ifndef IMPRES_LINE_SEARCH_H
#define IMPRES_LINE_SEARCH_H

#include <functional>

#include "cg.h"  // FuncType, DerFuncType

namespace optimiser::armadillo{

    // @brief Generic line search function signature
    template <typename... ExtraArgs>
    using LineSearchFuncVar = std::function<double(const FuncType&,
                                                   const DerFuncType&,
                                                   const arma::vec&,
                                                   const arma::vec&,
                                                   double,
                                                   ExtraArgs...)>;

    // @brief Line search function signature with no optional arguments
    using LineSearchFunc = LineSearchFuncVar<>;

    // @brief Backtrack function signature. Extra arguments are reduction factor and c1
    using LineSearchBackTrackFunc = LineSearchFuncVar<double, double>;

    /*
     * @brief Performs a backtracking line search to determine the step size.
     *
     * This function implements the Armijo condition-based backtracking line search.
     * It reduces the step size iteratively until the condition is satisfied.
     *
     * @param FuncType The type of the function that evaluates f(x).
     * @param DerFuncType The type of the function that evaluates the gradient df(x).
     * @param f The function to be minimized.
     * @param df The derivative (gradient) function of f.
     * @param x The current point in the optimization process.
     * @param direction The descent direction.
     * @param alpha0 The initial step size.
     * @param reduction_factor The factor by which alpha is reduced (must be in (0,1)).
     * @param c1 The Armijo condition parameter (must be in (0,1)).
     * @return The computed step size that satisfies the Armijo condition.
     */
     double line_search_backtrack(const FuncType& f,
                                  const DerFuncType& df,
                                  const arma::vec& x,
                                  const arma::vec& direction,
                                  double alpha0,
                                  double reduction_factor = 0.5,
                                  double c1 = 1.e-4);

    struct LineSearchParam{
        double reduction_factor = 0.5;
        double c1 = 1.e-4;
    };

}

#endif //IMPRES_LINE_SEARCH_H
