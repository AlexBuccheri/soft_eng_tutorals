#ifndef IMPRES_CG_BASE_CLASS_H
#define IMPRES_CG_BASE_CLASS_H
/*
 * Parent/base class for the non-linear CG algorithm.
 * Specific implementations, such as BFGS, can inherit from this
 * and implement their specialised methods without needing to reimplement
 * the CG algorithm, given in minimize().
 *
 * The class is not templated, so any non-virtual functions are defined
 * in the respective .cpp file to reduce compilation overhead.
 */
#include <functional>

#include <armadillo>

#include "cg.h"

namespace optimiser::armadillo::cg {

    // In the parent cg namespace and not cg::nonlinear, because
    // any nonlinear method (such as BFGS) can inherit from this
    // TODO As such, consider changing this class's name
    class NonLinearCG {
    // Child classes can access protected members
    protected:
        // Inputs
        const FuncType f;
        const DerFuncType df;
        const arma::vec x0;
        const int max_iter;
        const double tol;

        // Internal state
        arma::vec x;
        arma::vec g;       /// Gradient
        double alpha = 1;  /// Step length
        arma::vec d;       /// Search direction
        arma::mat hess;    /// Inv hessian or coefficient
        arma::vec x_next;
        arma::vec g_next;
        int iteration = 0;

    public:
        /**
         * @brief Constructs a non-linear conjugate gradient optimiser.
         *
         * @param f The objective function to minimize.
         * @param df The derivative (gradient) of the objective function.
         * @param x0 The initial guess for the solution.
         * @param max_iter The maximum number of iterations allowed.
         * @param tol The tolerance for convergence.
         */
        NonLinearCG(const FuncType &f,
                    const DerFuncType &df,
                    const arma::vec &x0,
                    int max_iter,
                    double tol) :
                f(f), df(df), x0(x0), max_iter(max_iter), tol(tol),
                x(x0), g(df(x0)), hess(arma::mat(x0.size(), x0.size())){}

        /// Destructor
        ~NonLinearCG() = default;

        /// Generic implementation of non-linear optimiser.
        CGResult minimize();

        // Pure virtual functions (no base class implementations)
        /// Initializes the search direction.
        virtual arma::vec initialise_search_direction() = 0;

        /// Performs a line search.
        virtual double line_search() = 0;

        /// Updates the Hessian approximation or coefficient.
        virtual arma::mat update_hessian_or_coefficient() = 0;

        /// Updates the search direction.
        virtual arma::vec update_search_direction() = 0;
        };

}
#endif //IMPRES_CG_BASE_CLASS_H
