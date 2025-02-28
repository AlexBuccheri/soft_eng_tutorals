#ifndef IMPRES_CG_BASE_CLASS_H
#define IMPRES_CG_BASE_CLASS_H

#include <functional>

#include <armadillo>

// Define a mathematical function that evaluates a vector to a float.
using FuncType = std::function<double(const arma::vec&)>;

// Define a derivative that accepts a vector and returns a vector of the same length.
using DerFuncType = std::function<arma::vec(const arma::vec&)>;

// Store results of a conjugate gradient calculation
struct CGResult {
    arma::vec x;
    int n_iter;
};

namespace optimiser {

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
        int iteration;

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
                    const int max_iter,
                    const double tol) :
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
