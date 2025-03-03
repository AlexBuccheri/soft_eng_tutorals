#ifndef IMRESS_CG_H
#define IMRESS_CG_H

#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

#include <armadillo>

#include "cg.h" // FuncType, DerFuncType, CGResult
#include "line_search.h" // LineSearchFunc

namespace optimiser::armadillo::cg {

    namespace linear{
        /**
         * @brief Solves the linear system of equations A * x = b using the
         *        Linear Conjugate Gradient method.
         *
         * This function iteratively solves the system of equations where `A` is a
         * positive semi-definite matrix, `b` is the known right-hand side vector,
         * and `x0` is an initial guess for the solution vector.
         *
         * @param a Positive semi-definite matrix (A).
         * @param x0 Initial guess for the left-hand side solution vector (x).
         * @param b Right-hand side solution vector (b).
         * @param tol Optional tolerance for convergence .
         * @param max_iter Optional maximum number of iterations.
         * @return Solution vector (x).
         */
        CGResult linear_conjugate_gradient(const arma::mat& a,
                                            const arma::vec& x0,
                                            const arma::vec& b,
                                            double tol = 1e-8,
                                            int max_iter = 1000);

    }

    namespace nonlinear{
        // TODO Document me
        CGResult non_linear_conjugate_gradient(const FuncType& f,
                                               const DerFuncType& df,
                                               const arma::vec& x0,
                                               LineSearchFunc&,
                                               int max_iter = 1000,
                                               double tol = 1.e-6);
    }

    namespace bfgs {

        // @brief Update the Hessian in BFGS. Interface required for class method
        arma::mat update_hessian(const arma::vec &s,
                                 const arma::vec &y,
                                 const arma::mat &hess);

        /* Issues with ADL and templating
         *
         * NOTE: The compiler will look in the namespace of the rvalue, not the alias
         * so aliasing like this does NOT work:
         *  using VecType = arma::vec;
         *
         *  Same deal with using ADL on using FuncType = std::function<double(const arma::vec&)>;
         *  It's going to ignore the alias, and look in the std and arma namespaces
         *
         *  One should define a struct containing settings in this namespace and pass that as the first arg
         *  to EVERY function call in the templated function, OR define a traits class of static members.
         *  The latter means that one needs to explicitly pass the template parameter rather than
         *  have the compiler infer it from the arguments, but wrapping all the required functionality
         *  of arma::vec or creating a BFGS instance to pass around was more effort. The latter also made
         *  no sense with the current design
         */
        struct Traits {
            // @brief Initialise the approximation to the inv Hessian
            static arma::mat init_hessian_or_coefficient(const arma::vec &x);

            // @brief Initialise the search direction for BFGS
            static arma::vec init_search_direction(arma::mat &hess, arma::vec &g);

            // @brief Alias to line_search_backtrack as signature of the template
            // CG matches (so no point writing an overload/wrapper in .cpp)
            //using line_search = decltype(&optimiser::armadillo::line_search_backtrack);
            static double line_search(const FuncType &f,
                               const DerFuncType &df,
                               const arma::vec &x,
                               const arma::vec &direction,
                               double alpha0,
                               const LineSearchParam &params);

            // @brief Update the Hessian in BFGS. Interface required for templated function
            static arma::mat update_hessian(const arma::vec &x,
                                     const arma::vec &g,
                                     const arma::vec &x_next,
                                     const arma::vec &g_next,
                                     const arma::mat &hess);

            // @brief Update the BFGS search direction
            static arma::vec update_search_direction(const arma::mat &hess, const arma::vec &g_next);
        };
    }

}

#endif
