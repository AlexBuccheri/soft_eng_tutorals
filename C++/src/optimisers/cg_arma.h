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

    namespace bfgs{
        // @brief Update the Hessian in BFGS.
        arma::mat update_hessian(const arma::vec &s,
                                 const arma::vec &y,
                                 const arma::mat &hess);

    }

}

#endif
