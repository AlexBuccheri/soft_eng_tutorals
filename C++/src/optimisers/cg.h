#ifndef IMRESS_CG_H
#define IMRESS_CG_H

#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

#include <armadillo>

namespace optimiser {

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
 * @return arma::vec Solution vector (x).
 */
arma::vec linear_conjugate_gradient(const arma::mat& a,
    const arma::vec& x0,
    const arma::vec& b,
    double tol = 1e-8,
    int max_iter = 1000);

// Define a mathematical function that evaluates a vector to a float.
using FuncType = std::function<double(const arma::vec&)>;

// Define a derivative that accepts a vector and returns a vector of the same length.
using DerFuncType = std::function<arma::vec(const arma::vec&)>;

// Store results of a conjugate gradient calculation
struct CGResult {
    arma::vec x;
    int n_iter;
};

// Generic line search function signature
template <typename... ExtraArgs>
using LineSearchFuncVar = std::function<double(const FuncType&,
    const DerFuncType&,
    const arma::vec&,
    const arma::vec&,
    double,
    ExtraArgs...)>;

// Line search function signature with no optional arguments
using LineSearchFunc = LineSearchFuncVar<>;

// Backtrack function signature. Extra arguments are reduction factor and c1
using LineSearchBackTrackFunc = LineSearchFuncVar<double, double>;

// TODO Document me
double line_search_backtrack(const FuncType& f,
    const DerFuncType& df,
    const arma::vec& x,
    const arma::vec& direction,
    double alpha0,
    double reduction_factor = 0.5,
    double c1 = 1.e-4);

// TODO Document me
CGResult non_linear_conjugate_gradient(const FuncType& f,
    const DerFuncType& df,
    const arma::vec& x0,
    LineSearchFunc&,
    int max_iter = 1000,
    double tol = 1.e-6);

}

#endif // IMRESS_CG_H
