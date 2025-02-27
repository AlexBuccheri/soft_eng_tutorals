#include <armadillo>

#include "cg.h"

namespace optimiser {

arma::vec linear_conjugate_gradient(const arma::mat& a,
    const arma::vec& x0,
    const arma::vec& b,
    const double tol,
    const int max_iter)
{
    arma::vec x = x0;
    arma::vec r = b - a * x;
    if (arma::norm(r) < tol) {
        return x;
    }
    arma::vec p = r;

    for (int k = 0; k < max_iter; ++k) {
        const arma::vec ap = a * p;
        const double alpha = arma::dot(r, r) / arma::dot(p, ap);
        x += alpha * p;
        const arma::vec r_next = r - alpha * ap;

        if (arma::norm(r_next) < tol) {
            return x;
        }

        const double beta = arma::dot(r_next, r_next) / arma::dot(r, r);
        p = r_next + beta * p;
        r = r_next;
    }

    return x;
}

double line_search_backtrack(const FuncType& f,
    const DerFuncType& df,
    const arma::vec& x,
    const arma::vec& direction,
    const double alpha0,
    const double reduction_factor,
    const double c1)
{

    if (0.0 < reduction_factor < 1.0) {
        throw std::invalid_argument("Require 0.0 < reduction_factor < 1.0");
    }
    if (0.0 < c1 < 1.0) {
        throw std::invalid_argument("Require 0.0 < c1 < 1.0");
    }

    // Initialise
    const double g_dot_d = arma::dot(df(x), direction);
    const double f_x = f(x);

    auto alpha = alpha0;
    arma::vec x_increment = x + alpha * direction;

    while (f(x_increment) > f_x + (c1 * alpha * g_dot_d)) {
        alpha *= reduction_factor;
        x_increment = x + alpha * direction;
    }

    return alpha;
}

double fletcher_reeves_coefficient(const arma::vec& grad, const arma::vec& grad_next)
{
    return arma::dot(grad_next, grad_next) / arma::dot(grad, grad);
}

CGResult non_linear_conjugate_gradient(const FuncType& f,
    const DerFuncType& df,
    const arma::vec& x0,
    LineSearchFunc& line_search_func,
    const int max_iter,
    const double tol)
{
    // Initialise variable
    arma::vec x = x0;
    // Initialise gradient
    arma::vec grad = df(x);
    // Initialise search direction
    arma::vec direction = -grad;
    // Initialise step size
    double alpha = 1.0;

    for (int k = 0; k < max_iter; ++k) {
        // Line search
        alpha = line_search_func(f, df, x, direction, alpha);

        // Update variable
        x += alpha * direction;

        // Compute new gradient
        const arma::vec grad_next = df(x);

        // Compute conjugate coefficient
        const double beta = fletcher_reeves_coefficient(grad, grad_next);

        // Check convergence
        if (arma::norm(grad_next) <= tol) {
            return { x, k };
        }

        // Update search direction
        grad = grad_next;
        direction = -grad + beta * direction;
    }

    return { x, max_iter };
}

} // namespace optimiser
