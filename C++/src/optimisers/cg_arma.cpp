#include <assert.h>

#include <armadillo>

#include "line_search.h"

#include "cg_arma.h"

namespace optimiser::armadillo::cg {

    namespace linear {
        CGResult linear_conjugate_gradient(const arma::mat &a,
                                            const arma::vec &x0,
                                            const arma::vec &b,
                                            const double tol,
                                            const int max_iter) {
            arma::vec x = x0;
            arma::vec r = b - a * x;
            if (arma::norm(r) < tol) {
                return CGResult{x, 0};
            }
            arma::vec p = r;

            for (int k = 0; k < max_iter; ++k) {
                const arma::vec ap = a * p;
                const double alpha = arma::dot(r, r) / arma::dot(p, ap);
                x += alpha * p;
                const arma::vec r_next = r - alpha * ap;

                if (arma::norm(r_next) < tol) {
                    return CGResult{x, k};
                }

                const double beta = arma::dot(r_next, r_next) / arma::dot(r, r);
                p = r_next + beta * p;
                r = r_next;
            }

            return CGResult{x, max_iter};
        }
    } // linear

    namespace nonlinear {
        double fletcher_reeves_coefficient(const arma::vec& grad, const arma::vec& grad_next)
        {
            return arma::dot(grad_next, grad_next) / arma::dot(grad, grad);
        }

        CGResult non_linear_conjugate_gradient(const FuncType &f,
                                               const DerFuncType &df,
                                               const arma::vec &x0,
                                               LineSearchFunc &line_search_func,
                                               const int max_iter,
                                               const double tol) {
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
                    return {x, k};
                }

                // Update search direction
                grad = grad_next;
                direction = -grad + beta * direction;
            }

            return {x, max_iter};
        }
    } // nonlinear

    namespace bfgs {
        arma::mat init_hessian_or_coefficient() {
            return arma::mat{};
        }

        arma::mat update_hessian(const arma::vec &s,
                                 const arma::vec &y,
                                 const arma::mat &hess){
            const auto n = s.size();
            assert(y.size() == n);
            assert(hess.n_cols == n);
            assert(hess.n_rows == n);

            const double p = 1.0 / arma::dot(y, s);
            const arma::mat identity = arma::eye(n, n);
            const arma::rowvec s_t = s.t();

            return (identity - p * s * y.t()) * hess * (identity - p * y * s_t) + (p * s * s_t);;
        }
    } // bfgs

}
