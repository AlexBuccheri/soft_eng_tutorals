#ifndef IMRESS_CG_H
#define IMRESS_CG_H

namespace algorithms {


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
    arma::vec linear_conjugate_gradient(const arma::mat &a,
                                        const arma::vec &x0,
                                        const arma::vec &b,
                                        double tol = 1e-8,
                                        int max_iter = 1000);
}

#endif //IMRESS_CG_H
