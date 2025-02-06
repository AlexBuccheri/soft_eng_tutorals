#include <armadillo>

#include "cg.h"

namespace algorithms{

arma::vec linear_conjugate_gradient(const arma::mat & a,
                                    const arma::vec & x0,
                                    const arma::vec & b,
                                    const double tol,
                                    const int max_iter){
        arma::vec x = x0;
        arma::vec r = b - a * x;
        if (arma::norm(r) < tol) {
            return x;
        }
        arma::vec p = r;

        for (int k = 0; k < max_iter; ++k) {
            const arma::vec ap = a * p;
            const double alpha = arma::dot(r.t(), r) / arma::dot(p.t(), ap);
            x += alpha * p;
            const arma::vec r_next = r - alpha * ap;


            if (arma::norm(r_next) < tol) {
                return x;
            }

            const double beta = arma::dot(r_next.t(), r_next) / arma::dot(r.t(), r);
            p = r_next + beta * p;
            r = r_next;
        }

        return x;
    }


} // namespace algorithms