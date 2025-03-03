#include <math.h>

#include <armadillo>

#include "finite_difference.h"

namespace finite_difference::armadillo {

    arma::vec central_difference(const optimiser::armadillo::FuncType& f,
        const arma::vec& x)
    {
        const auto n = x.size();
        arma::vec grad = arma::zeros(n);
        arma::vec x_forward = x;
        arma::vec x_backward = x;

        const auto eps_sqrt = sqrt(std::numeric_limits<double>::epsilon());

        for (int i = 0; i < x.size(); ++i) {
            const double h_i = eps_sqrt * std::max(1.0, abs(x[i]));
            x_forward[i] += 0.5 * h_i;
            x_backward[i] -= 0.5 * h_i;
            grad[i] = (f(x_forward) - f(x_backward)) / h_i;
            // Undo the ith displacement, as whole vector is used per
            // definition of grad[i]
            x_forward[i] -= 0.5 * h_i;
            x_backward[i] += 0.5 * h_i;
        }

        return grad;
    }

}