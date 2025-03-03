#include <armadillo>

#include "line_search.h"

namespace optimiser::armadillo{

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

}