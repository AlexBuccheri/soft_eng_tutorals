/*
 * Methods of the class defined in "cg_base_class.h"
 */
#include "cg_base_class.h"

namespace optimiser::armadillo::cg{

    CGResult NonLinearCG::minimize() {
        d = initialise_search_direction();

        for (int k = 0; k < max_iter; ++k) {
            // Current step
            iteration = k;

            // Compute new step length
            alpha = line_search();

            // Compute new variable
            x_next = x + alpha * d;

            // Compute new gradient
            g_next = df(x_next);

            // Update coefficient or approx inv Hessian
            hess = update_hessian_or_coefficient();

            if (arma::norm(g_next) <= tol) {
                return CGResult{x, k};
            }

            // Update quantities
            d = update_search_direction();
            x = x_next;
            g = g_next;
        }

        return CGResult{x, max_iter};
    };

}