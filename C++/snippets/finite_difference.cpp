#include "finite_difference.h"

#include <armadillo>

namespace algorithms::fd {

    EigenStates radial_schrodinger_eq(const int n_points, const double r_min, const double r_max){

        // Line element
        const double dr = (r_max - r_min) / static_cast<double>(n_points - 1);

        // Create a grid
        // TODO Check the start, end and number of points
        arma::vec r = arma::linspace(r_min, r_max, n_points);

        // Set singularity at r = 0 to the grid spacing
        r[0] = dr;

        // The KE operator in the central difference approximation is
        // 1 \frac{1}{2} {d^2}{dr^2} u(r) = (-0.5 / (dr)**2) * [u(r_{i-1} - 2 u(r_{i} + u(r_{i+1}]
        const auto coefficient = -0.5 / std::pow(dr, 2);
        const arma::vec stencil_weights = {1.0, -2.0, 1.0};

        // Set H according to the KE operator
        // Initialised with zeros as most elements never get touched: Sparse matrix
        arma::mat ham = arma::zeros(n_points, n_points);

        // First point, with one-sided difference
        ham(0, 0) = - 2 * coefficient;
        ham(0, 1) = coefficient;

        // Interior points For point i, the central difference stencil defines values points
        // i-1, i and i+1
        for (arma::uword  i = 1; i < n_points-1; ++i) {
            const arma::uvec indices = {i-1, i, i+1};
            ham.rows(indices) += coefficient * stencil_weights;
        }

        // Last point, with one-sided difference
        ham(n_points, n_points - 1) = coefficient;
        ham(n_points, n_points) = - 2 * coefficient;

        // Add Coulomb contribution, ignoring the singularity at r=0
        for (int i = 1; i < n_points; ++i) {
            ham(i, i) += 1.0 / r[i];
        }

        // Diagonalise. Approach one. std::move
        /*
        arma::vec eigval;
        arma::mat eigvec;
        arma::eig_sym(eigval, eigvec, ham);

        return EigenStates{std::move(eigval), std::move(eigvec)};
         */

        // Approach 2. Initialise obj and fill into its attributes
        auto es = EigenStates();
        arma::eig_sym(es.eignevalues, es.eigenvectors, ham);
        return es;
    }

}