#define CATCH_CONFIG_MAIN

#include <armadillo>
#include <catch2/catch_all.hpp>

#include "cg.h"

TEST_CASE("Linear Conjugate Gradient", "[algorithms]") {
    // SPD tri-diagonal matrix
    const arma::mat A = {
            {4, 1, 0, 0, 0},
            {1, 4, 1, 0, 0},
            {0, 1, 4, 1, 0},
            {0, 0, 1, 4, 1},
            {0, 0, 0, 1, 4}
    };
    const arma::vec b = {1, 2, 3, 4, 5};
    const arma::vec x0 = arma::zeros<arma::vec>(b.n_elem);
    const arma::vec x_ref = {0.16794872, 0.32820513, 0.51923077, 0.59487179, 1.10128205};

    arma::vec x = algorithms::linear_conjugate_gradient(A, x0, b);

    REQUIRE(arma::approx_equal(x, x_ref, "reldiff", 1e-6));
}
