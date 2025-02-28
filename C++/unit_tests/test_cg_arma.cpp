#define CATCH_CONFIG_MAIN

#include <armadillo>
#include <catch2/catch_all.hpp>

#include "optimisers/cg_arma.h"
#include "test_functions/test_functions.h"

TEST_CASE("Linear Conjugate Gradient", "[optimiser]") {
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

    arma::vec x = optimiser::linear_conjugate_gradient(A, x0, b);

    REQUIRE(arma::approx_equal(x, x_ref, "reldiff", 1e-6));
}

TEST_CASE("Non-Linear Conjugate Gradient", "[optimiser]") {

    using namespace optimiser;

    const auto f = test_functions::rosenbrock;
    const auto df = test_functions::derivative_rosenbrock;
    const arma::vec x0 = {0.2, 0.5};
    const arma::vec x_ref = {1.0, 1.0};

    // Pre-set values for the optional parameters
    constexpr double reduction_factor = 0.5;
    constexpr double c1 = 1.e-4;

    // Create a lambda that fixes the extra parameters.
    LineSearchFunc line_search_func =
            [my_reduction = reduction_factor,
             my_c1 = c1]
             (const FuncType & f,
              const DerFuncType & df,
              const arma::vec & x,
              const arma::vec & direction,
              double alpha0) -> double {
        return optimiser::line_search_backtrack(f, df, x, direction, alpha0, my_reduction, my_c1);
    };

    const auto cg_result = non_linear_conjugate_gradient(f, df, x0, line_search_func, 2000, 1.e-5);

    REQUIRE(arma::approx_equal(cg_result.x, x_ref, "both", 1.e-5, 1.e-4));
    REQUIRE(cg_result.n_iter == 1269);
}
