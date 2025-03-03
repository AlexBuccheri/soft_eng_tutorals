#define CATCH_CONFIG_MAIN

#include <catch2/catch_all.hpp>

#include "test_functions/test_functions.h"

#include "optimisers/cg_class.h"

TEST_CASE("BFGS Child Class", "[optimiser]") {
    using namespace optimiser::armadillo;

    const auto f = test_functions::rosenbrock;
    const auto df = test_functions::derivative_rosenbrock;

    const arma::vec x0 = {0.2, 0.5};
    // Result from the python implementation
    const arma::vec x_ref = {0.9999086, 0.99981565};

    // BFGS settings
    constexpr int max_iter = 3000;
    constexpr double tol=1.e-4;
    constexpr double reduction_factor = 0.5;
    constexpr double c1 = 1.e-4;

    cg::bfgs::BFGS bfgs{f, df, x0, max_iter, tol, reduction_factor, c1};
    const auto result = bfgs.minimize();

    REQUIRE(arma::approx_equal(result.x, x_ref, "both", 1.e-6, 1.e-5));
    // Does not quite converge with this number of max iterations
    REQUIRE(result.n_iter == max_iter);
}

