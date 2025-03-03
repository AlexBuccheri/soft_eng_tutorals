#define CATCH_CONFIG_MAIN

#include <armadillo>
#include <catch2/catch_all.hpp>

#include "optimisers/cg.h" // FuncType, DerFuncType
#include "test_functions/test_functions.h" // rosenbrock, derivative_rosenbrock
#include "optimisers/line_search.h" // LineSearchParam
#include "optimisers/cg_arma.h" // Traits

#include "optimisers/cg_template.h" // non_linear_conjugate_gradient

TEST_CASE("Templated non-linear CG, implementing BFGS", "[optimiser]") {

    // Should be equivalent to the BFGS test in test_cg_class.cpp
    using namespace optimiser;
    using Traits = optimiser::armadillo::cg::bfgs::Traits;

    const armadillo::FuncType f = test_functions::rosenbrock;
    const armadillo::DerFuncType df = test_functions::derivative_rosenbrock;
    const arma::vec x0 = {0.2, 0.5};

    // Result from the python implementation
    const arma::vec x_ref = {0.9999086, 0.99981565};

    constexpr int max_iter = 3000;
    constexpr double tol=1.e-4;
    const optimiser::armadillo::LineSearchParam ls_params{};

    const auto result = templated::non_linear_conjugate_gradient<Traits>(f, df, x0, max_iter, tol, ls_params);
    REQUIRE(arma::approx_equal(result.x, x_ref, "both", 1.e-6, 1.e-5));
    // Does not quite converge with this number of max iterations
    REQUIRE(result.n_iter == max_iter);
}
