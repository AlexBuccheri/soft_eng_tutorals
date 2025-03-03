#define CATCH_CONFIG_MAIN

#include <armadillo>
#include <catch2/catch_all.hpp>

#include "derivatives/finite_difference.h"
#include "test_functions/test_functions.h"

TEST_CASE("Numeric vs analytic gradient for Rosenbrock function", "[test_functions]") {
    // Some x-value that is not the global minimum, (1, 1)
    const arma::vec x = {-1.2, 1.0};
    using namespace test_functions;
    const arma::vec grad_analytic = derivative_rosenbrock(x);
    const arma::vec grad_fd = finite_difference::armadillo::central_difference(rosenbrock, x);

    REQUIRE(arma::approx_equal(grad_analytic, grad_fd, "reldiff", 1e-6));
}
