#include <armadillo>
#include <catch2/catch_all.hpp>

#include "integration/integration.h"

TEST_CASE("1D Trapezium Rule", "[integration]") {
    using namespace integration::armadillo;
    using namespace Catch::Matchers;

    SECTION( "Integrate x^2 between 0 and 10" ) {
        const Limits limits{0, 10};
        constexpr double dx = 0.001;
        const arma::vec x = arma::regspace(limits.start, dx, limits.end);
        const auto func_x_squared = [](const arma::vec &x) -> arma::vec{
            return arma::square(x);
        };

        // Integral = [x^3/3]^b_a = [10^3/3 - 0]
        const auto integral = trapezium(func_x_squared, x, dx);
        REQUIRE_THAT( integral, WithinAbs(333.333333, 2.1e-6) );
    }

    SECTION( "Integrate x^2 between 0 and 10 using second function signature" ) {
        const Limits limits{0, 10};
        constexpr double dx = 0.001;
        const auto func_x_squared = [](const arma::vec &x) -> arma::vec{
            return arma::square(x);
        };
        // Integral = [x^3/3]^b_a = [10^3/3 - 0]
        const auto integral = trapezium(func_x_squared, limits, dx);
        REQUIRE_THAT(integral, WithinAbs(333.333333, 2.1e-6));
    }
}

TEST_CASE("1D Simpson's Rule", "[integration]") {
    using namespace integration::armadillo;
    using namespace Catch::Matchers;

    SECTION( "Integrate x^2 between 0 and 10" ) {
        const Limits limits{0, 10};
        constexpr int npoints = 10001;
        const auto func_x_squared = [](const arma::vec &x) -> arma::vec{
            return arma::square(x);
        };
        // Integral = [x^3/3]^b_a = [10^3/3 - 0]
        const auto integral = simpson(func_x_squared, limits, npoints);
        REQUIRE_THAT(integral, WithinAbs(333.333333, 3.5e-7));
        // One could compute the expected accuracy and compare to what we get
    }

}



