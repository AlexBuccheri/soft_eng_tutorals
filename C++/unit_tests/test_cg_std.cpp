#define CATCH_CONFIG_MAIN

#include <iostream>

#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp> // See https://github.com/catchorg/Catch2/blob/devel/docs/comparing-floating-point-numbers.md

#include "optimisers/cg_stl.h"

TEST_CASE("Vector utility operations", "[optimiser]") {

    using namespace Catch::Matchers;

    const std::vector<double> u{1., 2., 3., 4.};
    const std::vector<double> v{0.1, 0.2, 0.3, 0.4};

    SECTION( "Dot product of two vectors" ) {
        const double u_dot_v = optimiser::stl::dot(u, v);
        REQUIRE_THAT( u_dot_v, WithinAbs(3, 1.e-10) );
    }

    SECTION( "Norm of a vector" ) {
        const double norm_u = optimiser::stl::norm(u);
        REQUIRE_THAT( norm_u, WithinAbs(5.4772255751, 1.e-10) );
    }

    SECTION( "Subtract vector v from vector u" ) {
        const std::vector<double> ref_u_mins_v = {0.9, 1.8, 2.7, 3.6};
        const auto u_mins_v = optimiser::stl::vector_subtract(u, v);
        REQUIRE(u_mins_v.size() == 4);
        REQUIRE(u_mins_v == ref_u_mins_v);
    }

    SECTION( "Add vector v to vector u" ) {
        const std::vector<double> ref_u_plus_v = {1.1, 2.2, 3.3, 4.4};
        const auto u_plus_v = optimiser::stl::vector_add(u, v);
        REQUIRE(u_plus_v.size() == 4);
        REQUIRE(u_plus_v == ref_u_plus_v);
    }

    SECTION( "Scale a vector by a constant" ) {
        std::vector<double> ref_scaled_u = {-1.0, -2.0, -3.0, -4.0};
        const auto scaled_u = optimiser::stl::vector_scale(u, -1.0);
        REQUIRE(scaled_u.size() == 4);
        REQUIRE(scaled_u == ref_scaled_u);
    }

    SECTION( "Matrix-vector Product" ) {
        const optimiser::stl::matrix a {{1.0, 3.0,  4.0},
                                        {6.0, 1.0,  5.0},
                                        {2.0, 10.0, 1.0}};

        const std::vector<double> x {1.0, 2.0, 3.0};

        const auto product = optimiser::stl::matrix_vector_product(a, x);
        REQUIRE(product.size() == 3);
        const std::vector<double> ref_product {19.0, 23.0, 25.0};
        REQUIRE(product == ref_product);
    }

    SECTION( "Matrix-vector Product Inconsistent Dimensions" ) {
        const optimiser::stl::matrix a {{1.0, 3.0,  4.0},
                                        {6.0, 1.0,  5.0},
                                        {2.0, 10.0, 1.0}};

        const std::vector<double> x {1.0, 2.0};
        REQUIRE_THROWS_AS(optimiser::stl::matrix_vector_product(a, x), std::invalid_argument);
        REQUIRE_THROWS_WITH(optimiser::stl::matrix_vector_product(a, x),
                            ContainsSubstring( "Vector x must have length equal to" ));
    }
}


TEST_CASE("Linear Conjugate Gradient implemented with std::vector", "[optimiser]") {
    // SPD tri-diagonal matrix
    const optimiser::stl::matrix a {
            {4, 1, 0, 0, 0},
            {1, 4, 1, 0, 0},
            {0, 1, 4, 1, 0},
            {0, 0, 1, 4, 1},
            {0, 0, 0, 1, 4}};

    const std::vector<double> b = {1, 2, 3, 4, 5};
    const std::vector<double> x0 = {0.0, 0.0, 0.0, 0.0, 0.0};

    const std::vector<double> x_ref = {0.16794872, 0.32820513, 0.51923077, 0.59487179, 1.10128205};
    const auto x = optimiser::stl::linear_conjugate_gradient(a, x0, b);
    REQUIRE_THAT( x, Catch::Matchers::Approx(x_ref) );
}
