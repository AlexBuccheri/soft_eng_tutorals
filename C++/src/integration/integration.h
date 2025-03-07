#ifndef IMPRS_INTEGRATION_H
#define IMPRS_INTEGRATION_H

#include <functional>

#include <armadillo>

namespace integration::armadillo {

    /// @brief Limits of integration
    struct Limits {
        double start;
        double end;
    };

    /// @brief Definition of a mathematical function evaluates f(x) at each x.
    using FuncType1D = std::function<const arma::vec(const arma::vec&)>;

    /// @brief Integrate a 1D function using the trapezium rule, as
    /// defined by https://en.wikipedia.org/wiki/Trapezoidal_rule
    double trapezium(const FuncType1D &f, const arma::vec &x, double dx);

    double trapezium(const FuncType1D &f, const Limits &limits, double dx);

    /// @brief Composite Simpson's 1/3 rule
    /// defined by https://en.wikipedia.org/wiki/Simpson%27s_rule
    double simpson(const FuncType1D &f, const arma::vec &x, int npoints);

    double simpson(const FuncType1D &f, const Limits &limits, int npoints);
}

#endif
