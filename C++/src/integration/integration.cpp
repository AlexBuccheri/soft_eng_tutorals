#include <cassert>

#include <armadillo>

#include "integration.h"

namespace integration::armadillo {

    double trapezium(const FuncType1D &f, const arma::vec &x, const double dx){
        const arma::vec f_x = f(x);
        const double integral = arma::sum(f_x.subvec(1, x.size()-2)) + 0.5 * (f_x[0] + f_x[x.size()-1]);
        return integral * dx;
    }

    double trapezium(const FuncType1D &f, const Limits &limits, const double dx){
        const arma::vec x = arma::regspace(limits.start, dx, limits.end);
        return trapezium(f, x, dx);
    }

    double simpson(const FuncType1D &f, const Limits &limits, const int npoints){
        assert(npoints % 2 == 1);
        const arma::vec x = arma::linspace(limits.start, limits.end, npoints);
        return simpson(f, x, npoints);
    }

    double simpson(const FuncType1D &f, const arma::vec &x, const int npoints){
        // Assert npoints is odd, as we require the number of subintervals to be even
        assert(npoints % 2 == 1);
        const auto n_subintervals = npoints - 1;
        const double h = (x[npoints-1] - x[0]) / static_cast<double>(n_subintervals);
        const arma::vec f_x = f(x);
        const int half_n_subintervals = static_cast<int>(0.5 * n_subintervals);

        const arma::uvec odd_indices = arma::linspace<arma::uvec>(1, n_subintervals-1, half_n_subintervals);
        const arma::uvec even_indices = arma::linspace<arma::uvec>(2, n_subintervals-2, half_n_subintervals - 1);

        const double integral = f_x[0] + (4 * arma::sum(f_x.elem(odd_indices))) +
                (2 * arma::sum(f_x.elem(even_indices))) + f_x[npoints-1];
        return (h / 3) * integral;
    }
}
