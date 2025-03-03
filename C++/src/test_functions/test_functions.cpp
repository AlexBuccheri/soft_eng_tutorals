#include <armadillo>

#include "test_functions.h"

namespace test_functions {

    double rosenbrock(const arma::vec& x)
    {
        const auto n = x.size();
        return arma::sum(
                100.0 * arma::square( x.subvec(1, n-1) - arma::square(x.subvec(0, n-2)) )
                + arma::square(x.subvec(0, n-2) - 1)
        );
    }

    arma::vec derivative_rosenbrock(const arma::vec& x)
    {
        const auto n = x.size();
        arma::vec df = arma::zeros(n);

        df(0) = -400 * x(0) * (x(1) - (x(0) * x(0))) - 2 * (1 - x(0));
        df(n-1) = 200 * (x(n - 1) - (x(n - 2) * x(n - 2)));

        if(n > 2) {
            df.subvec(1, n - 2) = (
                    -400 * x.subvec(1, n - 2) % (x.subvec(2, n - 1) - (x.subvec(1, n - 2) * x.subvec(1, n - 2)))
                    - 2 * (1 - x.subvec(1, n - 2))
                    + 200 * (x.subvec(1, n - 2) - (x.subvec(0, n - 3) * x.subvec(0, n - 3))));
        }
        return df;
    }

}