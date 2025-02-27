#ifndef IMRESS_FINITE_DIFFERENCE_H
#define IMRESS_FINITE_DIFFERENCE_H

#include <armadillo>

namespace algorithms::fd{

    // Results container from diagonalisation
    struct EigenStates{
        arma::vec eignevalues;
        arma::mat eigenvectors;
    };

    EigenStates radial_schrodinger_eq(int n_points, double r_min, double r_max);

}

#endif //IMRESS_FINITE_DIFFERENCE_H
