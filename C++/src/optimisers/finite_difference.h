#ifndef IMPRES_FINITE_DIFFERENCE_H
#define IMPRES_FINITE_DIFFERENCE_H

#include <armadillo>

#include "cg.h"

namespace finite_difference::armadillo {

    /**
     * @brief Computes the numerical gradient using central difference approximation.
     *
     * This function approximates the gradient of a scalar function using the central difference method.
     * It perturbs each dimension of the input vector slightly and estimates the derivative.
     * @note Central difference is not well-suited to oscillating functions like sin and cos, which
     * can erroneously yield zero derivative.
     *
     * @tparam FuncType The type of the function that evaluates f(x).
     * @param f The function for which the gradient is computed.
     * @param x The point at which the gradient is estimated.
     * @return The estimated gradient vector.
     */
    arma::vec central_difference(const optimiser::armadillo::FuncType& f, const arma::vec& x);

}

#endif
