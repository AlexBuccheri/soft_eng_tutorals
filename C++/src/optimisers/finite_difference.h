#ifndef IMPRES_FINITE_DIFFERENCE_H
#define IMPRES_FINITE_DIFFERENCE_H

#include <cmath>
#include <functional>

#include <armadillo>

namespace finite_difference {

// TODO Alex. These are defined here and in cg.h Should move to a single place
// Define a mathematical function that evaluates a vector to a float.
using FuncType = std::function<double(const arma::vec&)>;

// Define a derivative that accepts a vector and returns a vector of the same length.
using DerFuncType = std::function<arma::vec(const arma::vec&)>;

arma::vec central_difference(const FuncType& f, const arma::vec& x);

}

#endif // IMPRES_FINITE_DIFFERENCE_H
