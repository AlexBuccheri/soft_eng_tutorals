/* Function, derivative and results definitions for each implementation.
 */
#ifndef IMPRES_CG_H
#define IMPRES_CG_H

#include <functional>

#include <armadillo>

namespace optimiser{

    namespace armadillo{
        // Definition of a mathematical function that evaluates an arma::vector to a float.
        using FuncType = std::function<double(const arma::vec&)>;

        // Definition of a derivative that accepts an arma::vector and returns an arma::vector of the same length.
        using DerFuncType = std::function<arma::vec(const arma::vec&)>;
    }

    namespace stl{
        // Definition of a vector
        using vector = std::vector<double>;

        // Definition of a matrix
        using matrix = std::vector<std::vector<double>>;

        // Definition of a mathematical function that evaluates a std::vector to a float.
        using FuncType = std::function<double(const vector&)>;

        // Definition of a derivative that accepts an arma::vector and returns an arma::vector of the same length.
        using DerFuncType = std::function<vector(const vector&)>;
    }

    // Store results of a conjugate gradient calculation.
    struct CGResult {
        arma::vec x;
        int n_iter;
    };

}

#endif
