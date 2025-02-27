#ifndef IMPRES_CG_STL_H
#define IMPRES_CG_STL_H

#include <vector>

namespace optimiser::stl {

// Definition of a matrix
using matrix = std::vector<std::vector<double>>;

double dot(const std::vector<double>& u, const std::vector<double>& v);

double norm(const std::vector<double>& v);

std::vector<double> matrix_vector_product(const matrix& a,
    const std::vector<double>& x);

std::vector<double> linear_conjugate_gradient(const matrix& a,
    const std::vector<double>& x0,
    const std::vector<double>& b,
    double tol = 1e-8,
    int max_iter = 1000);

std::vector<double> vector_subtract(const std::vector<double>& u,
    const std::vector<double>& v);

std::vector<double> vector_add(const std::vector<double>& u,
    const std::vector<double>& v);

std::vector<double> vector_scale(const std::vector<double>& u,
    double alpha);

}

#endif // IMPRES_CG_STL_H
