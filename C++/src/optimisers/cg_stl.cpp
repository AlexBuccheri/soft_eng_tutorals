/*
 * Implementation of linear algebra routines using the STL and corresponding containers/
 */
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "cg_stl.h"

namespace optimiser::stl {

double dot(const std::vector<double>& u, const std::vector<double>& v)
{
    // Check size consistency - could be superfluous?
    if (u.size() != v.size()) {
        throw std::invalid_argument("Dimension mismatch in dot product.");
    }
    // See https://en.cppreference.com/w/cpp/algorithm/inner_product
    return std::inner_product(u.begin(), u.end(), v.begin(), 0.0);
}

double norm(const std::vector<double>& v)
{
    return std::sqrt(dot(v, v));
}

std::vector<double> matrix_vector_product(const matrix& a,
    const std::vector<double>& x)
{
    const uint n_row = a.size();
    const uint n_col = a[0].size();
    if (x.size() != n_row) {
        throw std::invalid_argument("Vector x must have length equal to the number of matrix rows");
    }

    std::vector<double> b(n_row, 0.0);
    for (int i = 0; i < n_row; ++i) {
        for (int j = 0; j < n_col; ++j) {
            b[i] += a[i][j] * x[j];
        }
    }

    return b;
}

std::vector<double> vector_subtract(const std::vector<double>& u,
    const std::vector<double>& v)
{
    if (u.size() != v.size()) {
        throw std::invalid_argument("Dimension mismatch in dot product.");
    }

    std::vector<double> u_minus_v(u.size());

    for (int i = 0; i < u.size(); ++i) {
        u_minus_v[i] = u[i] - v[i];
    }

    return u_minus_v;
}

std::vector<double> vector_add(const std::vector<double>& u,
    const std::vector<double>& v)
{
    if (u.size() != v.size()) {
        throw std::invalid_argument("Dimension mismatch in dot product.");
    }

    std::vector<double> u_plus_v(u.size());

    for (int i = 0; i < u.size(); ++i) {
        u_plus_v[i] = u[i] + v[i];
    }

    return u_plus_v;
}

std::vector<double> vector_scale(const std::vector<double>& u,
    const double alpha)
{
    std::vector<double> u_scaled(u.size());
    for (int i = 0; i < u.size(); ++i) {
        u_scaled[i] = alpha * u[i];
    }

    return u_scaled;
}

// TODO(Alex). Look at BLAS wrappers for C++
std::vector<double> linear_conjugate_gradient(const matrix& a,
    const std::vector<double>& x0,
    const std::vector<double>& b,
    const double tol,
    const int max_iter)
{
    std::vector<double> x = x0;
    auto r = vector_subtract(b, matrix_vector_product(a, x));
    if (norm(r) < tol) {
        return x;
    }
    auto p = r;

    for (int k = 0; k < max_iter; ++k) {
        const auto ap = matrix_vector_product(a, p);
        const double alpha = dot(r, r) / dot(p, ap);
        const auto alpha_times_p = vector_scale(p, alpha);
        x = vector_add(x, alpha_times_p);
        const auto r_next = vector_subtract(r, vector_scale(ap, alpha));

        if (norm(r_next) < tol) {
            return x;
        }

        const double beta = dot(r_next, r_next) / dot(r, r);
        p = vector_add(r_next, vector_scale(p, beta));
        r = r_next;
    }

    return x;
}

} // namespace optimiser::stl
