#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "cg_stl.h"

namespace optimiser::stl {

    double dot(const vector& u, const vector& v)
    {
        // Check size consistency - could be superfluous?
        if (u.size() != v.size()) {
            throw std::invalid_argument("Dimension mismatch in dot product.");
        }
        // See https://en.cppreference.com/w/cpp/algorithm/inner_product
        return std::inner_product(u.begin(), u.end(), v.begin(), 0.0);
    }

    double norm(const vector& v)
    {
        return std::sqrt(dot(v, v));
    }

    vector matrix_vector_product(const matrix& a,
        const vector& x)
    {
        const uint n_row = a.size();
        const uint n_col = a[0].size();
        if (x.size() != n_row) {
            throw std::invalid_argument("Vector x must have length equal to the number of matrix rows");
        }

        vector b(n_row, 0.0);
        for (int i = 0; i < n_row; ++i) {
            for (int j = 0; j < n_col; ++j) {
                b[i] += a[i][j] * x[j];
            }
        }

        return b;
    }

    vector vector_subtract(const vector& u,
        const vector& v)
    {
        if (u.size() != v.size()) {
            throw std::invalid_argument("Dimension mismatch in dot product.");
        }

        vector u_minus_v(u.size());

        for (int i = 0; i < u.size(); ++i) {
            u_minus_v[i] = u[i] - v[i];
        }

        return u_minus_v;
    }

    vector vector_add(const vector& u,
        const vector& v)
    {
        if (u.size() != v.size()) {
            throw std::invalid_argument("Dimension mismatch in dot product.");
        }

        vector u_plus_v(u.size());

        for (int i = 0; i < u.size(); ++i) {
            u_plus_v[i] = u[i] + v[i];
        }

        return u_plus_v;
    }

    vector vector_scale(const vector& u,
        const double alpha)
    {
        vector u_scaled(u.size());
        for (int i = 0; i < u.size(); ++i) {
            u_scaled[i] = alpha * u[i];
        }

        return u_scaled;
    }

    namespace cg::linear{
        vector linear_conjugate_gradient(const matrix& a,
                                         const vector& x0,
                                         const vector& b,
                                         const double tol,
                                         const int max_iter)
        {
            vector x = x0;
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
    }

    namespace cg::bfgs{
        // TODO Implement BFGS with std containers and operations
    }

}
