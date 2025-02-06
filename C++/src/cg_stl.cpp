/*
 * Implementation of linear algebra routines using the STL and corresponding containers/
 */
#include <vector>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "cg_stl.h"

namespace algorithms {

    double dot(const std::vector<double> & u, const std::vector<double> & v){
        // Check size consistency - could be superfluous?
        if (u.size() != v.size()) {
            throw std::invalid_argument("Dimension mismatch in dot product.");
        }
        // See https://en.cppreference.com/w/cpp/algorithm/inner_product
        return std::inner_product(u.begin(), u.end(), v.begin(), 0);
    }

    double norm(const std::vector<double> & v) {
        return std::sqrt(dot(v, v));
    }

    // TODO(Alex). Look at the algorithms header, and at BLAS wrappers for C++
    std::vector<double> linear_conjugate_gradient(const matrix & a,
                                                  const std::vector<double> & x0,
                                                  const std::vector<double> & b,
                                                  const double tol,
                                                  const int max_iter){
        std::vector<double> x = x0;
        std::vector<double> r; // MV product, followed by subtraction
        if (norm(r) < tol) {
            return x;
        }
        std::vector<double> p = r;

        for (int k = 0; k < max_iter; ++k) {
            const std::vector<double> ap; // TODO Matrix-vector product
            const double alpha = dot(r, r) / dot(p, ap);
            // TODO x = x + alpha * p
            std::vector<double> r_next; // MV product, followed by subtraction

            if (norm(r_next) < tol) {
                return x;
            }

            const double beta = dot(r_next, r_next) / dot(r, r);
            // TODO Update p
            r = r_next;
        }

        return x;
    }

} // namespace algorithms