#ifndef IMPRES_CG_STL_H
#define IMPRES_CG_STL_H

#include <vector>

#include "cg.h"

namespace optimiser::stl {
    
    double dot(const vector& u, const vector& v);

    double norm(const vector& v);
    
    vector matrix_vector_product(const matrix& a,
        const vector& x);

    vector vector_subtract(const vector& u,
        const vector& v);
    
    vector vector_add(const vector& u,
        const vector& v);
    
    vector vector_scale(const vector& u,
        double alpha);

    namespace cg::linear{
        vector linear_conjugate_gradient(const matrix& a,
                                         const vector& x0,
                                         const vector& b,
                                         double tol = 1e-8,
                                         int max_iter = 1000);
    }

    namespace cg::bfgs{
        // TODO Implement BFGS with std containers and operations
    }

}

#endif // IMPRES_CG_STL_H
