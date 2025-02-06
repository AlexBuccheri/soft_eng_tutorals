#ifndef IMPRES_CG_STL_H
#define IMPRES_CG_STL_H

namespace algorithms{

    // Definition of a matrix
    using matrix = std::vector<std::vector<double>>;

    // TODO document me
    std::vector<double> linear_conjugate_gradient(const matrix & a,
                                                  const std::vector<double> & x0,
                                                  const std::vector<double> & b,
                                                  double tol=1e-8,
                                                  int max_iter=1000);

}

#endif //IMPRES_CG_STL_H
