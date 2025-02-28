#include <armadillo>

#include "cg_arma.h"

#include "cg_class.h"

namespace optimiser {

    BFGS::BFGS(const FuncType &f,
         const DerFuncType &df,
         const arma::vec &x0,
         const int max_iter,
         const double tol,
         const double reduction_factor,
         const double c1
    ):
            NonLinearCG(f, df, x0, max_iter, tol),
            reduction_factor(reduction_factor),
            c1(c1) {
        hess = arma::eye(x0.size(), x0.size());
    }

    arma::vec BFGS::initialise_search_direction(){
        return  -hess * g;
    }

    double BFGS::line_search(){
        return line_search_backtrack(f, df, x, d, alpha, reduction_factor, c1);;
    }

    arma::mat BFGS::update_hessian_or_coefficient(){
        const arma::vec s = x_next - x;
        const arma::vec y = g_next - g;
        return update_hessian(s, y, hess);
    }

    arma::vec BFGS::update_search_direction(){
        return -hess * g_next;
    }

}