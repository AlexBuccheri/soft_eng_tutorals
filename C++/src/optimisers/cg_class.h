#ifndef IMPRES_CG_CLASS_H
#define IMPRES_CG_CLASS_H

#include "cg.h"

#include "cg_base_class.h"

namespace optimiser::armadillo::cg{

    namespace nonlinear{
        // TODO Add child class to implement nonlinear CG
    }

    namespace bfgs{
        /**
         * @class BFGS
         * @brief Implements the Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm for nonlinear optimization.
         *
         * This class provides an implementation of the BFGS algorithm, a quasi-Newton method for solving nonlinear
         * optimization problems. It inherits from the NonLinearCG class and overrides its virtual functions to:
         * - Initialize the search direction (@ref initialise_search_direction).
         * - Perform a line search (@ref line_search).
         * - Update the Hessian approximation (@ref update_hessian_or_coefficient).
         * - Update the search direction based on the new Hessian (@ref update_search_direction).
         *
         * The BFGS algorithm uses an approximation of the inverse Hessian, which is initially set to the identity matrix.
         * Two parameters control the line search's behaviour:
         * - @b reduction_factor: A scaling factor used during the line search.
         * - @b c1: A constant used in the line search condition.
         *
         * @see NonLinearCG
         */
        class BFGS: public NonLinearCG{

        private:
            const double reduction_factor; ///< Scaling factor for reducing step size during line search.
            const double c1;               ///< Constant for the line search condition.

            /// Initializes the search direction.
            arma::vec initialise_search_direction();

            /// Performs a line search.
            double line_search();

            /// Updates the Hessian approximation or coefficient.
            arma::mat update_hessian_or_coefficient();

            /// Updates the search direction.
            arma::vec update_search_direction();

        public:
            /**
             * @brief Constructor for the BFGS optimizer.
             *
             * Initializes the BFGS algorithm with the provided function, its derivative, initial guess, maximum iterations,
             * tolerance, and algorithm-specific parameters.
             *
             * @see NonLinearCG
             */
            BFGS(const FuncType &f,
                 const DerFuncType &df,
                 const arma::vec &x0,
                 int max_iter,
                 double tol,
                 double reduction_factor=0.5,
                 double c1 = 1.e-4
            );

            // When you define a destructor for your derived class (even if it's just the default),
            // the base class destructor is automatically called after your derived destructor's body runs.
            /// Destructor
            ~BFGS() = default;
        };
    }

}

#endif //IMPRES_CG_CLASS_H
