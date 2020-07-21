/* aponteeduardo@gmail.com */
/* copyright (C) 2017 */

#ifndef DYNAMICSTRAPPENBERG_H
#define DYNAMICSTRAPPENBERG_H

#include "mpdcm.h"
#include <armadillo>
#include <matrix.h>
#include <mex.h>
#include <math.h>
#define DIM_STATES_TRAPP 1

namespace Host
{

struct DynamicsTrappenberg
{

    static
    arma::Mat< double >
    create_x(int dim_x)
    {
        arma::Mat< double > x(dim_x, DIM_STATES_TRAPP);
        return x;
    }

    static
    int
    initialize_x(
            const arma::Mat< double > u,
            const mxArray *theta,
            const mxArray *ptheta,
            arma::Mat< double >& nx);

    static
    int
    update_y(
            const arma::Mat< double > x,
            const arma::Mat< double > u,
            const mxArray *theta, 
            const mxArray *ptheta,
            arma::Mat< double >& y);

    static
    arma::Mat< double >
    sigma(
            const arma::Mat< double > x,
            const arma::Mat< double > u,
            const mxArray *theta,
            const mxArray *ptheta);

    static
    arma::Mat< double >
    dynamics_x(
            const arma::Mat< double > x,
            const arma::Mat< double > u,
            const mxArray *theta, 
            const mxArray *ptheta);

};


}

#endif // DYNAMICSTRAPPENBERG_H

