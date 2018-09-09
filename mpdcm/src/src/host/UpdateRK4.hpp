/* aponteeduardo@gmail.com */
/* copyright (C) 2016 */

#ifndef UPDATERK4_H
#define UPDATERK4_H

#include "mpdcm.h"
#include <armadillo>
#include <matrix.h>
#include <mex.h>

namespace Host
{

template <class T_dynamics>
struct UpdateRK4
{

    static
    int
    update_x(const arma::Mat< double> x,
        const arma::Mat< double > u0,
        const arma::Mat< double > u1,
        const double dt,
        const mxArray *theta,
        const mxArray *ptheta, arma::Mat< double > &nx);

};


template <class T_dynamics>
int
UpdateRK4<T_dynamics>::update_x(const arma::Mat< double > x,
    const arma::Mat< double > u0,
    const arma::Mat< double > u1,
    const double dt,
    const mxArray *theta,
    const mxArray *ptheta,
    arma::Mat< double >& nx)
{

    arma::Mat< double > b = T_dynamics::dynamics_x(x, u0, theta, ptheta);
    arma::Mat< double > tu = 0.5 * (u0 + u1);

    nx.zeros();

    nx += 0.1666666666 * b;
    b = T_dynamics::dynamics_x(x + 0.5 * dt * b, tu, theta, ptheta);
    
    nx += 0.333333333 * b;
    b = T_dynamics::dynamics_x(x + 0.5 * dt * b, tu, theta, ptheta);

    nx += 0.333333333 * b;
    b = T_dynamics::dynamics_x(x + dt * b, u1, theta, ptheta);

    nx += 0.1666666666 * b;

    nx = x + dt * nx;

}






}

#endif // UPDATERK4_H

