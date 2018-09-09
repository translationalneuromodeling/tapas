/* aponteeduardo@gmail.com */
/* copyright (C) 2016 */

#ifndef UPDATEEULER_H
#define UPDATEEULER_H

#include "mpdcm.h"
#include "mpdcm.hcu"
#include <armadillo>
#include <matrix.h>
#include <mex.h>

namespace Host
{

template <class T_dynamics>
struct UpdateEuler 
{
 
    static
    int
    update_x(const arma::Mat< double> x, 
        const arma::Mat< double > u0,
        const arma::Mat< double > u1,
        const double dx,
        const mxArray *theta, 
        const mxArray *ptheta, arma::Mat< double > &nx);

};


template <class T_dynamics>
int
UpdateEuler<T_dynamics>::update_x(const arma::Mat< double > ox, 
    const arma::Mat< double > u0,
    const arma::Mat< double > u1, 
    const double dx, 
    const mxArray *theta, 
    const mxArray *ptheta, 
    arma::Mat< double >& nx)
{
    arma::Mat< double > tx = T_dynamics::dynamics_x(ox, u0, theta, ptheta);

    nx = ox + dx * tx;

}

}

#endif // UPDATEEULER_H


