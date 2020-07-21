/* aponteeduardo@gmail.com */
/* copyright (C) 2016 */

#ifndef DYNAMICSFMRI_H
#define DYNAMICSFMRI_H

#include "mpdcm.h"
#include "mpdcm.hcu"
#include <armadillo>
#include <matrix.h>
#include <mex.h>
#include <math.h>
#include <gsl/gsl_cblas.h>

namespace Host
{

struct DynamicsFmri
{

    static
    arma::Mat< double >
    create_x(int dim_x)
    {
        arma::Mat< double> x(dim_x, 5);
        return x;
    } 

    static
    int
    initialize_x(const arma::Mat< double > u,
            const mxArray *theta, 
            const mxArray *ptheta,
            arma::Mat< double >& nx);

    static
    arma::Mat< double >
    dynamics_x(
            const arma::Mat< double > x,
            const arma::Mat< double > u,
            const mxArray *theta, 
            const mxArray *ptheta);
           
    static
    int
    update_y(const arma::Mat< double > x,
            const arma::Mat< double > u,
            const mxArray *theta, const mxArray *ptheta,
            arma::Mat< double >& y);

    static
    int
    dynamics_f_dx(
        const arma::Mat< double > x, 
        const arma::Mat< double > u, 
        const mxArray *theta, 
        const mxArray *ptheta,
        arma::Mat< double >& dx);

    static
    int
    dynamics_f_dxA(
        const arma::Mat< double > x, 
        const arma::Mat< double > u, 
        const mxArray *theta, 
        const mxArray *ptheta,
        arma::Mat< double >& dx);

    static
    int
    dynamics_f_dxB(
        const arma::Mat< double > x, 
        const arma::Mat< double > u, 
        const mxArray *theta, 
        const mxArray *ptheta,
        arma::Mat< double >& dx);
    
    static
    int
    dynamics_f_dxC(
        const arma::Mat< double > x, 
        const arma::Mat< double > u, 
        const mxArray *theta, 
        const mxArray *ptheta,
        arma::Mat< double >& dx);

    static
    int
    dynamics_f_dxD(
        const arma::Mat< double > x, 
        const arma::Mat< double > u, 
        const mxArray *theta, 
        const mxArray *ptheta,
        arma::Mat< double >& dx);

    static
    int
    dynamics_f_ds(
        const arma::Mat< double > x, 
        const arma::Mat< double > u, 
        const mxArray *theta, 
        const mxArray *ptheta,
        arma::Mat< double >& dx);
    static
    int
    dynamics_f_df(
        const arma::Mat< double > x, 
        const arma::Mat< double > u, 
        const mxArray *theta, 
        const mxArray *ptheta,
        arma::Mat< double >& dx);
    static
    int
    dynamics_f_dv(
        const arma::Mat< double > x, 
        const arma::Mat< double > u, 
        const mxArray *theta, 
        const mxArray *ptheta,
        arma::Mat< double >& dx);
    
    static
    int
    dynamics_f_dq(
        const arma::Mat< double > x, 
        const arma::Mat< double > u, 
        const mxArray *theta, 
        const mxArray *ptheta,
        arma::Mat< double >& dx);

    static
    int
    dynamics_g(
        const arma::Mat< double > x, 
        const arma::Mat< double > u, 
        const mxArray *theta, 
        const mxArray *ptheta,
        arma::Mat< double >& y);


};


}
#endif // DYNAMICSFMRI_H

