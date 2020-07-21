/* aponteeduardo@gmail.com */
/* dschoebi@biomed.ee.ethz.ch */
/* copyright (C) 2016 */

#ifndef DYNAMICSERP_H
#define DYNAMICSERP_H

#include "mpdcm.h"
#include <armadillo>
#include <matrix.h>
#include <mex.h>
#include <math.h>
#define DIM_STATES_ERP 9

namespace Host
{

struct DynamicsErp
{

    static
    arma::Mat< double >
    create_x(int dim_x)
    {
        arma::Mat< double> x(dim_x, DIM_STATES_ERP);
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
	arma::Mat< double >
	saturation(const arma::Mat< double > x,
            const arma::Mat< double > u,
            const mxArray *theta, const mxArray *ptheta);

	
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
		const arma::Mat< double > sx, 
        const arma::Mat< double > u, 
        const mxArray *theta, 
        const mxArray *ptheta,
        arma::Mat< double >& dx);


    static
    int
    dynamics_f_dxG(
        const arma::Mat< double > x,
		const arma::Mat< double > sx, 
        const arma::Mat< double > u, 
        const mxArray *theta, 
        const mxArray *ptheta,
        arma::Mat< double >& dx);

  
    static
    int
    dynamics_f_dxC(
        const arma::Mat< double > x, 
		const arma::Mat< double > sx, 
        const arma::Mat< double > u, 
        const mxArray *theta, 
        const mxArray *ptheta,
        arma::Mat< double >& dx);

 
};


}
#endif // DYNAMICSFMRI_H

