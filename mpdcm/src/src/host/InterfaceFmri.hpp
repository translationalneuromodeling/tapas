/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */


#ifndef INTERFACEFMRI_H
#define INTERFACEFMRI_H

#include "mpdcm.h"
#include <iostream>
#include <armadillo>
#include <matrix.h>
#include "host/InterfaceHost.hpp"

#ifdef HAVE_OMP_H
#include <omp.h>
#endif

namespace Host
{

template < class T_Integrator >
class InterfaceFmri : public InterfaceHost
{
    public:

        std::vector< arma::Mat< double > >
        init_y(const mxArray *u, const mxArray *theta, const mxArray *ptheta); 
        int
        integrate(const std::vector< arma::Mat< double > > u,
            const mxArray *theta,
            const mxArray *ptheta,
            std::vector< arma::Mat< double > >& y);

};

template < class T_Integrator >
std::vector< arma::Mat< double > > 
InterfaceFmri< T_Integrator >::init_y(const mxArray *u, const mxArray *theta, 
    const mxArray *ptheta)
{
    const mwSize *st = mxGetDimensions(theta);
    int dim_x = (size_t ) *mxGetPr(mxGetField(mxGetCell(theta, 0), 
        0, "dim_x"));
    double dyu = *mxGetPr(mxGetField(ptheta, 0, "dyu"));
    int nu = mxGetDimensions(mxGetCell(u, 0))[1];    
    int nr = dim_x, nc = ceil(nu * dyu);
    int i;

    std::vector< arma::Mat< double > > vmats;
    vmats.reserve(st[0] * st[1]);

    for (i = 0; i < st[0] * st[1]; i++)
    {
        arma::Mat< double > tmat;
        tmat = arma::zeros< arma::Mat< double > >(nr, nc);
        vmats.push_back(tmat);
    }

    return vmats;
}

template < class T_Integrator >
int
InterfaceFmri< T_Integrator>::integrate(
    const std::vector< arma::Mat< double > > u,
    const mxArray *theta,
    const mxArray *ptheta,
    std::vector< arma::Mat< double > >& y)
{
    int i;
    #pragma omp parallel for 
    for (i = 0; i < y.size(); i++)
    {

        const T_Integrator integrator(
                (int ) y[i].n_rows,
                (int ) ceil(1/mxGetPr(mxGetField(ptheta, 0, "dt"))[0]),
                (double ) ((*mxGetPr(mxGetField(ptheta, 0, "dt")))
                * (*mxGetPr(mxGetField(ptheta, 0, "udt")))));

        integrator.integrate(u[i % u.size()], mxGetCell(theta, i),
            ptheta, y[i]);
    }

    return 0;
}


} // Host

#endif

