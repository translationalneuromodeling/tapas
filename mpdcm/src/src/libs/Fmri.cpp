/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include "mpdcm.hcu"
#include "cuda/fmri_ext.hcu"
#include "Fmri.hpp"
#include "utilities.hpp"

#include <iostream>

using namespace std;

namespace Host
{

Fmri::Fmri()
{

}

Fmri::~Fmri()
{

}


int
Fmri::init_y(const DataArray *u, const ThetaFmriArray *theta,
    const PThetaFmri *ptheta,  DataArray *y)
{
    y->nc = theta->nc;
    y->nr = theta->nr;

    y->nic = theta->data_host->dim_x;
    y->nir = ceil(u->nir * ptheta->dyu);

    return 0;
}

int
Fmri::transverse_ptheta(const mxArray *optheta, PThetaFmri *pt)
{
    pt->dt = (MPFLOAT ) *mxGetPr(mxGetField(optheta, 0, "dt"));
    pt->dyu = (MPFLOAT ) *mxGetPr(mxGetField(optheta, 0, "dyu"));
    // Precompute this value for efficency in cuda
    pt->de = pt->dt * mxGetPr(mxGetField(optheta, 0, "udt"))[0];
    pt->mode = 'f';
} 


int
Fmri::transverse_theta(const mxArray *ot, ThetaFmriArray *theta)
{
    unsigned int i, o, nx, nu;
    const mwSize *st = mxGetDimensions(ot);
    mxArray *ttheta;

    theta->nc = st[0];
    theta->nr = st[1];

    ttheta = mxGetCell(ot, 0);

    nx = (int ) *mxGetPr(mxGetField(ttheta, 0, "dim_x"));
    nu = (int ) *mxGetPr(mxGetField(ttheta, 0, "dim_u"));

    
    theta->linear.nc = theta->nc;
    theta->linear.nr = theta->nr;

    theta->linear.nic = nx;
    theta->linear.nir = (nx + nu + 1 + 1);  // A, C, Kappa, tau
   
    return 0;
}

int
Fmri::linearize_theta(const mxArray *ot, ThetaFmriArray *theta)
{
    int i, nu, nx;

    for (i = 0 ; i < theta->nc * theta->nr ; i++)
    {
        DataArray a[1];
        linearize_theta_fields(mxGetCell(ot, i), theta->data_host + i);
        
        
        nx = (theta->data_host + i)->dim_x;
        nu = (theta->data_host + i)->dim_u;

        a->data_host = (theta->linear).data_host + i * nx * (nx + 
            nu + 1 + 1); // A, C, Kappa, tau 
        
        linearize_theta_linear(mxGetCell(ot, i), a);  
    }    

	return 0;
}


int
Fmri::linearize_theta_fields(const mxArray *ttheta, ThetaFmri *itheta)
{

    
    itheta->dim_x = (int ) *mxGetPr(mxGetField(ttheta, 0, "dim_x"));
    itheta->dim_u = (int ) *mxGetPr(mxGetField(ttheta, 0, "dim_u"));

    
    itheta->fA = (int ) *mxGetPr(mxGetField(ttheta, 0, "fA")) ?
            MF_TRUE : MF_FALSE;
    itheta->fB = (int ) *mxGetPr(mxGetField(ttheta, 0, "fB")) ? 
            MF_TRUE : MF_FALSE;
    itheta->fC = (int ) *mxGetPr(mxGetField(ttheta, 0, "fC")) ? 
            MF_TRUE : MF_FALSE;
    itheta->fD = (int ) *mxGetPr(mxGetField(ttheta, 0, "fD")) ? 
            MF_TRUE : MF_FALSE;

    itheta->V0 = (MPFLOAT ) *mxGetPr(mxGetField(ttheta, 0, "V0"));
    itheta->E0 = (MPFLOAT ) *mxGetPr(mxGetField(ttheta, 0, "E0"));

    //For efficiency reasons some values are prepared.

    itheta->ln1_E0 = log(1 - itheta->E0);
    itheta->lnE0 = log(itheta->E0);
    itheta->k1 = (MPFLOAT ) *mxGetPr(mxGetField(ttheta, 0, "k1")) *
        itheta->V0;
    itheta->k2 = (MPFLOAT ) *mxGetPr(mxGetField(ttheta, 0, "k2")) * 
        itheta->V0;
    itheta->k3 = (MPFLOAT ) *mxGetPr(mxGetField(ttheta, 0, "k3")) * 
        itheta->V0;

    itheta->alpha = (MPFLOAT ) *mxGetPr(mxGetField(ttheta, 0, "alpha"));
    itheta->alpha = 1/itheta->alpha - 1;
    itheta->gamma = (MPFLOAT ) *mxGetPr(mxGetField(ttheta, 0, "gamma"));
    
    return 0;
}

int
Fmri::linearize_theta_linear(const mxArray *ttheta, DataArray *a)
{
    int i, k, nx, nu;
    double *ta;
    MPFLOAT *dtheta = a->data_host;

    nx = (int ) *mxGetPr(mxGetField(ttheta, 0, "dim_x"));
    nu = (int ) *mxGetPr(mxGetField(ttheta, 0, "dim_u"));
    

    i = nx * nx;
    ta = mxGetPr(mxGetField(ttheta, 0, "A"));
    for (k = 0; k < i; k++)
        dtheta[k] = (MPFLOAT ) ta[k];
    dtheta += i;

    i = nx * nu;
    ta = mxGetPr(mxGetField(ttheta, 0, "C"));
    for (k = 0; k < i; k++)
        dtheta[k] = (MPFLOAT ) ta[k];
    dtheta += i;

    i = nx;
    ta = mxGetPr(mxGetField(ttheta, 0, "K"));
    for (k = 0; k < i; k++)
        dtheta[k] = (MPFLOAT ) ta[k];
    dtheta += i; 

    i = nx;
    ta = mxGetPr(mxGetField(ttheta, 0, "tau"));
    for (k = 0; k < i; k++)
        dtheta[k] = (MPFLOAT ) ta[k];

    return 0;
}

} // Host
