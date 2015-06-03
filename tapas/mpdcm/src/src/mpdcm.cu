//
// Author: Eduardo Aponte
// Copyright 2015 by Eduardo Aponte <aponteeduardo@gmail.com>
//
// Licensed under GNU General Public License 3.0 or later.
// Some rights reserved. See COPYING, AUTHORS.
//
// Revision log:
//


#include "mpdcm.hcu"

// ===========================================================================
// Allocate memory
// ===========================================================================

// Device alloc memory theta
__host__ 
void
dam_theta(kernpars pars, void **pd_theta, MPFLOAT **dd_theta)
{
    int nx = pars.nx;
    int nu = pars.nu;
    int nt = pars.nt;
    int nb = pars.nb;

    int tp;

    // Allocate memory for the structures

    HANDLE_ERROR( cudaMalloc( pd_theta, nt * nb * sizeof(ThetaDCM)));
    HANDLE_ERROR( cudaMemcpy( *pd_theta, pars.p_theta, 
        nt * nb * sizeof(ThetaDCM), cudaMemcpyHostToDevice ) );

    // Allocate memory for the matrices. It is assumed that the parameters
    // are in a block of contiguous memory with the for A, Bs, C, x0, epsilon,
    // Kappa, tau

    tp = nt * nb * 
        (nx * nx +      // A:
        nx * nx * nu +  // B's
        nx * nu +       // C
        nx * nx * nx +  // D
        nx + // kappa
        nx); // tau 

    HANDLE_ERROR( cudaMalloc( dd_theta, tp * sizeof(MPFLOAT) ) );
    HANDLE_ERROR( cudaMemcpy( (void *) *dd_theta, pars.d_theta, 
        tp * sizeof(MPFLOAT), cudaMemcpyHostToDevice ) );

}

// Device alloc memory ptheta
__host__ 
void 
dam_ptheta( kernpars pars, void **pd_ptheta, MPFLOAT **dd_ptheta)
{
    HANDLE_ERROR( cudaMalloc( pd_ptheta, sizeof(PThetaDCM)));
    HANDLE_ERROR( cudaMemcpy( *pd_ptheta, pars.p_ptheta, sizeof(PThetaDCM),
        cudaMemcpyHostToDevice ) ); 
}

__host__
void 
dam_y(kernpars pars, MPFLOAT **d_y)
{
    HANDLE_ERROR( cudaMalloc( (void **) d_y,
        pars.nx * pars.ny * pars.nt * pars.nb * sizeof(MPFLOAT) ) );
}

__host__
void
dam_u(kernpars pars, MPFLOAT **d_u)
{
    HANDLE_ERROR( cudaMalloc( (void**) d_u,
        pars.nt * pars.nu * pars.dp *  sizeof(MPFLOAT) ) );

    HANDLE_ERROR( cudaMemcpy( *d_u, pars.u, pars.nt * pars.nu * pars.dp * 
        sizeof(MPFLOAT), cudaMemcpyHostToDevice ) );
}



// ===========================================================================
// Host code
// ===========================================================================

int 
mpdcm_fmri( kernpars pars, klauncher launcher)
{
    // TODO Can be done in a much better way
    //MPFLOAT *x = pars.x;
    MPFLOAT *y = pars.y;

    MPFLOAT *d_x, *d_y, *d_u;
    void *pd_theta, *pd_ptheta;
    MPFLOAT *dd_theta, *dd_ptheta;
    unsigned int errcode[1], *d_errcode;

    kernpars d_pars;

    // x
    d_x = 0;

    // y
    dam_y(pars, &d_y);
  
    // u
    dam_u(pars, &d_u);

    // Error code
    HANDLE_ERROR( cudaMalloc( (void**) &d_errcode, 
        sizeof( unsigned int ) ) );

    // Theta 
    dam_theta(pars, &pd_theta, &dd_theta);
 
    // PThetaDCM
    dam_ptheta(pars, &pd_ptheta, &dd_ptheta); 

    d_pars.y = d_y;
    d_pars.u = d_u;
    d_pars.p_theta = (ThetaDCM *) pd_theta;
    d_pars.d_theta = dd_theta;
    d_pars.p_ptheta = (PThetaDCM *) pd_ptheta;
    d_pars.d_ptheta = dd_ptheta;

    d_pars.nx = pars.nx;
    d_pars.ny = pars.ny;
    d_pars.nu = pars.nu;
    d_pars.dp = pars.dp;
    d_pars.nt = pars.nt;
    d_pars.nb = pars.nb;


    // Launch the kernel
    (*launcher)(
        d_x, d_y, d_u, 
        pd_theta, dd_theta, 
        pd_ptheta, dd_ptheta,
        pars.nx, pars.ny, pars.nu, pars.dp, pars.nt, pars.nb, d_errcode);

    // Get y back

    HANDLE_ERROR( cudaMemcpy(y, d_y,
        pars.nx * pars.ny * pars.nt * pars.nb * sizeof(MPFLOAT),
        cudaMemcpyDeviceToHost) );

    HANDLE_ERROR( cudaMemcpy(errcode, d_errcode, sizeof( unsigned int ), 
        cudaMemcpyDeviceToHost) );

    if ( *errcode != 0 ) 
        printf( "Error %d in %s at line %d\n", *errcode, __FILE__, __LINE__ );

    // free the memory allocated on the GPU
    //HANDLE_ERROR( cudaFree( d_x ) );
    HANDLE_ERROR( cudaFree( d_y ) );
    HANDLE_ERROR( cudaFree( d_u ) );
    HANDLE_ERROR( cudaFree( d_errcode ) );

    HANDLE_ERROR( cudaFree( pd_theta ) );
    HANDLE_ERROR( cudaFree( dd_theta ) );

    if ( DIM_PTHETA ) HANDLE_ERROR( cudaFree( pd_ptheta ) );
    if ( DIM_DPTHETA ) HANDLE_ERROR( cudaFree( dd_ptheta ) );
    
    return 0; 
}

// =======================================================================
// Externals
// =======================================================================

extern "C"
int
mpdcm_fmri_euler( kernpars pars )
{
   return mpdcm_fmri( pars, &ldcm_euler);
};

extern "C"
int
mpdcm_fmri_kr4( kernpars pars )
{
    return mpdcm_fmri(pars, &ldcm_kr4);
}

extern "C"
int
mpdcm_fmri_bs( kernpars pars)
{
    return mpdcm_fmri( pars, &ldcm_bs);
};
