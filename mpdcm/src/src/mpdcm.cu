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
dam_theta(
    void **theta, MPFLOAT **d_theta,
    void **pd_theta, MPFLOAT **dd_theta,
    int nx, int ny, int nu, int dp, int nt, int nb)
{

    int tp;

    // Allocate memory for the structures

    HANDLE_ERROR( cudaMalloc( pd_theta, nt * nb * sizeof(ThetaDCM)));
    HANDLE_ERROR( cudaMemcpy( *pd_theta, *theta, nt * nb * sizeof(ThetaDCM),
        cudaMemcpyHostToDevice ) );

    // Allocate memory for the matrices. It is assumed that the parameters
    // are in a block of contiguous memory with the for A, Bs, C, x0, epsilon,
    // Kappa, tau

    tp = nt * nb * 
        (nx * nx +      // A:
        nx * nx * nu +  // B's
        nx * nu +       // C
        nx * nx * nx +
        nx + // kappa
        nx); // tau 

    HANDLE_ERROR( cudaMalloc( dd_theta, tp * sizeof(MPFLOAT) ) );
    HANDLE_ERROR( cudaMemcpy( (void *) *dd_theta, (void *) *d_theta, 
        tp * sizeof(MPFLOAT), cudaMemcpyHostToDevice ) );

}

// Device alloc memory ptheta
__host__ 
void 
dam_ptheta(
    void **ptheta, MPFLOAT **d_ptheta, 
    void **pd_ptheta, MPFLOAT **dd_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb)
{

    HANDLE_ERROR( cudaMalloc( pd_ptheta, sizeof(PThetaDCM)));
    HANDLE_ERROR( cudaMemcpy( *pd_ptheta, *ptheta, sizeof(PThetaDCM),
        cudaMemcpyHostToDevice ) );
 
}

// ===========================================================================
// Host code
// ===========================================================================

extern "C"
int 
mpdcm_fmri( MPFLOAT *x, MPFLOAT *y, MPFLOAT *u,
    void *theta, MPFLOAT *d_theta,
    void *ptheta, MPFLOAT *d_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb,
    klauncher launcher)
{

    MPFLOAT *d_x, *d_y, *d_u;
    void *pd_theta, *pd_ptheta;
    MPFLOAT *dd_theta, *dd_ptheta;
    unsigned int errcode[1], *d_errcode;

    // x

    d_x = 0;

    // y

    HANDLE_ERROR( cudaMalloc( (void **) &d_y,
        nx * ny * nt * nb * sizeof(MPFLOAT) ) );

    // u

    HANDLE_ERROR( cudaMalloc( (void**) &d_u,
        nt * nu * dp *  sizeof(MPFLOAT) ) );

    HANDLE_ERROR( cudaMemcpy( d_u, u, nt * nu * dp * sizeof(MPFLOAT),
        cudaMemcpyHostToDevice ) );

    // Error code

    HANDLE_ERROR( cudaMalloc( (void**) &d_errcode, 
        sizeof( unsigned int ) ) );


    // Theta 
    dam_theta(
        &theta, &d_theta,
        &pd_theta, &dd_theta,
        nx, ny, nu, dp, nt, nb);
    
    // PThetaDCM

    dam_ptheta(
        &ptheta, &d_ptheta,
        &pd_ptheta, &dd_ptheta,
        nx, ny, nu, dp, nt, nb); 

    // Launch the kernel
    (*launcher)(
        d_x, d_y, d_u, 
        pd_theta, dd_theta, 
        pd_ptheta, dd_ptheta,
        nx, ny, nu, dp, nt, nb, d_errcode);

    // Get y back

    HANDLE_ERROR( cudaMemcpy(y, d_y,
        nx * ny * nt * nb * sizeof(MPFLOAT),
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
mpdcm_fmri_euler( MPFLOAT *x, MPFLOAT *y, MPFLOAT *u,
    void *theta, MPFLOAT *d_theta,
    void *ptheta, MPFLOAT *d_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb)
{
   int r = mpdcm_fmri(x, y, u,
        theta, d_theta,
        ptheta, d_ptheta, 
        nx, ny, nu, dp, nt, nb,
        &ldcm_euler);
    
    return r;
};

extern "C"
int
mpdcm_fmri_kr4( MPFLOAT *x, MPFLOAT *y, MPFLOAT *u,
    void *theta, MPFLOAT *d_theta,
    void *ptheta, MPFLOAT *d_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb)
{
    int r = mpdcm_fmri(x, y, u,
        theta, d_theta,
        ptheta, d_ptheta, 
        nx, ny, nu, dp, nt, nb,
        &ldcm_kr4);

    return r;
}

extern "C"
int
mpdcm_fmri_bs( MPFLOAT *x, MPFLOAT *y, MPFLOAT *u,
    void *theta, MPFLOAT *d_theta,
    void *ptheta, MPFLOAT *d_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb)
{
    int r = mpdcm_fmri(x, y, u,
        theta, d_theta,
        ptheta, d_ptheta, 
        nx, ny, nu, dp, nt, nb,
        &ldcm_bs);

    return r;
};
