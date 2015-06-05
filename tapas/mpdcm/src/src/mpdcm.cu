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
dam_theta(kernpars pars, kernpars *d_pars)
{
    int nx = pars.nx;
    int nu = pars.nu;
    int nt = pars.nt;
    int nb = pars.nb;

    int tp;
    int nB;
    int nD;

    // Allocate memory for the structures

    HANDLE_ERROR( cudaMalloc( &(d_pars->p_theta), nt * nb * sizeof(ThetaDCM)));
    HANDLE_ERROR( cudaMemcpy( d_pars->p_theta, pars.p_theta, 
        nt * nb * sizeof(ThetaDCM), cudaMemcpyHostToDevice ) );

    // Allocate memory for the matrices. It is assumed that the parameters
    // are in a block of contiguous memory with the for A, Bs, C, x0, epsilon,
    // Kappa, tau

    tp = nt * nb * 
        (nx * nx +      // A:
        nx * nu +       // C
        nx + // kappa
        nx); // tau 

    HANDLE_ERROR( cudaMalloc( &(d_pars->d_theta), tp * sizeof(MPFLOAT) ) );
    HANDLE_ERROR( cudaMemcpy( d_pars->d_theta, pars.d_theta, 
        tp * sizeof(MPFLOAT), cudaMemcpyHostToDevice ) );

    // Allocate memory for B
    
    nB = (nx + 1) * nu * nt * nb;
    
    HANDLE_ERROR( cudaMalloc( &(d_pars->jB), nB * sizeof( int )) );
    HANDLE_ERROR( cudaMemcpy( d_pars->jB, pars.sB->j, nB  * sizeof( int ),
        cudaMemcpyHostToDevice));
     
    nB = pars.sB->j[nB - 1];

    HANDLE_ERROR( cudaMalloc( &(d_pars->iB), nB * sizeof( int )) );
    HANDLE_ERROR( cudaMemcpy( d_pars->iB, pars.sB->i, nB  * sizeof( int ),
        cudaMemcpyHostToDevice));

    HANDLE_ERROR( cudaMalloc( &(d_pars->vB), nB * sizeof( MPFLOAT )) );
    HANDLE_ERROR( cudaMemcpy( d_pars->vB, pars.sB->v, 
        nB * sizeof( MPFLOAT ), cudaMemcpyHostToDevice));

    // Allocate memory for D

    nD = (nx + 1) * nx * nt * nb;
    
    HANDLE_ERROR( cudaMalloc( &(d_pars->jD), nD * sizeof( int )) );
    HANDLE_ERROR( cudaMemcpy( d_pars->jD, pars.sD->j, nD  * sizeof( int ),
        cudaMemcpyHostToDevice));
     
    nD = pars.sD->j[nD - 1];

    HANDLE_ERROR( cudaMalloc( &(d_pars->iD), nD * sizeof( int )) );
    HANDLE_ERROR( cudaMemcpy( d_pars->iD, pars.sD->i, nD  * sizeof( int ),
        cudaMemcpyHostToDevice));

    HANDLE_ERROR( cudaMalloc( &(d_pars->vD), nD * sizeof( MPFLOAT )) );
    HANDLE_ERROR( cudaMemcpy( d_pars->vD, pars.sD->v, 
        nD * sizeof( MPFLOAT ), cudaMemcpyHostToDevice));

}

// Device alloc memory ptheta
__host__ 
void 
dam_ptheta( kernpars pars, kernpars *d_pars)
{
    HANDLE_ERROR( cudaMalloc( &(d_pars->p_ptheta), sizeof(PThetaDCM)));
    HANDLE_ERROR( cudaMemcpy( d_pars->p_ptheta, pars.p_ptheta, 
        sizeof(PThetaDCM), cudaMemcpyHostToDevice ) ); 
}

__host__
void 
dam_y(kernpars pars, kernpars *d_pars)
{
    HANDLE_ERROR( cudaMalloc( &(d_pars->y),
        pars.nx * pars.ny * pars.nt * pars.nb * sizeof(MPFLOAT) ) );
}

__host__
void
dam_u(kernpars pars, kernpars *d_pars)
{
    HANDLE_ERROR( cudaMalloc( &(d_pars->u),
        pars.nt * pars.nu * pars.dp *  sizeof(MPFLOAT) ) );

    HANDLE_ERROR( cudaMemcpy( d_pars->u, pars.u, pars.nt * pars.nu * 
        pars.dp * sizeof(MPFLOAT), cudaMemcpyHostToDevice ) );
}



// ===========================================================================
// Host code
// ===========================================================================

int 
mpdcm_fmri( kernpars pars, klauncher launcher)
{
    MPFLOAT *y = pars.y;

    unsigned int errcode[1], *d_errcode;

    kernpars d_pars;

    // x
    //d_x = 0;

    // y
    dam_y(pars, &d_pars);
  
    // u
    dam_u(pars, &d_pars);

    // Error code
    HANDLE_ERROR( cudaMalloc( (void**) &d_errcode, 
        sizeof( unsigned int ) ) );

    // Theta 
    dam_theta(pars, &d_pars);
 
    // PThetaDCM
    dam_ptheta(pars, &d_pars); 
 
    d_pars.nx = pars.nx;
    d_pars.ny = pars.ny;
    d_pars.nu = pars.nu;
    d_pars.dp = pars.dp;
    d_pars.nt = pars.nt;
    d_pars.nb = pars.nb;


    // Launch the kernel
    (*launcher)( d_pars, d_errcode);

    // Get y back

    HANDLE_ERROR( cudaMemcpy(y, d_pars.y,
        pars.nx * pars.ny * pars.nt * pars.nb * sizeof(MPFLOAT),
        cudaMemcpyDeviceToHost) );

    HANDLE_ERROR( cudaMemcpy(errcode, d_errcode, sizeof( unsigned int ), 
        cudaMemcpyDeviceToHost) );

    if ( *errcode != 0 ) 
        printf( "Error %d in %s at line %d\n", *errcode, __FILE__, __LINE__ );

    // free the memory allocated on the GPU
    //HANDLE_ERROR( cudaFree( d_x ) );
    HANDLE_ERROR( cudaFree( d_pars.y ) );
    HANDLE_ERROR( cudaFree( d_pars.u ) );
    HANDLE_ERROR( cudaFree( d_errcode ) );

    HANDLE_ERROR( cudaFree( d_pars.p_theta ) );
    HANDLE_ERROR( cudaFree( d_pars.d_theta ) );

    if ( DIM_PTHETA ) HANDLE_ERROR( cudaFree( d_pars.p_ptheta ) );
    if ( DIM_DPTHETA ) HANDLE_ERROR( cudaFree( d_pars.d_ptheta ) );

    HANDLE_ERROR( cudaFree( d_pars.jB ) );
    HANDLE_ERROR( cudaFree( d_pars.iB ) );
    HANDLE_ERROR( cudaFree( d_pars.vB ) );

    HANDLE_ERROR( cudaFree( d_pars.jD ) );
    HANDLE_ERROR( cudaFree( d_pars.iD ) );
    HANDLE_ERROR( cudaFree( d_pars.vD ) );
   
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
}
;

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
